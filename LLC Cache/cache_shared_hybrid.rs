use std::{
    io::{BufRead, BufReader, BufWriter, Write},
    str::FromStr,
};

use cf_qemu_post::{
    log_parser::{self},
    memory_access::{MemRecord, MemoryAccess},
};
use clap::Parser;

#[derive(Debug)]
pub struct Cache {
    block_size: usize, // in bytes
    sets: Vec<CacheSet>,
}

/// LRFU (Least Recently/Frequently Used) replacement policy per set.
///
/// Reference: D. Lee, J. Choi, J.-H. Kim, S. H. Noh, S. L. Min, Y. Cho,
///            and C. S. Kim, "LRFU: A Spectrum of Policies that Subsumes the
///            Least Recently Used and Least Frequently Used Policies,"
///            IEEE Trans. Computers, vol. 50, no. 12, pp. 1352–1361, Dec. 2001.
///
/// Weighting function: F(x) = 2^(-λx), with λ = LRFU_LAMBDA.
///   λ → 0 : F(x) → 1 for all x → pure LFU.
///   λ → ∞ : F(x) → 0 for x > 0  → pure LRU.
///   0 < λ < 1 : balanced hybrid of recency and frequency.
const LRFU_LAMBDA: f64 = 0.5;
const LRFU_BASE: f64 = 2.0;

#[derive(Debug)]
struct CacheSet {
    // Each cache line stores an optional tag (here, a u64 representing the block address)
    lines: Vec<Option<u64>>,
    // LRFU: Combined Recency and Frequency value per line
    crf: Vec<f64>,
    // LRFU: timestamp of last access per line
    last_access_time: Vec<u64>,
    // Monotonic per-set logical clock
    time_counter: u64,
}

impl CacheSet {
    pub fn new(associativity: usize) -> Self {
        CacheSet {
            lines: vec![None; associativity],
            crf: vec![0.0; associativity],
            last_access_time: vec![0; associativity],
            time_counter: 0,
        }
    }

    /// Compute the LRFU weighting function F(delta) = 2^(-λ * delta).
    #[inline]
    fn weight(delta: u64) -> f64 {
        LRFU_BASE.powf(-LRFU_LAMBDA * delta as f64)
    }

    /// Update LRFU metadata on a hit: CRF(b) = 1 + F(Δ) · CRF_prev(b).
    fn lrfu_hit(&mut self, pos: usize) {
        let delta = self.time_counter - self.last_access_time[pos];
        self.crf[pos] = 1.0 + Self::weight(delta) * self.crf[pos];
        self.last_access_time[pos] = self.time_counter;
        self.time_counter += 1;
    }

    /// Initialise LRFU metadata for a newly installed block: CRF = 1.0.
    fn lrfu_install(&mut self, pos: usize) {
        self.crf[pos] = 1.0;
        self.last_access_time[pos] = self.time_counter;
        self.time_counter += 1;
    }

    /// Select the eviction victim: the line with the smallest effective CRF.
    /// effective_CRF(b) = F(t_c - t_last(b)) · CRF(b)
    fn lrfu_victim(&self) -> usize {
        let tc = self.time_counter;
        let mut min_crf = f64::MAX;
        let mut min_idx = 0;
        for (i, &crf_val) in self.crf.iter().enumerate() {
            let delta = tc - self.last_access_time[i];
            let effective = Self::weight(delta) * crf_val;
            if effective < min_crf {
                min_crf = effective;
                min_idx = i;
            }
        }
        min_idx
    }

    /// Returns true if tag hit; false if miss.
    pub fn access(&mut self, tag: u64) -> bool {
        if let Some(pos) = self.lines.iter().position(|&line| line == Some(tag)) {
            // Cache hit: update LRFU CRF via recurrence.
            // The CRF accumulates contributions from all CPUs that access this
            // block, so a block that is hot across multiple cores builds a
            // high CRF and is protected from eviction — correct shared behaviour.
            self.lrfu_hit(pos);
            true
        } else {
            // Cache miss: find a free line or evict the LRFU victim.
            if let Some(free_pos) = self.lines.iter().position(|&line| line.is_none()) {
                // Found a free line, so use it.
                self.lines[free_pos] = Some(tag);
                self.lrfu_install(free_pos);
            } else {
                // No free line: evict the line with minimum effective CRF.
                let evict_index = self.lrfu_victim();
                self.lines[evict_index] = Some(tag);
                self.lrfu_install(evict_index);
            }
            false
        }
    }

    /// Invalidate a specific block tag in this set (if present).
    ///
    /// CRF and last_access_time are intentionally left stale. lrfu_victim()
    /// is only called when no free slot (None) exists, so an invalidated line
    /// is always claimed by the free-slot path before lrfu_victim() runs.
    /// lrfu_install() then overwrites the stale metadata unconditionally.
    pub fn invalidate(&mut self, tag: u64) {
        if let Some(pos) = self.lines.iter().position(|&line| line == Some(tag)) {
            self.lines[pos] = None;
        }
    }
}

impl Cache {
    pub fn new(size: usize, block_size: usize, associativity: usize) -> Self {
        let num_lines = size / block_size;
        let num_sets = num_lines / associativity;
        let sets = (0..num_sets)
            .map(|_| CacheSet::new(associativity))
            .collect();
        Cache { block_size, sets }
    }

    /// Simulate an access to the cache.
    /// Returns true if hit, false if miss.
    pub fn access(&mut self, address: u64) -> bool {
        let block_addr = address / (self.block_size as u64);
        let set_index = (block_addr as usize) % self.sets.len();
        self.sets[set_index].access(block_addr)
    }

    pub fn invalidate_page(&mut self, address: u64) {
        const PAGE_SIZE: u64 = 4096;
        assert!(address % PAGE_SIZE == 0);

        let start_block = address / (self.block_size as u64);
        let end_block = (address + PAGE_SIZE - 1) / (self.block_size as u64);

        for block_addr in start_block..=end_block {
            let set_index = (block_addr as usize) % self.sets.len();
            self.sets[set_index].invalidate(block_addr);
        }
    }
}

fn parse_rowclone_record(line: &str) -> Result<MemoryAccess, Box<dyn std::error::Error>> {
    MemoryAccess::from_str(line)
}

fn parse_binary_record(line: &str) -> Result<MemoryAccess, Box<dyn std::error::Error>> {
    let access = log_parser::LogRecord::from_str(line)?;
    Ok(MemoryAccess::Regular(MemRecord {
        cpu: access.cpu.into(),
        address: access.address,
        insn_count: access.insn_count,
        store: access.store == 1,
    }))
}

#[derive(Parser, Debug)]
#[command(about)]
struct Args {
    /// Whether the input logs are in binary format
    #[arg(short, long, default_value_t = false)]
    binary_in: bool,

    /// The number of CPUs
    #[arg(short, long, default_value_t = 8)]
    cpus: usize,

    /// Directory for output cpu_N.trace files
    #[arg(short, long)]
    log_dir: String,

    /// Total shared LLC size in bytes (default: 8 MiB)
    #[arg(long, default_value_t = 8 * 1024 * 1024)]
    llc_size: usize,

    /// LLC associativity (default: 16-way)
    #[arg(long, default_value_t = 16)]
    llc_assoc: usize,

    /// LLC block size in bytes (default: 64 B, must match the plugin)
    #[arg(long, default_value_t = 64)]
    llc_block_size: usize,
}

fn ramulator_mem_format(rec: &MemRecord, prev_insn_count: &u64) -> String {
    let bubble = rec.insn_count - prev_insn_count;
    if rec.store {
        format!("{} -1 0x{:016x}", bubble, rec.address)
    } else {
        format!("{} 0x{:016x}", bubble, rec.address)
    }
}

fn main() {
    let args = Args::parse();

    let reader = BufReader::new(std::io::stdin());

    let mut writers: Vec<BufWriter<std::fs::File>> = (0..args.cpus)
        .map(|cpu_id| {
            let filename = format!("{}/cpu_{}.trace", args.log_dir, cpu_id);
            let file = std::fs::File::create(&filename)
                .unwrap_or_else(|_| panic!("Failed to create output file: {}", filename));
            BufWriter::new(file)
        })
        .collect();

    let input_parser = if args.binary_in {
        parse_binary_record
    } else {
        parse_rowclone_record
    };

    // Single shared LLC for all CPUs using LRFU eviction.
    //
    // All CPUs compete for the same capacity, meaning:
    //   - A block loaded by CPU N can be hit by CPU M.
    //   - Eviction pressure from one CPU affects all others.
    //   - Cross-core data reuse is correctly captured.
    //   - The CRF of a block accumulates hits from all CPUs, so data that is
    //     genuinely hot across multiple cores builds a high CRF and is
    //     protected from eviction — correct shared LRFU behaviour.
    //   - A block that was recently accessed by any CPU has a high recency
    //     component in its effective CRF, so inter-CPU temporal locality is
    //     also captured correctly.
    //
    // The input stream from log_merger is already sorted by logical_clock,
    // so accesses arrive in correct global time order. The per-set
    // time_counter therefore reflects true interleaved access order across
    // all cores, making both the recency and frequency components of the
    // CRF accurate.
    let mut shared_llc = Cache::new(args.llc_size, args.llc_block_size, args.llc_assoc);

    let mut first = vec![true; args.cpus];
    let mut prev_insn_count = vec![0u64; args.cpus];

    for line in reader.lines() {
        let line = match line {
            Ok(l) => l,
            Err(e) => {
                eprintln!("Error reading input: {}", e);
                break;
            }
        };

        let rec = match input_parser(&line) {
            Ok(r) => r,
            Err(_) => continue,
        };

        match rec {
            MemoryAccess::Regular(mem) => {
                let cpu = mem.cpu;

                if first[cpu] {
                    prev_insn_count[cpu] = mem.insn_count;
                    first[cpu] = false;
                }

                // Access the shared LLC. A hit updates the CRF regardless of
                // which CPU originally loaded the block — the recency and
                // frequency contributions from all CPUs are pooled into a
                // single CRF value per block, which is the correct behaviour
                // for a shared LRFU LLC. Only misses reach DRAM and are
                // written to the per-CPU Ramulator trace.
                if !shared_llc.access(mem.address) {
                    writeln!(
                        writers[cpu],
                        "{}",
                        ramulator_mem_format(&mem, &prev_insn_count[cpu])
                    )
                    .unwrap_or_else(|e| eprintln!("Write error on cpu_{}.trace: {}", cpu, e));
                    prev_insn_count[cpu] = mem.insn_count;
                }
            }

            MemoryAccess::Rowclone(rc) => {
                let cpu = rc.cpu;

                if first[cpu] {
                    prev_insn_count[cpu] = rc.insn_count;
                    first[cpu] = false;
                }

                // A rowclone operation writes new content directly to a DRAM
                // page. Any cached copy of the destination page in the shared
                // LLC is now stale. One invalidation on the shared instance is
                // sufficient — there is no per-CPU state to iterate over.
                // Stale CRF values are left in place but are safely overwritten
                // by lrfu_install() when the way is next used.
                shared_llc.invalidate_page(rc.to);

                writeln!(
                    writers[cpu],
                    "{} 0x{:016x} 0x{:016x}",
                    rc.insn_count - prev_insn_count[cpu],
                    rc.from,
                    rc.to,
                )
                .unwrap_or_else(|e| eprintln!("Write error on cpu_{}.trace: {}", cpu, e));

                prev_insn_count[cpu] = rc.insn_count;
            }
        }
    }
}
