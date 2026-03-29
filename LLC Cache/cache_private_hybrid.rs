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
    pub fn invalidate(&mut self, tag: u64) {
        if let Some(pos) = self.lines.iter().position(|&line| line == Some(tag)) {
            // Remove the line; CRF/timestamp will be overwritten on next install.
            self.lines[pos] = None;
        }
    }
}

impl Cache {
    pub fn new(size: usize, block_size: usize, associativity: usize) -> Self {
        // total number of cache lines = size / block_size
        // number of sets = (size / block_size) / associativity
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
        // The tag can simply be the block_addr
        self.sets[set_index].access(block_addr)
    }

    pub fn invalidate_page(&mut self, address: u64) {
        const PAGE_SIZE: u64 = 4096;
        assert!(address % PAGE_SIZE == 0);

        // Compute block indices in page
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
    // Whether the input logs are in binary format
    #[arg(short, long, default_value_t = false)]
    binary_in: bool,

    // the number of CPUs
    #[arg(short, long, default_value_t = 8)]
    cpus: usize,

    #[arg(short, long)]
    log_dir: String,
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

    // Create an L2 cache: 4MB, 64B blocks, 16-way associative.
    // no need for an L1 since we model inclusive cache and only care about
    // memory accesses
    // (total_size_in_bytes, block_size_in_bytes, associativity)
    let mut caches: Vec<Cache> = (0..args.cpus)
        .map(|_| Cache::new(8 * 1024 * 1024, 64, 16))
        .collect();

    let mut lines = reader.lines();
    let mut first = vec![true; args.cpus];
    let mut prev_insn_count = vec![0; args.cpus];

    while let Some(Ok(line)) = lines.next() {
        if let Ok(rec) = input_parser(&line) {
            match rec {
                MemoryAccess::Regular(mem) => {
                    let cpu = mem.cpu;
                    if first[cpu] {
                        prev_insn_count[cpu] = mem.insn_count;
                        first[cpu] = false;
                    }
                    if !caches[cpu].access(mem.address) {
                        writeln!(
                            writers[cpu],
                            "{}",
                            ramulator_mem_format(&mem, &prev_insn_count[cpu])
                        );
                        prev_insn_count[cpu] = mem.insn_count;
                    }
                }
                MemoryAccess::Rowclone(rc) => {
                    let cpu = rc.cpu;
                    if first[cpu] {
                        prev_insn_count[cpu] = rc.insn_count;
                        first[cpu] = false;
                    }
                    for cache in &mut caches {
                        cache.invalidate_page(rc.to);
                    }
                    writeln!(
                        writers[cpu],
                        "{} 0x{:016x} 0x{:016x}",
                        rc.insn_count - prev_insn_count[cpu],
                        rc.from,
                        rc.to,
                    );
                    prev_insn_count[cpu] = rc.insn_count;
                }
            }
        }
    }
}
