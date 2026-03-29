use std::{
    collections::VecDeque,
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

#[derive(Debug)]
struct CacheSet {
    // Each cache line stores an optional tag (a u64 representing the block address).
    lines: Vec<Option<u64>>,
    // FIFO insertion order: front = oldest (next to be evicted), back = newest.
    // Only updated on insertion (miss); hits do NOT touch this queue.
    fifo_order: VecDeque<usize>,
}

impl CacheSet {
    pub fn new(associativity: usize) -> Self {
        CacheSet {
            lines: vec![None; associativity],
            fifo_order: VecDeque::new(),
        }
    }

    /// Returns true on a cache hit, false on a miss.
    ///
    /// FIFO property: a hit does NOT alter `fifo_order`. Only the installation
    /// of a new block (miss path) enqueues the way index. In a shared LLC this
    /// means the eviction order is determined purely by insertion time across
    /// all CPUs — a block inserted early by CPU 3 will be evicted before a
    /// block inserted later by CPU 0, regardless of how many times either has
    /// been accessed since insertion.
    pub fn access(&mut self, tag: u64) -> bool {
        if self.lines.iter().any(|&line| line == Some(tag)) {
            // Cache hit: FIFO eviction order is unchanged.
            true
        } else {
            // Cache miss: find a free way first, then fall back to FIFO eviction.
            if let Some(free_pos) = self.lines.iter().position(|&line| line.is_none()) {
                // Install into the free way and record its insertion order.
                self.lines[free_pos] = Some(tag);
                self.fifo_order.push_back(free_pos);
            } else {
                // All ways occupied: evict the oldest-inserted block (front of queue).
                let evict_index = self
                    .fifo_order
                    .pop_front()
                    .expect("fifo_order must be non-empty when all ways are valid");
                self.lines[evict_index] = Some(tag);
                // Re-enqueue the way at the back — it is now the newest insertion.
                self.fifo_order.push_back(evict_index);
            }
            false
        }
    }

    /// Invalidate a specific block tag in this set (if present).
    pub fn invalidate(&mut self, tag: u64) {
        if let Some(pos) = self.lines.iter().position(|&line| line == Some(tag)) {
            // Remove the block.
            self.lines[pos] = None;
            // Remove the way index from the FIFO queue so it is not later
            // mistakenly evicted as if it still held valid data.
            self.fifo_order.retain(|&i| i != pos);
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

    /// Total shared LLC size in bytes (default: 4 MiB)
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

    // Single shared LLC for all CPUs using FIFO eviction.
    //
    // All CPUs compete for the same capacity, meaning:
    //   - A block loaded by CPU N can be hit by CPU M.
    //   - Eviction pressure from one CPU affects all others.
    //   - Cross-core data reuse is correctly captured.
    //   - Eviction order is determined by global insertion time across all
    //     CPUs: the block that was installed earliest (by any CPU) is always
    //     evicted first when the set is full. This is the correct shared FIFO
    //     behaviour — there is no per-CPU ownership of insertion order.
    //
    // The input stream from log_merger is already sorted by logical_clock,
    // so blocks are inserted into the shared FIFO queue in true global time
    // order, making the eviction order accurate across all cores.
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

                // Access the shared LLC. A hit does not alter the FIFO order
                // of any block — including blocks loaded by other CPUs. Only
                // misses install new blocks and advance the FIFO queue. Only
                // misses reach DRAM and are written to the per-CPU Ramulator
                // trace.
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
                // sufficient. The invalidated ways are also removed from
                // fifo_order so they are correctly treated as free slots on
                // the next access to that set.
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
