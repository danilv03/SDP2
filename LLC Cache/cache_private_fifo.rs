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
    /// of a new block (miss path) enqueues the way index.
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
    /// Create a new cache.
    ///
    /// Configuration used by the caller:
    ///   block_size   = 64 B
    ///   associativity = 8 ways
    ///   size         = 256 KB  →  num_sets = (256*1024 / 64) / 8 = 512 sets
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

    // L2 cache: 4B, 64B blocks, 16-way associative → 4096 sets, FIFO eviction.
    let mut caches: Vec<Cache> = (0..args.cpus)
        .map(|_| Cache::new(8 * 1024 * 1024, 64, 16))
        .collect();

    let mut lines = reader.lines();
    let mut first = vec![true; args.cpus];
    let mut prev_insn_count = vec![0u64; args.cpus];

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
