use std::{
    io::{BufRead, BufReader, BufWriter, Write},
    str::FromStr,
};

use cf_qemu_post::{
    log_parser::{self},
    memory_access::{MemRecord, MemoryAccess},
};
use clap::Parser;

/* ---- Cache Data Structures ---- */

#[derive(Debug)]
pub struct Cache {
    block_size: usize, // in bytes
    sets: Vec<CacheSet>,
}

#[derive(Debug)]
struct CacheSet {
    // Each cache line stores an optional tag (block address)
    lines: Vec<Option<u64>>,
    // LFU frequency counters — one per way.
    // Invariant: frequencies[i] == 0 <=> lines[i] is None.
    frequencies: Vec<u64>,
}

impl CacheSet {
    pub fn new(associativity: usize) -> Self {
        CacheSet {
            lines: vec![None; associativity],
            frequencies: vec![0; associativity],
        }
    }

    /// Returns true if tag hit; false if miss.
    pub fn access(&mut self, tag: u64) -> bool {
        if let Some(pos) = self.lines.iter().position(|&line| line == Some(tag)) {
            // --- CACHE HIT ---
            // Increment the frequency of the accessed block
            self.frequencies[pos] += 1;
            true
        } else {
            // --- CACHE MISS ---
            // 1. Check for an empty (invalid) slot first
            let target_index = if let Some(free_pos) = self.lines.iter().position(|&line| line.is_none()) {
                free_pos
            } else {
                // 2. If full, find the way with the minimum frequency (LFU)
                // Tie-breaking: .min_by_key returns the first (lowest) index encountered.
                self.frequencies
                    .iter()
                    .enumerate()
                    .min_by_key(|&(_, &freq)| freq)
                    .map(|(idx, _)| idx)
                    .unwrap_or(0)
            };

            // Install new block
            self.lines[target_index] = Some(tag);
            // Invariant: Initializing to 1 (provides a grace period, matches C plugin logic)
            self.frequencies[target_index] = 1;
            false
        }
    }

    /// Invalidate a specific block tag in this set (if present).
    pub fn invalidate(&mut self, tag: u64) {
        if let Some(pos) = self.lines.iter().position(|&line| line == Some(tag)) {
            // Remove the line
            self.lines[pos] = None;
            // Reset frequency to 0 so the way is considered "empty"
            self.frequencies[pos] = 0;
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

    pub fn access(&mut self, address: u64) -> bool {
        let block_addr = address / (self.block_size as u64);
        let set_index = (block_addr as usize) % self.sets.len();
        self.sets[set_index].access(block_addr)
    }

    pub fn invalidate_page(&mut self, address: u64) {
        const PAGE_SIZE: u64 = 4096;
        debug_assert!(address % PAGE_SIZE == 0);

        let block_size = self.block_size as u64;
        let start_block = address / block_size;
        let num_blocks = PAGE_SIZE / block_size;

        for i in 0..num_blocks {
            let block_addr = start_block + i;
            let set_index = (block_addr as usize) % self.sets.len();
            self.sets[set_index].invalidate(block_addr);
        }
    }
}

/* ---- Parsing & I/O Helpers ---- */

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
    #[arg(short, long, default_value_t = false)]
    binary_in: bool,

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

/* ---- Main Execution Loop ---- */

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

    // Shared LLC (L2): 4MB, 64B blocks, 16-way associative
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
                        ).expect("Write failed");
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
                    ).expect("Write failed");
                    prev_insn_count[cpu] = rc.insn_count;
                }
            }
        }
    }
}
