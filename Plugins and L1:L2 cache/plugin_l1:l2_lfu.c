/*
 * This plugin creates a load store trace in a binary format given by struct
 * LogRecord. It creates per-cpu-logfiles log.txt.[cpu idx]. The tracing is
 * enabled when plugin logs are enabled (qemu monitor command `log plugin`). The
 * tracing is disabled when plugin logs are disabled (qemu monitor command `log
 * none`).
 *
 * Attention: even when the tracing is disabled the plugin slows down the guest
 * significantly. This is because the plugin callbacks are still injected during
 * translation and executed they just do not do anything. One could disable the
 * callback registering completely, but you run the risk of losing some
 * load/stores due to qemu's caching of translation blocks. (I.e. it may still
 * execute cached translation blocks without the plugin callbacks even when the
 * plugin is enabled). If you do not need the plugin do not add it in the qemu
 * command line to avoid slowdowns.
 *
 * The logfiles are closed (and any pending writes flushed) on qemu monitor
 * command `stop`. The logfiles are cleared and reopened on qemu monitor command
 * `continue`.
 *
 * Per-CPU arithmetic instruction counts are written into the shared memory
 * CPU struct (arith_count field) so that logger.c can read them at any time
 * and write per-CPU arith_intensity.N files. Counting is gated by
 * qemu_plugin_log_is_enabled(), matching the behaviour of the memory trace.
 * arith_count is reset to 0 at the start of each logging session so that
 * each log plugin / log none window is counted independently.
 * insn_count is added to the CPU struct in shared memory and reset to 0 at
 * session start; logger.c reads it at shutdown and writes insn_count.N,
 * mirroring the arith_intensity.N output.
 *
 * L1 D-cache configuration: LFU eviction, 4-way associativity, 64B block
 * size, 64 sets → 16 KB total capacity.
 *
 * L2 D-cache configuration: LFU eviction, 8-way associativity, 64B block
 * size, 512 sets → 256 KB total capacity. Private per core, inclusive of L1.
 *
 * Two-level access logic (vcpu_mem):
 *   L1 hit                → increment L2 frequency for this block so L2
 *                           recency stays consistent with L1 usage; return.
 *   L1 miss, L2 hit       → refill L1 (possible L1 eviction, no L2
 *                           back-invalidation needed); silent return.
 *   L1 miss, L2 miss      → refill L2 (possible L2 eviction triggers L1
 *                           back-invalidation to maintain inclusion); refill
 *                           L1; emit LogRecord into shared memory ring.
 *
 * Inclusive invariant: every block present in L1 is also present in L2.
 * Enforced by: (a) installing into L2 on every L2 miss before L1 refill,
 * and (b) back-invalidating L1 whenever L2 evicts a block.
 *
 * LFU frequency invariant:
 *   lfu_freq[way] == 0  <=>  blocks[way].valid == false
 * New blocks are installed with freq=1 so lfu_get_lfu_block() never
 * selects an invalid way as a victim when all ways are occupied.
 */

#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <glib.h>

#include <qemu-plugin.h>

QEMU_PLUGIN_EXPORT int qemu_plugin_version = QEMU_PLUGIN_VERSION;

static int limit;
static bool sys;

typedef struct {
    uint64_t tag;
    bool valid;
} CacheBlock;

typedef struct {
    CacheBlock *blocks;
    uint64_t *lfu_freq;  /* per-way frequency counter; 0 => invalid, >=1 => valid */
} CacheSet;

typedef struct {
    CacheSet *sets;
    int num_sets;
    int cachesize;
    int assoc;
    int blksize_shift;
    uint64_t set_mask;
    uint64_t tag_mask;
    uint64_t accesses;
    uint64_t misses;
} Cache;

typedef struct LogRecord {
  uint64_t logical_clock;
  uint64_t insn_count;
  char cpu;
  char store;
  char access_size;
  uint64_t address;
} LogRecord;

#define MAX_BUFFER_SIZE 2000000

/* All counter fields live in shared memory so logger.c can read them at
 * shutdown and write the corresponding per-CPU stat files. All counters are
 * reset to 0 at session start and only incremented while logging is enabled,
 * so they reflect the current session only.
 *
 * Field layout (must be identical in logger.c):
 *   head          : ring-buffer write index (plugin writes)
 *   tail          : ring-buffer read  index (logger  writes)
 *   arith_count   : arithmetic instructions executed
 *   insn_count    : total instructions executed
 *   l1_miss_count : L1 D-cache misses (L1 miss regardless of L2 outcome)
 *   l2_miss_count : L2 D-cache misses (= records emitted into logRecord ring;
 *                   cross-check: equals log.txt.N file size / 32) */
typedef struct CPU {
  uint64_t head;
  uint64_t tail;
  uint64_t arith_count;              /* per-CPU arithmetic op count          */
  uint64_t insn_count;               /* per-CPU session instruction count    */
  uint64_t l1_miss_count;            /* per-CPU L1 D-cache miss count        */
  uint64_t l2_miss_count;            /* per-CPU L2 D-cache miss count        */
  LogRecord logRecord[MAX_BUFFER_SIZE];
} CPU;

static CPU *cpus;
static int cpu_len;

static struct qemu_plugin_scoreboard *insn_count_score;
static qemu_plugin_u64 insn_count_entry;

/* arith_count_score scoreboard is intentionally absent: the count lives in
 * cpus[cpu_index].arith_count inside shared memory. */

static Cache **l1_dcaches;
static Cache **l2_dcaches;

/* Session tracking: per-CPU flag that records whether logging was active on
 * the previous vcpu_mem call. Used to detect the log-on transition so all
 * counters in shared memory are reset at session start.
 * Allocated with g_new0 so all entries start as false. */
static bool *session_active;

/* ---------------------------------------------------------------------------
 * AArch64 opcode-based arithmetic instruction detection.
 *
 * AArch64 instructions are always 4 bytes (little-endian). We read the raw
 * opcode via qemu_plugin_insn_data() and apply bitmasks to identify
 * arithmetic instruction families.
 *
 * References: ARM Architecture Reference Manual ARMv8-A (DDI 0487)
 * --------------------------------------------------------------------------- */
static bool is_arithmetic_insn(const struct qemu_plugin_insn *insn)
{
    uint32_t op = 0;
    if (qemu_plugin_insn_data(insn, &op, sizeof(op)) != sizeof(op)) {
        return false;
    }

    /* ADD/SUB shifted register: op[28:24] = 0b01011
     * Covers: ADD, ADDS, SUB, SUBS (shifted register, 32 and 64-bit) */
    if ((op & 0x1f000000) == 0x0b000000) return true;

    /* ADD/SUB immediate: op[28:23] = 0b100010
     * Covers: ADD, ADDS, SUB, SUBS (immediate, 32 and 64-bit) */
    if ((op & 0x1f800000) == 0x11000000) return true;

    /* ADD/SUB extended register: op[28:24] = 0b01011, op[21] = 1 */
    if ((op & 0x1fe00000) == 0x0b200000) return true;

    /* ADC, ADCS, SBC, SBCS (add/sub with carry) */
    if ((op & 0x1fe00000) == 0x1a000000) return true;

    /* Data processing 3 source (MADD, MSUB, SMADDL, SMSUBL, SMULH,
     * UMADDL, UMSUBL, UMULH — MUL and MNEG are aliases):
     * op[28:24] = 0b11011 */
    if ((op & 0x1f000000) == 0x1b000000) return true;

    /* SDIV / UDIV — data processing 2 source:
     * op[28:21] = 0b11010110, op[15:10] = 000010 (UDIV) or 000011 (SDIV) */
    if ((op & 0x1fe00000) == 0x1ac00000) {
        uint32_t opcode2 = (op >> 10) & 0x3f;
        if (opcode2 == 0x02 || opcode2 == 0x03) return true;
    }

    /* FP data-processing 2 source (FADD, FSUB, FMUL, FDIV):
     * op[31:24] = 0b00011110, op[21] = 1, op[11:10] = 0b10 */
    if ((op & 0xff200c00) == 0x1e200800) return true;

    /* FP data-processing 3 source (FMADD, FMSUB, FNMADD, FNMSUB):
     * op[28:24] = 0b11111 */
    if ((op & 0x1f000000) == 0x1f000000) return true;

    /* Advanced SIMD three-register same (vector ADD, SUB, MUL, MLA, MLS,
     * FADD, FSUB, FMUL, FMLA, FMLS, SDOT, UDOT, USDOT):
     * op[28]=0, op[24]=1, op[21]=1, op[10]=1 */
    if ((op & 0x0f200400) == 0x0e200400) return true;

    /* SIMD three-register different (SMLAL, UMLAL, SMLSL, UMLSL,
     * SMULL, UMULL, SQDMULH, SQRDMULH) */
    if ((op & 0x0f200400) == 0x0e200000) return true;

    return false;
}

/* --------------------------------------------------------------------------- */

static int pow_of_two(int num)
{
    g_assert((num & (num - 1)) == 0);
    int ret = 0;
    while (num /= 2) {
        ret++;
    }
    return ret;
}

/* lfu_freq_init: allocate and zero per-way frequency counters for every set.
 * g_new0 zeroes the memory so all counters start at 0 (invalid/empty). */
static void lfu_freq_init(Cache *cache)
{
    int i;

    for (i = 0; i < cache->num_sets; i++) {
        cache->sets[i].lfu_freq = g_new0(uint64_t, cache->assoc);
    }
}

/* lfu_update_freq: called on a hit. Increments the frequency of the accessed way. */
static void lfu_update_freq(Cache *cache, int set_idx, int blk_idx)
{
    cache->sets[set_idx].lfu_freq[blk_idx]++;
}

/* lfu_get_lfu_block: called on a miss when all ways are occupied.
 * Returns the way index with the lowest frequency counter.
 * Tie-breaking: lowest way index wins — deterministic and cheap. */
static int lfu_get_lfu_block(Cache *cache, int set_idx)
{
    int i, min_idx;
    uint64_t min_freq;

    min_freq = cache->sets[set_idx].lfu_freq[0];
    min_idx  = 0;

    for (i = 1; i < cache->assoc; i++) {
        if (cache->sets[set_idx].lfu_freq[i] < min_freq) {
            min_freq = cache->sets[set_idx].lfu_freq[i];
            min_idx  = i;
        }
    }
    return min_idx;
}

/* lfu_freq_destroy: free per-set frequency arrays. */
static void lfu_freq_destroy(Cache *cache)
{
    int i;

    for (i = 0; i < cache->num_sets; i++) {
        g_free(cache->sets[i].lfu_freq);
    }
}

/* --------------------------------------------------------------------------- */

static inline uint64_t extract_tag(Cache *cache, uint64_t addr)
{
    return addr & cache->tag_mask;
}

static inline uint64_t extract_set(Cache *cache, uint64_t addr)
{
    return (addr & cache->set_mask) >> cache->blksize_shift;
}

static const char *cache_config_error(int blksize, int assoc, int cachesize)
{
    if (cachesize % blksize != 0) {
        return "cache size must be divisible by block size";
    } else if (cachesize % (blksize * assoc) != 0) {
        return "cache size must be divisible by set size (assoc * block size)";
    } else {
        return NULL;
    }
}

static bool bad_cache_params(int blksize, int assoc, int cachesize)
{
    return (cachesize % blksize) != 0 || (cachesize % (blksize * assoc) != 0);
}

/* cache_init: LFU eviction — initialise per-set frequency arrays. */
static Cache *cache_init(int blksize, int assoc, int cachesize)
{
    Cache *cache;
    int i;
    uint64_t blk_mask;

    g_assert(!bad_cache_params(blksize, assoc, cachesize));

    cache = g_new(Cache, 1);
    cache->assoc = assoc;
    cache->cachesize = cachesize;
    cache->num_sets = cachesize / (blksize * assoc);
    cache->sets = g_new(CacheSet, cache->num_sets);
    cache->blksize_shift = pow_of_two(blksize);
    cache->accesses = 0;
    cache->misses = 0;

    for (i = 0; i < cache->num_sets; i++) {
        cache->sets[i].blocks = g_new0(CacheBlock, assoc);
    }

    blk_mask = blksize - 1;
    cache->set_mask = ((cache->num_sets - 1) << cache->blksize_shift);
    cache->tag_mask = ~(cache->set_mask | blk_mask);

    lfu_freq_init(cache);

    return cache;
}

static Cache **caches_init(int blksize, int assoc, int cachesize)
{
    Cache **caches;
    int i;

    if (bad_cache_params(blksize, assoc, cachesize)) {
        fprintf(stderr, "Bad parameters\n");
        return NULL;
    }

    caches = g_new(Cache *, cpu_len);

    for (i = 0; i < cpu_len; i++) {
        caches[i] = cache_init(blksize, assoc, cachesize);
    }

    return caches;
}

static int get_invalid_block(Cache *cache, uint64_t set)
{
    int i;

    for (i = 0; i < cache->assoc; i++) {
        /* Enforce the LFU invariant: valid==false <=> freq==0.
         * If this fires, a code path set one field without the other. */
        g_assert(cache->sets[set].blocks[i].valid ==
                 (cache->sets[set].lfu_freq[i] != 0));
        if (!cache->sets[set].blocks[i].valid) {
            return i;
        }
    }

    return -1;
}

static int in_cache(Cache *cache, uint64_t addr)
{
    int i;
    uint64_t tag, set;

    tag = extract_tag(cache, addr);
    set = extract_set(cache, addr);

    for (i = 0; i < cache->assoc; i++) {
        if (cache->sets[set].blocks[i].tag == tag &&
                cache->sets[set].blocks[i].valid) {
            return i;
        }
    }

    return -1;
}

/**
 * access_cache(): Simulate a cache access with LFU eviction.
 * @cache: The cache under simulation
 * @addr: The address of the requested memory location
 *
 * Returns true if hit, false if miss. On a hit the frequency counter of the
 * accessed way is incremented. On a miss the least-frequently-used way is
 * evicted and the new line is installed with frequency 1.
 *
 * Frequency is initialised to 1 on install (not 0) to maintain the invariant:
 *   lfu_freq[way] == 0  <=>  blocks[way].valid == false
 * This ensures lfu_get_lfu_block() never selects an invalid way as a victim.
 */
static bool access_cache(Cache *cache, uint64_t addr)
{
    int hit_blk, replaced_blk;
    uint64_t tag, set;

    tag = extract_tag(cache, addr);
    set = extract_set(cache, addr);

    hit_blk = in_cache(cache, addr);
    if (hit_blk != -1) {
        lfu_update_freq(cache, set, hit_blk);
        return true;
    }

    replaced_blk = get_invalid_block(cache, set);

    if (replaced_blk == -1) {
        replaced_blk = lfu_get_lfu_block(cache, set);
    }

    cache->sets[set].blocks[replaced_blk].tag   = tag;
    cache->sets[set].blocks[replaced_blk].valid  = true;
    cache->sets[set].lfu_freq[replaced_blk]      = 1;

    return false;
}

/* ---------------------------------------------------------------------------
 * l2_install_block(): Install addr into the L2 cache (LFU eviction).
 *
 * Returns the reconstructed physical address of the block evicted from L2,
 * or 0 if no valid block was displaced (invalid way used, or addr already
 * present). The caller passes this address to l1_back_invalidate() to remove
 * the corresponding block from L1, maintaining the inclusive invariant.
 *
 * The return value is a reconstructed address, NOT a raw tag. A raw tag has
 * the set-index bits zeroed by tag_mask, so extracting the L1 set from it
 * always yields 0. Instead we reconstruct:
 *   evicted_addr = evicted_tag | (set << l2->blksize_shift)
 * which restores the set-index bits so l1_back_invalidate() can call
 * extract_set(l1, evicted_addr) correctly for any set, not just set 0.
 *
 * On a hit, the block's frequency is incremented (keeping L2 usage counts
 * consistent with L1 hits) and 0 is returned — no eviction occurred.
 *
 * Returning 0 is safe as a sentinel because physical address 0 is never
 * a valid data block in our workloads.
 * --------------------------------------------------------------------------- */
static uint64_t l2_install_block(Cache *l2, uint64_t addr)
{
    uint64_t tag = extract_tag(l2, addr);
    uint64_t set = extract_set(l2, addr);

    /* L2 hit: increment frequency to reflect the L1 hit that triggered this
     * call, keeping L2 usage counts consistent with actual access patterns. */
    int hit_blk = in_cache(l2, addr);
    if (hit_blk != -1) {
        lfu_update_freq(l2, set, hit_blk);
        return 0;
    }

    /* L2 miss: find a victim way. */
    int victim = get_invalid_block(l2, set);
    uint64_t evicted_addr = 0;

    if (victim == -1) {
        /* All ways valid — evict the LFU block.
         * Reconstruct the full address from the stored tag and the current
         * set index so l1_back_invalidate() can locate it correctly in L1. */
        victim = lfu_get_lfu_block(l2, set);
        evicted_addr = l2->sets[set].blocks[victim].tag
                       | (set << l2->blksize_shift);
    }

    l2->sets[set].blocks[victim].tag   = tag;
    l2->sets[set].blocks[victim].valid = true;
    l2->sets[set].lfu_freq[victim]     = 1;

    return evicted_addr;
}

/* ---------------------------------------------------------------------------
 * l1_back_invalidate(): Remove the block at evicted_addr from L1 if present.
 *
 * Called whenever L2 evicts a valid block. Because L2 is inclusive of L1,
 * any block leaving L2 must also be removed from L1 — otherwise L1 would
 * hold a block not present in L2, violating the inclusion invariant.
 *
 * evicted_addr == 0 means no valid block was displaced from L2, so nothing
 * to do.
 *
 * evicted_addr is a reconstructed physical address (tag | set << blksize_shift)
 * returned by l2_install_block(). Using a full address — not a raw tag — is
 * required because extract_set(l1, addr) needs the set-index bits, which are
 * zeroed in a tag. Passing the raw L2 tag would always map to L1 set 0.
 *
 * The invalidated way's lfu_freq is reset to 0 to restore the invariant
 *   lfu_freq[way] == 0  <=>  blocks[way].valid == false
 * so that get_invalid_block() finds it before lfu_get_lfu_block() is called.
 * --------------------------------------------------------------------------- */
static void l1_back_invalidate(Cache *l1, uint64_t evicted_addr)
{
    if (evicted_addr == 0) {
        return;
    }

    uint64_t set = extract_set(l1, evicted_addr);
    int way = in_cache(l1, evicted_addr);

    if (way != -1) {
        l1->sets[set].blocks[way].valid = false;
        /* Reset frequency to 0 to restore the LFU invariant:
         * lfu_freq[way]==0 signals an invalid way to get_invalid_block(). */
        l1->sets[set].lfu_freq[way] = 0;
    }
    /* Block not in L1 — nothing to do. This is the common case: most L2
     * evictions displace blocks that were already evicted from the smaller
     * L1 earlier. */
}

static void vcpu_mem(unsigned int cpu_index, qemu_plugin_meminfo_t info,
                     uint64_t vaddr, void *udata) {
  bool log_enabled = qemu_plugin_log_is_enabled();

  /* Detect log-on transition: logging just became enabled.
   * Reset all counters in shared memory to 0 so this session starts from a
   * clean slate. logger.c reads all counters at shutdown. */
  if (log_enabled && !session_active[cpu_index]) {
    session_active[cpu_index] = true;
    __atomic_store_n(&cpus[cpu_index].arith_count,   0, __ATOMIC_RELAXED);
    __atomic_store_n(&cpus[cpu_index].insn_count,    0, __ATOMIC_RELAXED);
    __atomic_store_n(&cpus[cpu_index].l1_miss_count, 0, __ATOMIC_RELAXED);
    __atomic_store_n(&cpus[cpu_index].l2_miss_count, 0, __ATOMIC_RELAXED);
  }

  /* Detect log-off transition: logging just became disabled.
   * Mark the session inactive so the next log-on triggers a fresh reset. */
  if (!log_enabled && session_active[cpu_index]) {
    session_active[cpu_index] = false;
  }

  if (!log_enabled) {
    return;
  }
  uint64_t effective_addr;
  int cache_idx;
  struct qemu_plugin_hwaddr *hwaddr = qemu_plugin_get_hwaddr(info, vaddr);
  if (qemu_plugin_hwaddr_is_io(hwaddr)) {
    return;
  }
  effective_addr = hwaddr ? qemu_plugin_hwaddr_phys_addr(hwaddr) : vaddr;
  cache_idx = cpu_index % cpu_len;

  /* ── Two-level cache access ───────────────────────────────────────────────
   *
   * L1 hit:
   *   Increment L2 frequency for this block so L2 usage counts stay
   *   consistent with actual access patterns (L1 hits are real accesses).
   *   Return — no miss to record.
   *
   * L1 miss, L2 hit:
   *   Refill L1 from L2. The L1 eviction (if any) does NOT back-invalidate
   *   L2 — L2 is the inclusive superset, so an L1 eviction simply means the
   *   block is now only in L2, which is fine. The evicted L1 way's lfu_freq
   *   is NOT reset here because access_cache() replaces it immediately with
   *   freq=1 for the newly installed block.
   *   Return — not an L2 miss, nothing to record.
   *
   * L1 miss, L2 miss:
   *   Install block in L2 first (freq=1). If L2 evicts a valid block,
   *   back-invalidate that block from L1 (and reset its freq to 0) to
   *   maintain the inclusive invariant.
   *   Then install block in L1 (freq=1).
   *   Emit LogRecord into the shared memory ring (true LLC-bound miss).
   * ---------------------------------------------------------------------- */
  bool hit_in_l1 = access_cache(l1_dcaches[cache_idx], effective_addr);

  if (hit_in_l1) {
    /* L1 hit: keep L2 frequency consistent — promote without recording miss. */
    l2_install_block(l2_dcaches[cache_idx], effective_addr);
    return;
  }

  /* L1 miss — count it regardless of what L2 does. */
  __atomic_fetch_add(&cpus[cpu_index].l1_miss_count, 1, __ATOMIC_RELAXED);

  /* L1 miss: consult L2. */
  bool hit_in_l2 = (in_cache(l2_dcaches[cache_idx], effective_addr) != -1);

  if (hit_in_l2) {
    /* L2 hit: promote L2 frequency, then refill L1. No LogRecord needed. */
    l2_install_block(l2_dcaches[cache_idx], effective_addr);
    access_cache(l1_dcaches[cache_idx], effective_addr);
    return;
  }

  /* L2 miss — count it, install in L2, back-invalidate L1 if needed,
   * refill L1, then emit the LogRecord into the shared memory ring. */
  __atomic_fetch_add(&cpus[cpu_index].l2_miss_count, 1, __ATOMIC_RELAXED);
  uint64_t evicted_addr = l2_install_block(l2_dcaches[cache_idx], effective_addr);
  l1_back_invalidate(l1_dcaches[cache_idx], evicted_addr);
  access_cache(l1_dcaches[cache_idx], effective_addr);
  uint64_t index = __atomic_fetch_add(&cpus[cpu_index].head, 1, __ATOMIC_SEQ_CST);
  uint64_t real_index = index % MAX_BUFFER_SIZE;
  if (qemu_plugin_mem_is_store(info)) {
    cpus[cpu_index].logRecord[real_index].store = 1;
  } else {
    cpus[cpu_index].logRecord[real_index].store = 0;
  }
  uint64_t addr = qemu_plugin_hwaddr_phys_addr(hwaddr);
  cpus[cpu_index].logRecord[real_index].address = addr;
  cpus[cpu_index].logRecord[real_index].cpu = cpu_index;
  cpus[cpu_index].logRecord[real_index].access_size = qemu_plugin_mem_size_shift(info);
  cpus[cpu_index].logRecord[real_index].logical_clock = qemu_plugin_u64_sum(insn_count_entry);
  uint64_t *val = qemu_plugin_scoreboard_find(insn_count_score, cpu_index);
  cpus[cpu_index].logRecord[real_index].insn_count = *val;
}

/* Instruction callback. Increments insn_count in shared memory so logger.c
 * can read the session total at shutdown. Gated identically to vcpu_arith.
 *
 * This callback also mirrors the session state machine from vcpu_mem.
 * vcpu_mem only fires on memory instructions, so if the VM is stopped while
 * session_active is true and then `log none` is issued while stopped, the
 * log-off transition in vcpu_mem never executes and session_active stays true.
 * When `log plugin` is re-issued and the VM resumes, vcpu_mem would then see
 * (log_enabled && !session_active) == false and skip the counter reset,
 * causing all counters to accumulate across sessions (~2x on the second run).
 *
 * By mirroring both transitions here, the first instruction executed after
 * any log state change correctly drives session_active and resets counters,
 * regardless of whether a memory access has fired yet. The transitions are
 * idempotent — if vcpu_mem fires first, vcpu_insn's check is a no-op, and
 * vice versa. */
static void vcpu_insn(unsigned int cpu_index, void *udata) {
  bool log_enabled = qemu_plugin_log_is_enabled();

  if (log_enabled && !session_active[cpu_index]) {
    session_active[cpu_index] = true;
    __atomic_store_n(&cpus[cpu_index].arith_count,   0, __ATOMIC_RELAXED);
    __atomic_store_n(&cpus[cpu_index].insn_count,    0, __ATOMIC_RELAXED);
    __atomic_store_n(&cpus[cpu_index].l1_miss_count, 0, __ATOMIC_RELAXED);
    __atomic_store_n(&cpus[cpu_index].l2_miss_count, 0, __ATOMIC_RELAXED);
  }
  if (!log_enabled && session_active[cpu_index]) {
    session_active[cpu_index] = false;
  }

  if (!log_enabled) {
    return;
  }
  __atomic_fetch_add(&cpus[cpu_index].insn_count, 1, __ATOMIC_RELAXED);
}

/* Arithmetic instruction callback. Gated by qemu_plugin_log_is_enabled()
 * so it only counts during active logging sessions, exactly like vcpu_mem.
 *
 * Atomically increments arith_count in shared memory instead of using a
 * private scoreboard, so logger.c can read it live.
 * ATOMIC_RELAXED is sufficient — we only need atomicity, not ordering. */
static void vcpu_arith(unsigned int cpu_index, void *udata) {
  if (!qemu_plugin_log_is_enabled()) {
    return;
  }
  __atomic_fetch_add(&cpus[cpu_index].arith_count, 1, __ATOMIC_RELAXED);
}

static void vcpu_tb_trans(qemu_plugin_id_t id, struct qemu_plugin_tb *tb) {
  struct qemu_plugin_insn *insn;
  size_t n_insns = qemu_plugin_tb_n_insns(tb);

  for (size_t i = 0; i < n_insns; i++) {
    insn = qemu_plugin_tb_get_insn(tb, i);

    /* Memory callback + instruction counter. */
    qemu_plugin_register_vcpu_mem_cb(insn, vcpu_mem, QEMU_PLUGIN_CB_NO_REGS,
                                     QEMU_PLUGIN_MEM_RW, NULL);
    qemu_plugin_register_vcpu_insn_exec_inline_per_vcpu(insn,
      QEMU_PLUGIN_INLINE_ADD_U64, insn_count_entry, 1);

    /* Instruction counter: every instruction gets a gated callback that
     * increments insn_count in shared memory when logging is enabled. */
    qemu_plugin_register_vcpu_insn_exec_cb(insn, vcpu_insn,
        QEMU_PLUGIN_CB_NO_REGS, NULL);

    /* Arithmetic counter: identify instruction by opcode at translation time
     * and register a gated callback only for matching instructions. */
    if (is_arithmetic_insn(insn)) {
        qemu_plugin_register_vcpu_insn_exec_cb(insn, vcpu_arith,
            QEMU_PLUGIN_CB_NO_REGS, NULL);
    }
  }
}

static void cache_free(Cache *cache)
{
    for (int i = 0; i < cache->num_sets; i++) {
        g_free(cache->sets[i].blocks);
    }

    lfu_freq_destroy(cache);

    g_free(cache->sets);
    g_free(cache);
}

static void caches_free(Cache **caches)
{
    int i;

    for (i = 0; i < cpu_len; i++) {
        cache_free(caches[i]);
    }
}

static void vcpu_init(qemu_plugin_id_t id, unsigned int vcpu_index) {

}

/* No arith_count_score to free since the scoreboard was replaced by the
 * shared memory field. */
static void plugin_exit(qemu_plugin_id_t id, void *p) {
  qemu_plugin_scoreboard_free(insn_count_score);
  caches_free(l1_dcaches);
  caches_free(l2_dcaches);
  munmap(cpus, cpu_len * sizeof(CPU));
  g_free(session_active);
}

QEMU_PLUGIN_EXPORT int qemu_plugin_install(qemu_plugin_id_t id,
                                           const qemu_info_t *info, int argc,
                                           char **argv) {
  int l1_dassoc, l1_dblksize, l1_dcachesize;
  int l2_dassoc, l2_dblksize, l2_dcachesize;

  limit = 32;
  sys = info->system_emulation;

  /*
   * L1 D-cache configuration:
   *   Block size   : 64 B
   *   Associativity: 4 ways
   *   Sets         : 64
   *   Total size   : 64 * 4 * 64 = 16 KB
   *   Replacement  : LFU
   */
  l1_dassoc     = 4;
  l1_dblksize   = 64;
  l1_dcachesize = l1_dblksize * l1_dassoc * 256;  /* 16 384 B = 16 KB */

  /*
   * L2 D-cache configuration:
   *   Block size   : 64 B  (must match L1)
   *   Associativity: 8 ways
   *   Sets         : 512
   *   Total size   : 64 * 8 * 512 = 256 KB
   *   Replacement  : LFU
   *   Inclusion    : inclusive of L1 (back-invalidation on L2 eviction)
   */
  l2_dassoc     = 8;
  l2_dblksize   = 64;
  l2_dcachesize = l2_dblksize * l2_dassoc * 4096; /* 262 144 B = 256 KB */

  cpu_len = info->system.max_vcpus;
  session_active = g_new0(bool, cpu_len);

  l1_dcaches = caches_init(l1_dblksize, l1_dassoc, l1_dcachesize);
  if (!l1_dcaches) {
      const char *err = cache_config_error(l1_dblksize, l1_dassoc, l1_dcachesize);
      fprintf(stderr, "L1 dcache cannot be constructed from given parameters\n");
      if (err) {
          fprintf(stderr, "%s\n", err);
      }
      return -1;
  }

  l2_dcaches = caches_init(l2_dblksize, l2_dassoc, l2_dcachesize);
  if (!l2_dcaches) {
      const char *err = cache_config_error(l2_dblksize, l2_dassoc, l2_dcachesize);
      fprintf(stderr, "L2 dcache cannot be constructed from given parameters\n");
      if (err) {
          fprintf(stderr, "%s\n", err);
      }
      caches_free(l1_dcaches);
      return -1;
  }

  uint64_t total_size = cpu_len * sizeof(CPU);
  int fd = shm_open("/qemu_trace", O_CREAT | O_RDWR | O_TRUNC, 0666);
  if (ftruncate(fd, total_size) == -1) {
    perror("ftruncate");
    close(fd);
    return -1;
  }
  cpus = mmap(NULL, total_size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
  memset(cpus, 0, total_size);

  insn_count_score = qemu_plugin_scoreboard_new(sizeof(uint64_t));
  insn_count_entry = qemu_plugin_scoreboard_u64(insn_count_score);

  /* arith_count_score allocation intentionally absent: the count lives in
   * cpus[i].arith_count, zeroed by memset above. */

  /* Register init, translation block and exit callbacks */
  qemu_plugin_register_vcpu_init_cb(id, vcpu_init);
  qemu_plugin_register_vcpu_tb_trans_cb(id, vcpu_tb_trans);
  qemu_plugin_register_atexit_cb(id, plugin_exit, NULL);

  return 0;
}
