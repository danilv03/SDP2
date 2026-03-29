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
 * CPU struct (arith_count field) so that logger.c can read them at any time.
 * Counting is gated by qemu_plugin_log_is_enabled(), matching the behaviour
 * of the memory trace. arith_count is reset to 0 at the start of each logging
 * session so that each log plugin / log none window is counted independently.
 * insn_count is added to the CPU struct in shared memory and reset to 0 at
 * session start; logger.c reads it at shutdown and writes insn_count.N.
 *
 * L1 D-cache configuration: FIFO eviction, 4-way associativity,
 * 64B block size, 64 sets → 16 KB total capacity.
 *
 * L2 D-cache configuration: FIFO eviction, 8-way associativity,
 * 64B block size, 512 sets → 256 KB total capacity. Private per core,
 * inclusive of L1.
 *
 * Replacement policy: FIFO (both levels).
 *
 * Design rationale:
 *   FIFO evicts the block that has been resident longest, regardless of how
 *   many times it has been accessed since insertion. fifo_times[way] is
 *   stamped once when the block is *installed* on a miss; hits never modify
 *   fifo_times. The block with the smallest stamp in a set is always the
 *   oldest and is chosen as the victim when all ways are occupied.
 *
 *   FIFO is simpler and cheaper than LRU (no per-hit metadata update), while
 *   still exploiting temporal locality for blocks that are reused within their
 *   residency window. It is the natural baseline to compare against LRU and
 *   Hybrid for workloads with regular, streaming access patterns.
 *
 * Two-level access logic (vcpu_mem):
 *   L1 hit                → call l2_promote() to assert inclusion; return.
 *                           No L2 metadata update — FIFO hits touch nothing.
 *   L1 miss, L2 hit       → l2_install_block() promotes nothing (FIFO hit
 *                           path is a no-op); refill L1; return.
 *   L1 miss, L2 miss      → l2_install_block() evicts oldest L2 block if
 *                           needed, back-invalidates L1, refills L1, emits
 *                           LogRecord into shared memory ring.
 *
 * Inclusive invariant: every block present in L1 is also present in L2.
 * Enforced by: (a) installing into L2 on every L2 miss before L1 refill,
 * and (b) back-invalidating L1 whenever L2 evicts a block.
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
    /*
     * fifo_times[i] records the generation counter at the moment block i was
     * *inserted* (filled on a miss). On a hit we intentionally do NOT update
     * this value — that is the defining property of FIFO vs LRU.
     */
    uint64_t *fifo_times;
    uint64_t  fifo_gen_counter;
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

/* ---------------------------------------------------------------------------
 * FIFO replacement policy helpers.
 *
 * fifo_times[i] is stamped once when block i is *inserted* on a miss.
 * The block with the smallest stamp is the oldest and is evicted first.
 * Cache hits do not touch fifo_times — preserving insertion order is the
 * defining property of FIFO.
 * --------------------------------------------------------------------------- */

static void fifo_init(Cache *cache)
{
    int i;

    for (i = 0; i < cache->num_sets; i++) {
        cache->sets[i].fifo_times      = g_new0(uint64_t, cache->assoc);
        cache->sets[i].fifo_gen_counter = 0;
    }
}

/* Called only on a miss, when a block is being installed. */
static void fifo_update_blk(Cache *cache, int set_idx, int blk_idx)
{
    CacheSet *set = &cache->sets[set_idx];
    set->fifo_times[blk_idx] = set->fifo_gen_counter;
    set->fifo_gen_counter++;
}

/* Return the index of the block that was inserted earliest (smallest stamp). */
static int fifo_get_evict_block(Cache *cache, int set_idx)
{
    int i, min_idx;
    uint64_t min_time;

    min_time = cache->sets[set_idx].fifo_times[0];
    min_idx  = 0;

    for (i = 1; i < cache->assoc; i++) {
        if (cache->sets[set_idx].fifo_times[i] < min_time) {
            min_time = cache->sets[set_idx].fifo_times[i];
            min_idx  = i;
        }
    }
    return min_idx;
}

static void fifo_destroy(Cache *cache)
{
    int i;

    for (i = 0; i < cache->num_sets; i++) {
        g_free(cache->sets[i].fifo_times);
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

static Cache *cache_init(int blksize, int assoc, int cachesize)
{
    Cache *cache;
    int i;
    uint64_t blk_mask;

    g_assert(!bad_cache_params(blksize, assoc, cachesize));

    cache = g_new(Cache, 1);
    cache->assoc      = assoc;
    cache->cachesize  = cachesize;
    cache->num_sets   = cachesize / (blksize * assoc);
    cache->sets       = g_new(CacheSet, cache->num_sets);
    cache->blksize_shift = pow_of_two(blksize);
    cache->accesses   = 0;
    cache->misses     = 0;

    for (i = 0; i < cache->num_sets; i++) {
        cache->sets[i].blocks = g_new0(CacheBlock, assoc);
    }

    blk_mask         = blksize - 1;
    cache->set_mask  = ((cache->num_sets - 1) << cache->blksize_shift);
    cache->tag_mask  = ~(cache->set_mask | blk_mask);

    fifo_init(cache);

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
 * access_cache(): Simulate a cache access with FIFO eviction.
 * @cache: The cache under simulation
 * @addr:  The address of the requested memory location
 *
 * Returns true on a hit, false on a miss.
 *
 * Hit path:
 *   Return true immediately. fifo_times is intentionally NOT updated — a hit
 *   must not alter a block's eviction order. This is the defining FIFO
 *   property.
 *
 * Miss path:
 *   Prefer an invalid (empty) way; fall back to fifo_get_evict_block().
 *   Stamp the chosen way with fifo_update_blk() and install the new block.
 */
static bool access_cache(Cache *cache, uint64_t addr)
{
    int hit_blk, replaced_blk;
    uint64_t tag, set;

    tag = extract_tag(cache, addr);
    set = extract_set(cache, addr);

    hit_blk = in_cache(cache, addr);
    if (hit_blk != -1) {
        /* FIFO: do NOT update fifo_times on a hit. */
        return true;
    }

    /* Miss: fill an invalid way first, then fall back to FIFO eviction. */
    replaced_blk = get_invalid_block(cache, set);

    if (replaced_blk == -1) {
        replaced_blk = fifo_get_evict_block(cache, set);
    }

    /* Stamp the newly installed block with the current FIFO generation. */
    fifo_update_blk(cache, set, replaced_blk);

    cache->sets[set].blocks[replaced_blk].tag   = tag;
    cache->sets[set].blocks[replaced_blk].valid = true;

    return false;
}

/* ---------------------------------------------------------------------------
 * l2_promote(): Assert L2 inclusion on an L1 hit.
 *
 * Called on the L1-hit path in place of l2_install_block(). Under FIFO,
 * hits do not update any metadata — fifo_times is stamped only on miss
 * installs — so there is nothing to do beyond verifying that the block is
 * present in L2 (inclusion invariant check).
 *
 * The g_assert fires if the block is in L1 but absent from L2, which would
 * mean the inclusive invariant has been violated somewhere.
 * --------------------------------------------------------------------------- */
static void l2_promote(Cache *l2, uint64_t addr)
{
    int hit_blk = in_cache(l2, addr);

    /* The inclusive invariant guarantees this block is in L2 whenever it is
     * in L1. If this assert fires, the inclusion invariant has been violated. */
    g_assert(hit_blk != -1);

    /* FIFO: no metadata update on a hit. */
}

/* ---------------------------------------------------------------------------
 * l2_install_block(): Install addr into the L2 cache (FIFO eviction).
 *
 * Called ONLY on the L1-miss path (L2 hit or L2 miss). Never called on L1
 * hits — use l2_promote() for that case.
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
 * On an L2 hit, 0 is returned and no metadata is updated (FIFO hit = no-op).
 * On an L2 miss, the oldest block (smallest fifo_times) is evicted if all
 * ways are occupied.
 *
 * Returning 0 is safe as a sentinel because physical address 0 is never
 * a valid data block in our workloads.
 * --------------------------------------------------------------------------- */
static uint64_t l2_install_block(Cache *l2, uint64_t addr)
{
    uint64_t tag = extract_tag(l2, addr);
    uint64_t set = extract_set(l2, addr);

    /* L2 hit: FIFO does not update any metadata on a hit. */
    int hit_blk = in_cache(l2, addr);
    if (hit_blk != -1) {
        return 0;
    }

    /* L2 miss: find a victim way. */
    int victim = get_invalid_block(l2, set);
    uint64_t evicted_addr = 0;

    if (victim == -1) {
        /* All ways valid — evict the oldest (smallest fifo_times stamp).
         * Reconstruct the full address from the stored tag and the current
         * set index so l1_back_invalidate() can locate it correctly in L1. */
        victim = fifo_get_evict_block(l2, set);
        evicted_addr = l2->sets[set].blocks[victim].tag
                       | (set << l2->blksize_shift);
    }

    /* Stamp and install new block. */
    fifo_update_blk(l2, set, victim);
    l2->sets[set].blocks[victim].tag   = tag;
    l2->sets[set].blocks[victim].valid = true;

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
 * fifo_times[way] does not need to be reset. fifo_get_evict_block() is only
 * called after get_invalid_block() returns -1 (all ways valid), so a
 * back-invalidated way (valid=false) is always found by get_invalid_block()
 * before fifo_get_evict_block() is reached. The stale fifo_times value is
 * overwritten by fifo_update_blk() when the way is next installed.
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
   *   Call l2_promote() to assert the inclusion invariant. Under FIFO, L1
   *   hits require no L2 metadata update (fifo_times is only stamped on
   *   miss installs). Return.
   *
   * L1 miss, L2 hit:
   *   l2_install_block() hits the L2 hit path — returns 0, no eviction, no
   *   fifo_times update. Refill L1 via access_cache(). Return.
   *
   * L1 miss, L2 miss:
   *   l2_install_block() runs the miss path — evicts the oldest L2 block if
   *   all ways are occupied, reconstructs evicted_addr, stamps and installs
   *   the new block. Back-invalidate L1 if evicted_addr != 0. Refill L1.
   *   Emit LogRecord into the shared memory ring.
   * ---------------------------------------------------------------------- */
  bool hit_in_l1 = access_cache(l1_dcaches[cache_idx], effective_addr);

  if (hit_in_l1) {
    /* L1 hit: assert inclusion invariant. No L2 metadata update (FIFO). */
    l2_promote(l2_dcaches[cache_idx], effective_addr);
    return;
  }

  /* L1 miss — count it regardless of what L2 does. */
  __atomic_fetch_add(&cpus[cpu_index].l1_miss_count, 1, __ATOMIC_RELAXED);

  /* L1 miss: consult L2. */
  bool hit_in_l2 = (in_cache(l2_dcaches[cache_idx], effective_addr) != -1);

  if (hit_in_l2) {
    /* L2 hit: l2_install_block() hit path is a no-op; refill L1. */
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

    fifo_destroy(cache);

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
   *   Replacement  : FIFO
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
   *   Replacement  : FIFO
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
