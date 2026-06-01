/*
===============================================================================
GEMINI LINGUISTIC ENGINE — Internal Structures
Shared definitions for core engine modules.
===============================================================================
*/

#ifndef GEMINI_INTERNAL_H
#define GEMINI_INTERNAL_H

/* On Linux, _GNU_SOURCE is a superset of _POSIX_C_SOURCE=200809L and also
 * enables pthread_rwlock_t, strdup, clock_gettime, and CLOCK_MONOTONIC.
 * It must be defined before any system header is included.  We set it here
 * only if the translation unit has not already set a conflicting macro.
 * Compilation flags should not set _POSIX_C_SOURCE to a value < 200809L;
 * if they do, _GNU_SOURCE takes precedence and supersedes it. */
#ifndef _GNU_SOURCE
#  define _GNU_SOURCE
#endif

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdatomic.h>
#include <string.h>
#include <sched.h>
#include <pthread.h>  /* Now safely exposes pthread_rwlock_t */
#include <math.h>  /* For isfinite() */
#include <stdio.h> /* For fprintf */
#include "csr_hypergraph.h"

/**
 * SymbolID - Unique identifier for a DAWG unit.
 */
typedef uint32_t SymbolID;

typedef struct DawgTransition {
    _Atomic SymbolID target;
    /* PRODUCTION FIX: Weight stored as IEEE754 float bit-cast via _Atomic uint32_t.
     * Uses atomic_store_float() and atomic_load_float() to avoid fractional truncation.
     * This preserves fractional learning rate increments (e.g., 0.5) that would
     * otherwise be lost if stored as integer. See weight_to_fixed_point/weight_from_fixed_point
     * for explicit fixed-point scaling when higher precision is needed. */
    _Atomic uint32_t weight;
    /* C-4 FIX: `next` promoted from plain uint32_t to _Atomic uint32_t.
     *
     * Rationale: add_dawg_transition writes t->next (list prepend) while
     * engine_step / dawg_predict / beam_search read t->next concurrently.
     * In C11, a concurrent non-atomic write and non-atomic read on the same
     * memory location is undefined behaviour regardless of hardware ordering.
     *
     * All read sites must use:
     * atomic_load_explicit(&t->next, memory_order_acquire)
     * All write sites must use:
     * atomic_store_explicit(&t->next, val, memory_order_release)
     *
     * The acquire on read pairs with the release in add_dawg_transition and
     * le_clear_symbol_transitions_safe, ensuring the full transition is visible
     * before the pointer to it is published. */
    _Atomic uint32_t next;  /* Index into dawg_transitions[] pool (INVALID = end) */
} DawgTransition;

/* --- PORTABLE ATOMIC FLOAT WRAPPERS ---
 *
 * FIX (Issue #7): The original wrappers called atomic_load/atomic_store with no
 * explicit memory order, defaulting to memory_order_seq_cst. On ARM and POWER
 * this emits a full memory barrier on every call, including the hot
 * transition-weight reads in le_beam_search and dawg_predict inner loops.
 *
 * Callers should use the cheapest ordering that still preserves correctness:
 *   - _relaxed: weight updates that are eventually consistent (e.g. CAS loops)
 *   - _acquire: first read in a dependent chain (e.g. loading t->weight before
 *               using it to compute a score that gates further reads)
 *   - _release: storing a newly computed weight for concurrent readers
 *   - (seq_cst variants kept for correctness-critical code that needs ordering)
 */
static inline float atomic_load_float(_Atomic uint32_t *p) {
    uint32_t bits = atomic_load(p);          /* seq_cst — safe default */
    float f; memcpy(&f, &bits, 4); return f;
}
static inline float atomic_load_float_acquire(_Atomic uint32_t *p) {
    uint32_t bits = atomic_load_explicit(p, memory_order_acquire);
    float f; memcpy(&f, &bits, 4); return f;
}
static inline float atomic_load_float_relaxed(_Atomic uint32_t *p) {
    uint32_t bits = atomic_load_explicit(p, memory_order_relaxed);
    float f; memcpy(&f, &bits, 4); return f;
}
static inline void atomic_store_float(_Atomic uint32_t *p, float v) {
    uint32_t bits; memcpy(&bits, &v, 4);
    atomic_store(p, bits);                   /* seq_cst — safe default */
}
static inline void atomic_store_float_release(_Atomic uint32_t *p, float v) {
    uint32_t bits; memcpy(&bits, &v, 4);
    atomic_store_explicit(p, bits, memory_order_release);
}
static inline void atomic_init_float(_Atomic uint32_t *p, float v) {
    uint32_t bits;
    memcpy(&bits, &v, 4);
    atomic_init(p, bits);
}

/* ------------------------------------------------------------------ */
/* SAFE_STRDUP — does NOT free caller memory                          */
/* ------------------------------------------------------------------ */
static inline char* gemini_safe_strdup(const char *s) {
    return (s && *s) ? strdup(s) : NULL;
}
#define SAFE_STRDUP(s) gemini_safe_strdup(s)

/**
 * atomic_fetch_add_float - Atomic add for floats using portable bit-casting.
 * Uses exponential backoff under contention to prevent livelock.
 *
 * FIX (Issue #18): The original code silently discarded the increment when the
 * livelock ceiling (MAX_YIELDS) was hit. For a learning system where edge
 * weights encode frequency-derived probability mass, silent drops corrupt the
 * trained model. We now fall back to a mutex-protected add so the value is
 * always committed.
 *
 * g_afa_fallback_mutex is module-internal; initialised by le_init().
 * Callers outside this translation unit should not touch it directly.
 */
extern pthread_mutex_t g_afa_fallback_mutex;

static inline void atomic_fetch_add_float(_Atomic uint32_t* p, float val) {
    if (!isfinite(val)) return;

    uint32_t old_bits = atomic_load_explicit(p, memory_order_relaxed);
    uint32_t new_bits;
    float old_val, new_val;
    uint32_t iterations = 0;
    uint32_t yields = 0;
    int backoff_count = 1;
    const uint32_t BACKOFF_THRESHOLD = 128;
    const uint32_t MAX_YIELDS = 10000;

    do {
        iterations++;
        if (iterations > BACKOFF_THRESHOLD) {
            for (int i = 0; i < backoff_count && yields < MAX_YIELDS; i++) {
                sched_yield();
                yields++;
            }
            backoff_count = (backoff_count < 1024) ? backoff_count * 2 : 1024;
            if (yields >= MAX_YIELDS) {
                /* FIX (Issue #18): Fall back to mutex-protected add rather than
                 * silently discarding. This guarantees the weight is committed. */
                pthread_mutex_lock(&g_afa_fallback_mutex);
                uint32_t cur = atomic_load_explicit(p, memory_order_relaxed);
                float curval; memcpy(&curval, &cur, 4);
                float newval = curval + val;
                uint32_t newbits; memcpy(&newbits, &newval, 4);
                atomic_store_explicit(p, newbits, memory_order_relaxed);
                pthread_mutex_unlock(&g_afa_fallback_mutex);
                return;
            }
        }
        memcpy(&old_val, &old_bits, 4);
        new_val = old_val + val;
        memcpy(&new_bits, &new_val, 4);
    } while (!atomic_compare_exchange_strong(p, &old_bits, new_bits));
}

/* ============================= CONFIGURATION =============================== */
#define MAX_NODES           2000000
#define MAX_EDGES           8000000
#define MAX_REVERSE_EDGES   (MAX_EDGES * 2)
#define MAX_SCCS            200000
/* MAX_SYMBOLS raised from 100,000 → 500,000.
 * Memory cost: +~17 MB for Symbol + GeminiSymbolDisk arrays (calloc'd in le_init).
 * Snapshot files written with this limit are NOT backward-compatible with
 * binaries compiled against the old 100,000 value — bump GEMINI_VERSION when
 * deploying to avoid silent mismatches on le_load_mmap. */
#define MAX_SYMBOLS         500000
#define MAX_BEAM_WIDTH      10
#define MAX_SEARCH_DEPTH    20
#define HASH_SIZE           400009  /* Prime number > MAX_NODES */
#define INVALID             0xFFFFFFFF
#define CLAIMED             0xFFFFFFFE  /* Sentinel for hash slot being claimed */
#define MAX_READERS         16      /* Maximum concurrent reader threads */

/* Initialization flags for EngineContext - prevent destroying uninitialized primitives */
#define INIT_FLAG_NODES          (1u << 0)
#define INIT_FLAG_EDGES          (1u << 1)
#define INIT_FLAG_SCC_NODES      (1u << 2)
#define INIT_FLAG_EDGE_POOL      (1u << 3)
#define INIT_FLAG_LARGE_SCRATCH  (1u << 4)
#define INIT_FLAG_SCRATCH_MUTEX  (1u << 5)
#define INIT_FLAG_LEXEME_LOCK    (1u << 6)
#define INIT_FLAG_POOL_LOCK      (1u << 7)
#define INIT_FLAG_EPOCH_RWLOCK   (1u << 8)  /* CONC-1 FIX: Epoch transition read-write lock */
#define MAX_IN_EDGES        1024    /* FIXED: Increased for Luganda's high fan-in prefixes (was 128, then 512) */

#define DEFAULT_EPOCH_SIZE          10000
#define DEFAULT_H_FORCED            0.1f  

#define TURBULENCE_THRESHOLD 1e-3f  /* Min delta to propagate upstream */
#define MAX_DEFERRED_EDGES   50000  /* Deferred cross-SCC edges buffer */
#define MAX_DAWG_TRANSITIONS 500000 /* Pool for DAWG symbol-to-symbol edges */
#define ENTROPY_BOUND_EPS    0.05f  /* MDL tolerance for promotion gate */

/* Default decay parameters */
#define DEFAULT_LAMBDA      0.999f
#define DEFAULT_THETA       1e-5f

/* PRODUCTION FIX: Fixed-point weight scaling to prevent fractional truncation.
 * Weights are stored as uint32_t fixed-point values scaled by WEIGHT_SCALE_FACTOR.
 * This allows fractional increments (e.g., 0.5) to accumulate correctly without
 * truncating to zero. Example: 1.5 * 1024 = 1536 (stored), then 1536/1024 = 1.5 (real). */
#define WEIGHT_SCALE_FACTOR 1024

/**
 * Convert a real-valued weight to fixed-point representation.
 * Example: weight_to_fixed_point(1.5) = 1536
 */
static inline uint32_t weight_to_fixed_point(float weight) {
    if (weight < 0.0f) return 0;
    if (weight > ((float)(UINT32_MAX / WEIGHT_SCALE_FACTOR))) {
        return UINT32_MAX;  /* Clamp to max representable value */
    }
    return (uint32_t)(weight * WEIGHT_SCALE_FACTOR);
}

/**
 * Convert a fixed-point weight back to real-valued representation.
 * Example: weight_from_fixed_point(1536) = 1.5
 */
static inline float weight_from_fixed_point(uint32_t fixed_weight) {
    return (float)fixed_weight / WEIGHT_SCALE_FACTOR;
}

/* ============================= TYPES ======================================= */
#include "libgemini.h"

typedef uint32_t EdgeID;
typedef uint32_t SccEdgeID;
typedef uint32_t EpochID;

#define GEMINI_ENGINE_TYPES_DEFINED

/* ============================= RELATIONAL HYPERGRAPH ======================= */

/**
 * Relation types for semantic edge encoding.
 * These distinguish syntactic connectors from semantic transitions.
 */
typedef enum {
    REL_NONE = 0,           /* No explicit relation (default adjacency) */
    REL_POSSESSIVE = 1,     /* "ya" - possession */
    REL_CONJUNCTION = 2,    /* "na" - and/with */
    REL_ASSOCIATIVE = 3,    /* "wa" - of/belonging */
    REL_LOCATIVE = 4,       /* "mu" - in/inside */
    REL_DIRECTIONAL = 5,    /* "ku" - to/towards */
    REL_INSTRUMENTAL = 6,   /* "nga" - by/with (instrumental) */
    REL_HONORIFIC = 7,      /* Honorific prefixes */
    REL_NEGATIVE = 8,       /* "te" - not */
    REL_TEMPORAL = 9,       /* Temporal markers */
    REL_CAUSAL = 10,        /* Causal connectors */
    REL_MAX = 65535         /* Sentinel for invalid relation */
} RelationType;

/**
 * Bit-Packed Relational Edge Design (64-bit Scalar)
 * ┌─────────────────────────┬───────────────────┬───────────────────┐
 * │   Destination Node ID   │   Relation Type   │   Packed Weight   │
 * │        (32 bits)        │     (16 bits)     │     (16 bits)     │
 * └─────────────────────────┴───────────────────┴───────────────────┘
 *
 * This design maintains lock-free atomicity with _Atomic uint64_t while
 * encoding semantic relation information for hypergraph traversal.
 */
typedef uint64_t PackedEdge;

/* Bit manipulation macros for packed edges */
#define EXTRACT_DST(e)      ((NodeID)((e) >> 32))
#define EXTRACT_REL(e)      ((uint16_t)(((e) >> 16) & 0xFFFF))
#define EXTRACT_WEIGHT(e)   (unpack_float_16((uint16_t)((e) & 0xFFFF)))
#define PACK_EDGE(dst, rel, w) \
    (((uint64_t)(dst) << 32) | ((uint64_t)(rel) << 16) | (pack_float_16(w)))

/**
 * pack_float_16 - Compress a 32-bit float to 16-bit representation.
 * Uses a fixed-point scaling approach suitable for edge weights.
 * Weights are typically in range [0, 100] for probability scores.
 * Scale factor: 65535 / 100 = 655.35
 */
static inline uint16_t pack_float_16(float weight) {
    if (!isfinite(weight) || weight < 0.0f) return 0;
    if (weight > 100.0f) return 0xFFFF;  /* Clamp to max */
    return (uint16_t)(weight * 655.35f);
}

/**
 * unpack_float_16 - Decompress 16-bit representation back to 32-bit float.
 * Inverse of pack_float_16.
 */
static inline float unpack_float_16(uint16_t packed) {
    return (float)packed / 655.35f;
}

/* ===============================================================================
 * LOG-QUANTIZED FIXED-POINT WEIGHT SPACE
 * HYPERGRAPH UPDATE: Logarithmic fixed-point representation for exponential decay
 * ===============================================================================
 * 
 * Dynamic Range: 
 * q = 0     => w = 2^-32  (~2.32e-10) -> Infinite tail resolution for decay
 * q = 65535 => w = 2^32   (~4.29e+09) -> Handles massive raw observation counts
 * 
 * Benefits:
 * - Exponential edge decay becomes integer subtraction instead of float multiplication
 * - Eliminates subnormal underflow and rounding bias in long-tail semantic links
 * - Enables SIMD-friendly saturated integer operations for decay passes
 */

/**
 * le_pack_weight_log - Compress float weight to 16-bit log-quantized representation.
 * Uses log-base-2 fixed-point with Q10.6 equivalent scaling.
 * 
 * Formula: q = (log2(w) + 32) * 1024
 * 
 * This maps:
 * - w = 2^-32  => q = 0
 * - w = 1.0    => q = 32768 (32 * 1024)
 * - w = 2^32   => q = 65535
 */
static inline uint16_t le_pack_weight_log(float w) {
    if (w <= 2.3283064e-10f) return 0; /* Safe boundary clamp for 2^-32 */
    
    float logv = log2f(w);
    if (logv < -32.0f) return 0;
    if (logv >= 31.999f) return 65535;
    
    return (uint16_t)((logv + 32.0f) * 1024.0f);
}

/**
 * le_unpack_weight_log - Decompress 16-bit log-quantized representation back to float.
 * Inverse of le_pack_weight_log.
 * 
 * Formula: w = 2^((q / 1024) - 32)
 */
static inline float le_unpack_weight_log(uint16_t q) {
    if (q == 0) return 0.0f;
    float logv = ((float)q / 1024.0f) - 32.0f;
    return exp2f(logv);
}

/**
 * le_decay_delta_q - Compute integer decay delta for a given decay factor.
 * 
 * For exponential decay w_{t+1} = lambda * w_t:
 * log2(w_{t+1}) = log2(lambda) + log2(w_t)
 * 
 * The decay delta is a static integer: delta_q = round(log2(lambda) * 1024)
 * 
 * Example: lambda = 0.95 => delta_q = round(log2(0.95) * 1024) = -76
 */
static inline int16_t le_decay_delta_q(float lambda) {
    if (lambda <= 0.0f || lambda > 1.0f) return 0;
    return (int16_t)roundf(log2f(lambda) * 1024.0f);
}

/* ===============================================================================
 * GRAMMAR CLASSIFICATION LOOKUP
 * HYPERGRAPH UPDATE: Bitfield-based grammatical role descriptors for context disambiguation
 * ===============================================================================
 * 
 * Provides O(1) lookup of grammatical metadata for tokens, enabling:
 * - Deterministic context disambiguation for hybrid tokens
 * - Constrained path pruning in beam search based on parts-of-speech
 * - Semantic enrichment during dictionary ingestion
 */

/* Bitfield for grammatical roles */
typedef uint16_t GrammarRole;
#define G_NONE        0x0000  /* No grammatical role assigned */
#define G_NOUN        0x0001  /* Noun - semantic manifold node */
#define G_VERB        0x0002  /* Verb - action or state */
#define G_ADJ         0x0004  /* Adjective - modifier */
#define G_PARTICLE    0x0008  /* Particle - grammatical connector */
#define G_GENITIVE    0x0010  /* Genitive marker - possession */
#define G_HYBRID      0x0020  /* High-ambiguity token requiring context disambiguation */
#define G_CONNECTOR   0x0040  /* Pure connector - skip node creation, use edge relation */
#define G_NEGATIVE    0x0080  /* Negation marker */
#define G_TEMPORAL    0x0100  /* Temporal marker */
#define G_HONORIFIC   0x0200  /* Honorific prefix/suffix */

typedef struct {
    GrammarRole *table;       /* Dynamic array mapping TokenID -> GrammarRole */
    uint32_t     capacity;    /* Table capacity (matches tokenizer vocab size) */
    _Atomic uint32_t initialized; /* Flag indicating table is ready for use */
} GrammarLookupTable;

/**
 * TokenClass - Classification for hybrid token handling.
 * Distinguishes between semantic nodes, pure connectors, and hybrid tokens.
 */
typedef enum {
    TOKEN_CLASS_SEMANTIC = 0,    /* Full semantic node - always instantiate */
    TOKEN_CLASS_CONNECTOR = 1,   /* Pure connector - skip node creation, use edge relation */
    TOKEN_CLASS_HYBRID = 2       /* Context-dependent - evaluate local window */
} TokenClass;

/**
 * TokenBehavior - Configuration for token classification.
 * Maps TokenID to its class and default relation type.
 */
typedef struct {
    TokenClass class_type;
    uint16_t default_relation;
} TokenBehavior;

/* ============================= CORE STRUCTURES ============================= */

/**
 * CompactorEpoch - QSBR state for lock-free reclamation
 */
typedef struct {
    _Atomic uint64_t global_epoch;
    _Atomic uint64_t reader_epochs[MAX_READERS];
} CompactorEpoch;

/**
 * GraphScratch — Small per-thread scratch for SCC-level traversals.
 * ~5 MB per thread (scc_vis widened to uint64_t).  Safe for _Thread_local.
 */
typedef struct {
    SccID    path_scratch[MAX_SCCS];
    SccID    collect_stack[MAX_SCCS];
    SccID    reachable_stack[MAX_SCCS];
    bool     visited_scratch[MAX_SCCS];
    /*
     * Generation stamps for le_scc_reachable.  uint64_t so the counter
     * (scc_vis_gen in EngineContext) never wraps — no inter-thread clearing
     * required on wrap-around.
     */
    uint64_t scc_vis[MAX_SCCS];
    uint32_t vis_stamp; /* Used for generation-stamp traversal (Issue #1) */
} GraphScratch;

/**
 * GraphScratchLarge — Node-sized scratch buffers (28 MB+).
 * Allocated once, protected by a mutex for single-writer paths.
 */
typedef struct {
    SccID    collect_stack[MAX_SCCS];
    bool     visited_scratch[MAX_SCCS];
    NodeID   queue_scratch[MAX_NODES];
    bool     in_queue_scratch[MAX_NODES];
    float    ext_weights_scratch[MAX_NODES];
    EdgeID   edge_ids[MAX_NODES];  /* For compactor_sweep to avoid malloc */
    uint32_t capacity;            /* General capacity flag */
    uint32_t edge_ids_capacity;   /* Capacity of edge_ids buffer specifically (MAX_EDGES) */

    /* Kosaraju SCC algorithm scratch buffers */
    NodeID   rev_heads[MAX_NODES];      /* Reverse adjacency list heads */
    EdgeID   rev_nexts[MAX_EDGES];     /* Reverse adjacency list next pointers */
    NodeID   rev_source[MAX_EDGES];    /* Reverse edge source nodes */
} GraphScratchLarge;

/**
 * EdgeSpan — Descriptor for a node's inbound edges in the global edge pool.
 * PHASE 2A: Replaces fixed `in_edges[MAX_IN_EDGES]` with dynamic allocation.
 * * Design invariants:
 * 1. pool_offset is a relative index (not a pointer) for mmap relocatability
 * 2. count <= capacity ensures amortized O(1) edge insertion via 1.5x growth
 * 3. All offsets are verified at initialization to prevent out-of-bounds access
 */
typedef struct {
    _Atomic uint32_t pool_offset;  /* Index into EngineContext.edge_pool.edges[] */
    _Atomic uint32_t count;        /* Current number of inbound edges (reader threads check this) */
    _Atomic uint32_t capacity;     /* Allocated slots in pool (for growth without realloc) */
} EdgeSpan;

/**
 * RelationalEdgePool — Parallel array layout for hypergraph edge storage.
 * HYPERGRAPH UPDATE: Isolates immutable structural headers from mutable weights
 * to eliminate cache line dirtying during high-frequency weight updates.
 * 
 * Layout:
 * - headers: Immutable (NodeID + RelationType) - only written during edge creation
 * - log_weights: Mutable 16-bit log-quantized weights - lock-free CAS updates
 * 
 * This design enables:
 * - Lock-free reads during beam search (headers never change after allocation)
 * - Reduced write contention (weights are isolated 16-bit words)
 * - SIMD-friendly decay operations (contiguous 16-bit array)
 */
typedef uint64_t PackedEdgeHeader;

/* Bit packing for PackedEdgeHeader:
 * [ Bits 63 ... 32 ] : Destination NodeID (32 bits)
 * [ Bits 31 ... 0  ] : Relation Type ID  (32 bits)
 */
#define PACK_EDGE_HEADER(dst, rel) (((uint64_t)(dst) << 32) | ((uint32_t)(rel)))
#define UNPACK_EDGE_DST(hdr)       ((NodeID)((hdr) >> 32))
#define UNPACK_EDGE_REL(hdr)       ((uint32_t)((hdr) & 0xFFFFFFFFU))

typedef struct {
    PackedEdgeHeader *headers;       /* Immutable structure (Read-Lock protected on resize) */
    _Atomic uint16_t *log_weights;   /* High-contention mutable weights (Lock-free updates) */
    uint32_t          capacity;      /* Max indexable edge slots */
    _Atomic uint32_t  pool_ptr;      /* Bump allocator index pointer */
    pthread_rwlock_t  pool_lock;     /* Exclusive lock held ONLY during pool resizing/relocation */
    _Atomic uint32_t  pool_utilization; /* Tracking metric for runtime performance tuning */
} RelationalEdgePool;

/**
 * NodeTransientState — Per-node runtime state, decoupled from disk.
 * PHASE 2A: in_edges replaced with EdgeSpan descriptor (64 bytes → ~40 bytes per node).
 *
 * FIX (Issue #3): scc_id and next_in_scc promoted to _Atomic uint32_t.
 * Both fields are written by the engine thread during SCC merges and read
 * concurrently by inference (engine_step), accessor (le_node_scc), and
 * beam-search paths. Declaring them plain uint32_t while using
 * atomic_load_explicit on them is C11 undefined behaviour (§7.17.3).
 * All read sites already use atomic_load_explicit with memory_order_acquire;
 * all write sites (merge code in libgemini.so) must use atomic_store_explicit
 * with memory_order_release.
 */
typedef struct {
    _Atomic uint32_t scc_id;        /* SccID: written on merge, read on every step */
    _Atomic uint32_t next_in_scc;   /* Linked-list next: written on merge, read in traversal */
    _Atomic uint64_t freq;
    EdgeSpan in_edges;  /* PHASE 2A: Pooled adjacency descriptor */
    _Atomic bool is_promoted;       /* CRITICAL FIX: Track promotion on nodes to prevent re-promotion */
    _Atomic SymbolID symbol_id;     /* Symbol ID assigned when node's SCC was promoted */
} NodeTransientState;

/* * Node struct has been completely removed to prevent accidental usage.
 * Use GeminiNodeDisk (disk format) + NodeTransientState (runtime) instead.
 */

/**
 * Edge - directed transition with decayed weight
 * * PRODUCTION FIX: Weight stored as IEEE754 float bit-cast via _Atomic uint32_t.
 * Uses atomic_fetch_add_float() for updates to preserve fractional increments.
 * This prevents truncation of soft learning-rate updates (e.g., 0.5 * freq) to zero.
 */
typedef struct {
    NodeID target;          /* Destination node */
    _Atomic uint32_t weight;   /* IEEE754 float bits (use atomic_load_float/atomic_store_float) */
    EdgeID next;            /* Next edge in adjacency list (index-based) */
} Edge;

/**
 * ReverseEdge - reverse adjacency list entry (pool-allocated)
 */
typedef struct {
    uint32_t source;
    uint32_t next;
} ReverseEdge;

/**
 * SccEdge - represents a directed edge in the SCC DAG
 *
 * R-2 FIX: `next` promoted to _Atomic SccEdgeID.
 * le_add_scc_edge writes `.next` as part of a CAS-loop list insertion,
 * while le_scc_reachable / le_collect_scc_path / le_merge_sccs read it
 * concurrently.  On weakly-ordered CPUs (ARM/POWER), the writer's store
 * may not be visible to concurrent readers without acquire/release pairing.
 * All read sites must use atomic_load_explicit with memory_order_acquire;
 * all write sites must use atomic_store_explicit with memory_order_release.
 */
typedef struct {
    SccID target_scc;       /* Target SCC ID in the SCC-DAG (Fixed: Was NodeID / SccEdgeID ambiguity) */
    _Atomic SccEdgeID next; /* R-2 FIX: Was plain SccEdgeID — concurrent read/write UB */
} SccEdge;

/**
 * SccNode - represents a strongly connected component
 *
 * FIX (Issues #2 & #13): Fields read on the hot inference path (is_promoted,
 * symbol_id, head) and during beam search (avg_entropy, coherence) are now
 * _Atomic. Previously they were plain types, written by the engine/epoch thread
 * and read concurrently by inference/accessor threads — a C11 data race producing
 * undefined behaviour on weakly-ordered CPUs (ARM/POWER).
 *
 * Float fields (coherence, avg_entropy) use the _Atomic uint32_t bit-cast
 * pattern already established for DawgTransition.weight. Use the inline helpers
 * scc_load_coherence() / scc_store_coherence() and their avg_entropy
 * equivalents at all read/write sites.
 *
 * Write sites in libgemini.so (promote_scc, le_merge_sccs, update_scc_metrics)
 * MUST be updated to use atomic_store_explicit with memory_order_release on
 * these fields.
 */
typedef struct {
    _Atomic NodeID   head;           /* First node in SCC — written on merge, read in engine_step */
    _Atomic NodeID   tail;           /* Last node in SCC — written on merge */
    _Atomic uint32_t member_count;   /* CRITICAL FIX: Made atomic for thread-safe concurrent reads */
    _Atomic SccEdgeID first_scc_edge;

    /* Structural metrics */
    _Atomic uint32_t internal_edges;
    _Atomic uint32_t external_edges;
    _Atomic uint32_t coherence_bits;   /* float bit-cast: use scc_load_coherence() */
    _Atomic uint32_t avg_entropy_bits; /* float bit-cast: use scc_load_avg_entropy() */

    /* Usage metrics */
    _Atomic uint64_t traversal_count;
    _Atomic uint64_t freq;

    /* Stability tracking */
    _Atomic EpochID first_seen;
    _Atomic EpochID last_modified;
    _Atomic uint32_t stable_epochs;
    _Atomic uint32_t last_member_count; /* Track member_count changes separately from edge changes */

    /* Promotion state — read on every engine_step(); must be atomic */
    _Atomic bool     is_promoted;
    _Atomic uint32_t symbol_id;    /* SymbolID */
    _Atomic bool     is_candidate;
    bool             is_weak;      /* Hairball detection (engine-thread only) */
    _Atomic bool     is_forced;    /* Deterministic transition (H -> 0) */
} SccNode;

/* Inline helpers for SccNode float bit-cast fields */
static inline float scc_load_coherence(const SccNode *s) {
    uint32_t bits = atomic_load_explicit(
        &((SccNode *)s)->coherence_bits, memory_order_acquire);
    float f; memcpy(&f, &bits, 4); return f;
}
static inline void scc_store_coherence(SccNode *s, float v) {
    uint32_t bits; memcpy(&bits, &v, 4);
    atomic_store_explicit(&s->coherence_bits, bits, memory_order_release);
}
static inline float scc_load_avg_entropy(const SccNode *s) {
    uint32_t bits = atomic_load_explicit(
        &((SccNode *)s)->avg_entropy_bits, memory_order_acquire);
    float f; memcpy(&f, &bits, 4); return f;
}
static inline void scc_store_avg_entropy(SccNode *s, float v) {
    uint32_t bits; memcpy(&bits, &v, 4);
    atomic_store_explicit(&s->avg_entropy_bits, bits, memory_order_release);
}

/**
 * SccMetric - Lightweight struct for candidate ranking and sorting.
 */
typedef struct {
    SccID scc_id;
    float rho;              /* Coherence */
    float h;                /* Entropy */
} SccMetric;

/**
 * Symbol - promoted SCC that acts as atomic unit in DAWG
 */
typedef struct {
    NodeID canonical_node;      /* CRITICAL FIX: Store canonical node instead of SCC slot index
                                 * SCC slots are ephemeral across Kosaraju runs, but the canonical
                                 * node provides stable identity for symbol resolution. */
    float stability_score;
    uint32_t          transition_count;   /* Number of outgoing DAWG edges */
    _Atomic uint32_t  first_transition;   /* Head of DAWG transition list (INVALID = none).
                                           * Writers publish via release store;
                                           * Readers (beam search) acquire via acquire load. */
    float entropy_delta;        /* ΔH = H_pre - H_post (learning amount) */
    SymbolID          free_list_next;      /* Next pointer for symbol_free_list (Treiber stack) */
} Symbol;

/**
 * ActiveState - distributed activation for semantic flexibility
 * Replaces single-node inference with spreading activation, beam diffusion,
 * and weighted resonance. This enables analogical jumps and better semantic
 * composition.
 */
typedef struct {
    NodeID node;
    float activation;
} ActiveState;

/**
 * Tagged pointer for ABA-safe lock-free operations
 * Combines a pointer/index with a generation counter to prevent ABA problems
 */
typedef struct {
    uint32_t ptr;      /* Pointer/index value */
    uint32_t tag;      /* Generation counter */
} TaggedPtr;

/* Tagged pointer operations */
static inline TaggedPtr make_tagged(uint32_t ptr, uint32_t tag) {
    TaggedPtr tp = {ptr, tag};
    return tp;
}

static inline uint64_t tagged_to_u64(TaggedPtr tp) {
    return ((uint64_t)tp.tag << 32) | (uint64_t)tp.ptr;
}

static inline TaggedPtr u64_to_tagged(uint64_t val) {
    TaggedPtr tp = {(uint32_t)(val & 0xFFFFFFFF), (uint32_t)(val >> 32)};
    return tp;
}

/* BeamState is defined in libgemini.h */

/**
 * GeminiNodeDisk - Compact memory-stable node representation.
 * (40 bytes exactly with padding)
 */
typedef struct {
    _Atomic uint32_t first_edge;
    _Atomic uint32_t edge_count;
    _Atomic uint32_t entropy_bits;  /* CRITICAL FIX: Made atomic for thread-safe concurrent reads (bit-cast) */
    _Atomic uint32_t total_mass;    /* Preserves M_i(t) decay state */
    float    turbulence;
    uint32_t token_id;      /* Preserve vocab mapping */
    /*
     * Issue 10 fix: last_written_step is read and written by apply_lazy_decay,
     * which can be called concurrently by le_add_edge and compactor_sweep on
     * the same node.  Making it _Atomic uint64_t prevents the torn-read UB on
     * 32-bit platforms and formalises the acquire/release ordering required for
     * the decay-factor computation to be correct.
     */
    _Atomic uint64_t last_written_step; /* Baseline step for decay */
    uint8_t  absorbing;
    uint8_t  reserved[15];      /* Pad to 48 bytes total */
} GeminiNodeDisk;

_Static_assert(sizeof(GeminiNodeDisk) == 48,
               "GeminiNodeDisk layout changed — update snapshot schema version");

typedef struct {
    uint32_t target;
    _Atomic uint32_t weight;
} GeminiEdgeDisk;

_Static_assert(sizeof(GeminiEdgeDisk) == 8,
               "GeminiEdgeDisk layout changed - update snapshot schema version");

/**
 * GeminiSymbolDisk - Compact symbol representation.
 */
typedef struct {
    uint32_t node_id;
    uint32_t string_offset;
} GeminiSymbolDisk;

typedef struct {
    _Atomic uint32_t from;
    uint32_t to;
} DeferredSccCheck;

/**
 * LexemeEntry - Canonical lexeme registry entry.
 * Maps surface forms to canonical lexeme IDs for aggregated statistics.
 */
typedef struct {
    uint64_t hash;           /* FNV-1a hash of surface form */
    uint32_t lexeme_id;      /* Canonical ID assigned */
    char    *surface;        /* Owned string (UTF-8) */
    _Atomic uint32_t freq;   /* Total occurrence count across all contexts */
} LexemeEntry;

/**
 * NodeLifecycle - Persistent lifecycle tracking anchored to invariant NodeID.
 * Decouples transient SCC topology from permanent promotion state to prevent
 * state contamination when graph partitioning reshuffles SCC array indices.
 */
typedef struct {
    _Atomic uint32_t stable_epochs;   /* Invariant age tracking anchored to the node */
    _Atomic SymbolID assigned_symbol; /* Symbol ID mapping, defaults to INVALID */
    _Atomic uint32_t lifecycle_flags; /* State bitmasks (e.g., PROMOTED_BIT) */
} NodeLifecycle;

/* Lifecycle flag bits */
#define LIFECYCLE_FLAG_PROMOTED (1U << 0)

/**
 * EngineContext - main engine state (memory-mappable)
 */
typedef struct EngineContext {
    /* Configuration */
    uint32_t magic;         /* Magic number for validation: 0x47454D49 "GEMI" */
    uint32_t version;
    float rho_min;
    float h_max;
    float h_forced;
    uint32_t min_freq;      /* Base frequency threshold */
    uint32_t promotion_epochs;
    _Atomic uint32_t promotion_budget; /* CRITICAL FIX: Made atomic for thread-safe atomic_load operations */

    /* CRITICAL FIX: Initialization flags to prevent destroying uninitialized primitives */
    uint32_t init_flags;

    /* Adaptive promotion thresholds */
    float base_freq;        /* Base frequency for dynamic threshold calculation */
    float semantic_decay;   /* Decay factor for DAWG transitions during epoch boundaries */

    /* Decay and Compaction Settings */
    _Atomic uint32_t ingestion_weight_modifier; /* Multiplier for incoming token/edge weights (float bitcast) */
    float lambda;           /* Decay factor λ ∈ (0, 1) */
    float theta;            /* Pruning threshold θ */
    uint32_t epoch_size;    /* Token count per epoch */
    _Atomic uint64_t current_step; /* Global token counter */
    CompactorEpoch epochs;  /* QSBR state */

    /* Counters (Atomic for thread-safe reporting/allocation) */
    _Atomic uint32_t node_count;
    _Atomic uint32_t edge_count;
    _Atomic uint32_t scc_count;
    _Atomic uint32_t scc_edge_count;
    _Atomic uint32_t symbol_count;
    uint32_t max_symbols;         /* Maximum capacity for dawg_nodes array */
    _Atomic uint32_t token_count;
    _Atomic EpochID current_epoch;  /* CRITICAL FIX: Made atomic for thread-safe reads (Issue #8) */
    _Atomic uint32_t epoch_in_progress; /* Bug 6 fix: CAS flag for epoch entry */
    _Atomic uint64_t edge_free_list_tagged; /* ABA-safe tagged free list for Edge structs */
    /*
     * Issue 3 fix: scc_edge_free_list was a plain SccEdgeID, producing a
     * pop/push race when le_add_scc_edge was called concurrently.  Changed to
     * _Atomic so the Treiber-stack CAS in le_add_scc_edge is correct.
     */
    _Atomic SccEdgeID scc_edge_free_list;
    /*
     * Generation counter for le_scc_reachable TLS visited arrays.
     * Widened to uint64_t so it never wraps in practice
     * (1.8×10¹⁹ calls before overflow).  Each thread stores stamps of
     * this type in its GraphScratch.scc_vis[]; no clearing on wrap needed.
     */
    _Atomic uint64_t scc_vis_gen;
    
    /*
     * CRITICAL FIX (Issue #7): Reader thread registration must check
     * that thread_id < MAX_READERS before storing in reader_epochs[].
     * Currently there is no visible guard — a 17th thread would overflow.
     * This should be enforced at registration time or via _Static_assert.
     */
    
    /* Mapped Data Regions (Zero-Copy) */
    GeminiNodeDisk* nodes;
    GeminiEdgeDisk* edges;
    GeminiSymbolDisk* symbols;
    _Atomic EdgeID* edge_nexts;  // CRITICAL FIX S-4: Made atomic for thread-safe concurrent reads

    /* Mmap Bookkeeping */
    void* map_base;
    size_t map_size;
    int fd;
    bool is_writable;
    bool nodes_are_mmap;
    _Atomic uint32_t re_pool_ptr;

    NodeTransientState* transient_nodes;
    NodeLifecycle* node_lifecycles; /* Persistent lifecycle tracking anchored to invariant NodeID */
    uint32_t* node_in_edges;        /* Flat slab: MAX_NODES * MAX_IN_EDGES uint32_t's -- DEPRECATED BY PHASE 2A */

    /* PHASE 2A: Global edge pool for pooled adjacency model */
    RelationalEdgePool edge_pool;    /* Replaces EdgePoolState; enables parallel array layout */

    /* HYPERGRAPH: Grammar classification lookup table for context disambiguation */
    GrammarLookupTable grammar_table; /* Maps TokenID -> GrammarRole bitfield */

    /* HYPERGRAPH: Relation modifiers for dynamic scoring in beam search */
    float relation_modifiers[REL_MAX + 1];  /* Per-relation scoring multipliers */
    
    /* HYPERGRAPH: CSR hypergraph for deterministic grammatical rule overrides */
    CSRHypergraph csr_hypergraph;  /* Memory-mapped CSR matrix for rule evaluation */

    /* Shared node-sized scratch (Bug 7: avoids 31 MB/thread TLS) */
    GraphScratchLarge* large_scratch;
    pthread_mutex_t    large_scratch_mutex;
    pthread_rwlock_t   epoch_rwlock;  /* CONC-1 FIX: Protects epoch transitions from concurrent ingestion */
    NodeID*            edge_source_scratch;  /* O(E) temp array for metrics recompute */

    SccNode* scc_nodes;
    SccEdge* scc_edges;             /* SCC-DAG edges */
    _Atomic NodeID* node_hash;      /* Hash table for O(1) lookup (properly atomic for thread-safety) */
    
    /* Deferred cross-SCC edges for batched cycle detection */
    DeferredSccCheck* deferred_scc_checks;
    _Atomic uint32_t deferred_scc_count;  /* CRITICAL FIX: Made atomic for thread-safe atomic_fetch_add/sub operations */
    Symbol* dawg_nodes;
    
    /* DAWG transition pool (flat array for mmap persistence) */
    DawgTransition* dawg_transitions;
    _Atomic uint32_t dawg_transition_free_list;  /* CRITICAL FIX: Made atomic (Issue #6) for correct Treiber-stack CAS in le_update_dawg_transitions */
    _Atomic uint32_t dawg_transition_count;   /* Allocated DAWG transitions */

    /* Reverse-edge pool (dynamic linked-list reverse adjacency) */
    ReverseEdge* reverse_edges;
    _Atomic uint32_t reverse_edge_free_list;  /* Head of reverse-edge free pool (CAS) */
    _Atomic uint32_t reverse_edge_count;     /* Allocated reverse edges */
    
    /* Global Metrics & Statistics - CRITICAL FIX: Made entropy fields atomic for thread-safe concurrent reads */
    _Atomic uint32_t global_entropy_bits;       /* H_global: weighted system entropy (bit-cast for atomic ops) */
    _Atomic uint32_t prev_global_entropy_bits;  /* Previous H for monotonicity check (bit-cast for atomic ops) */
    _Atomic uint64_t total_promotions;  /* CRITICAL FIX: Widened from uint32_t to uint64_t (Issue #9) to prevent wrap at 4.3B */
    _Atomic uint64_t total_merges;     /* CRITICAL FIX: Widened from uint32_t to uint64_t (Issue #9) to prevent wrap at 4.3B */

    /* R-5 FIX: Compact active symbol list for O(active) demotion sweeps.
     * Maintained by promote_scc (append) and demote_symbol (swap-remove).
     * le_demote_stale_symbols iterates only this list instead of all symbol_count slots. */
    SymbolID* active_symbols;
    _Atomic uint32_t active_symbol_count;
    _Atomic SymbolID symbol_free_list;  /* ADDED: For O(1) recycling of demoted symbol slots */

    /* Lexeme Canonicalization Registry
     * Maps identical surface forms to canonical lexeme IDs.
     * Enables aggregated statistics and proper SCC merging of duplicate lexemes.
     * Thread-safe: Read-write lock pattern in le_intern_lexeme(). */
    LexemeEntry *lexemes;           /* Dynamic array of interned lexemes */
    uint32_t     lexeme_count;      /* Number of unique lexemes */
    uint32_t     lexeme_capacity;   /* Allocated slots in lexemes array */
    pthread_rwlock_t lexeme_rwlock; /* Protects lexeme array growth/insertion */

    /* C-2 FIX: Striped mutex array for per-node edge insertion serialization
     * Prevents duplicate edges under concurrent ingestion without a global bottleneck.
     * Uses 1024 stripes with hash-based selection for cache-friendly locking. */
    pthread_mutex_t node_insert_locks[1024];
} EngineContext;

/* Inline accessor functions for grammar classification
 * Must be defined after EngineContext struct is complete */
static inline GrammarRole le_get_grammar_role(EngineContext *ctx, TokenID tid) {
    if (!ctx || tid >= ctx->grammar_table.capacity) return G_NONE;
    return ctx->grammar_table.table[tid];
}

static inline void le_update_grammar_role(EngineContext *ctx, TokenID tid, GrammarRole role) {
    if (ctx && tid < ctx->grammar_table.capacity) {
        ctx->grammar_table.table[tid] = role;
    }
}

/**
 * le_get_symbol_token_id - Get canonical TokenID for a Symbol (SCC)
 * Returns the TokenID of the canonical node, or 0 if invalid.
 * CRITICAL FIX: Updated to use canonical_node instead of original_scc.
 */
static inline TokenID le_get_symbol_token_id(EngineContext *ctx, SymbolID sym_id) {
    if (!ctx || sym_id >= MAX_SYMBOLS) return 0;
    Symbol *sym = &ctx->dawg_nodes[sym_id];
    NodeID canonical_node = sym->canonical_node;
    if (canonical_node == INVALID || canonical_node >= MAX_NODES) return 0;
    return atomic_load_explicit(&ctx->nodes[canonical_node].token_id, memory_order_acquire);
}

/**
 * le_disambiguate_hybrid_by_token - Deterministic state machine for hybrid tokens (TokenID version).
 * HYPERGRAPH: Resolves G_HYBRID tokens using grammatical context from TokenIDs.
 * This version avoids SymbolID/NodeID type collision by working directly with TokenIDs.
 *
 * Rules:
 * - If PREV is NOUN and NEXT is NOUN -> Token is GENITIVE (possession)
 * - If PREV is VERB and NEXT is NOUN -> Token is PARTICLE (grammatical connector)
 * - If NEXT is VERB or ADJ -> Token is CONNECTOR (or part of noun root)
 * - Default: Remove HYBRID flag, return base role
 *
 * @param ctx: Engine context with grammar lookup table
 * @param prev_tid: Previous token ID (0 if none)
 * @param curr_tid: Current token ID being evaluated
 * @param next_tid: Next token ID (0 if none)
 * @return: Resolved GrammarRole for the current token
 */
static inline GrammarRole le_disambiguate_hybrid_by_token(EngineContext *ctx, 
                                                         TokenID prev_tid, 
                                                         TokenID curr_tid, 
                                                         TokenID next_tid) {
    if (!ctx || curr_tid == 0) return G_NONE;
    
    GrammarRole role_curr = le_get_grammar_role(ctx, curr_tid);
    
    /* Only process tokens marked as G_HYBRID */
    if (!(role_curr & G_HYBRID)) return role_curr;
    
    /* Get predecessor context */
    GrammarRole role_prev = (prev_tid != 0) ? le_get_grammar_role(ctx, prev_tid) : G_NONE;
    
    /* Get successor context */
    GrammarRole role_next = (next_tid != 0) ? le_get_grammar_role(ctx, next_tid) : G_NONE;
    
    /* Transition Logic Table */
    /* NOUN + NOUN context -> Genitive marker (possession) */
    if ((role_prev & G_NOUN) && (role_next & G_NOUN)) {
        return G_GENITIVE;
    }
    
    /* VERB + NOUN context -> Particle (grammatical connector) */
    if ((role_prev & G_VERB) && (role_next & G_NOUN)) {
        return G_PARTICLE;
    }
    
    /* VERB or ADJ following -> Connector or part of compound */
    if (role_next & (G_VERB | G_ADJ)) {
        return G_CONNECTOR;
    }
    
    /* Default: Remove HYBRID flag, return base role */
    return role_curr & ~G_HYBRID;
}

/**
 * le_disambiguate_hybrid - Deterministic state machine for hybrid tokens.
 * HYPERGRAPH: Resolves G_HYBRID tokens using grammatical context from neighbors.
 *
 * Rules:
 * - If PREV is NOUN and NEXT is NOUN -> Token is GENITIVE (possession)
 * - If PREV is VERB and NEXT is NOUN -> Token is PARTICLE (grammatical connector)
 * - If NEXT is VERB or ADJ -> Token is CONNECTOR (or part of noun root)
 * - Default: Remove HYBRID flag, return base role
 *
 * @param ctx: Engine context with grammar lookup table
 * @param prev_sym_id: Previous symbol in sequence (INVALID if none)
 * @param curr_sym_id: Current symbol being evaluated
 * @param next_sym_id: Next symbol in sequence (INVALID if none)
 * @return: Resolved GrammarRole for the current token
 */
static inline GrammarRole le_disambiguate_hybrid(EngineContext *ctx, 
                                                  SymbolID prev_sym_id, 
                                                  SymbolID curr_sym_id, 
                                                  SymbolID next_sym_id) {
    if (!ctx || curr_sym_id >= MAX_SYMBOLS) return G_NONE;
    
    TokenID tid_curr = le_get_symbol_token_id(ctx, curr_sym_id);
    GrammarRole role_curr = le_get_grammar_role(ctx, tid_curr);
    
    /* Only process tokens marked as G_HYBRID */
    if (!(role_curr & G_HYBRID)) return role_curr;
    
    /* Get predecessor context */
    GrammarRole role_prev = G_NONE;
    if (prev_sym_id != INVALID && prev_sym_id < MAX_SYMBOLS) {
        TokenID tid_prev = le_get_symbol_token_id(ctx, prev_sym_id);
        role_prev = le_get_grammar_role(ctx, tid_prev);
    }
    
    /* Get successor context */
    GrammarRole role_next = G_NONE;
    if (next_sym_id != INVALID && next_sym_id < MAX_SYMBOLS) {
        TokenID tid_next = le_get_symbol_token_id(ctx, next_sym_id);
        role_next = le_get_grammar_role(ctx, tid_next);
    }
    
    /* Transition Logic Table */
    /* NOUN + NOUN context -> Genitive marker (possession) */
    if ((role_prev & G_NOUN) && (role_next & G_NOUN)) {
        return G_GENITIVE;
    }
    
    /* VERB + NOUN context -> Particle (grammatical connector) */
    if ((role_prev & G_VERB) && (role_next & G_NOUN)) {
        return G_PARTICLE;
    }
    
    /* VERB or ADJ following -> Connector or part of compound */
    if (role_next & (G_VERB | G_ADJ)) {
        return G_CONNECTOR;
    }
    
    /* Default: Remove HYBRID flag, return base role */
    return role_curr & ~G_HYBRID;
}

/* ============================= BINARY SNAPSHOT SCHEMA ====================== */

#define GEMINI_MAGIC   0x47454D49  /* "GEMI" */
/* GEMINI_VERSION bumped 1→2: MAX_SYMBOLS 100k→500k changes Symbol and
 * GeminiSymbolDisk array sizes. Snapshots from version 1 MUST NOT be loaded
 * by version 2 binaries — le_load_mmap rejects version mismatches.
 * GEMINI_VERSION bumped 2→3: Added max_symbols field to header for dynamic
 * symbol table capacity. Snapshots from version 2 MUST NOT be loaded by
 * version 3 binaries. */
/* R-5: Bumped 3→4: SccEdge.next is now _Atomic, active_symbols[] added to EngineContext.
 * Snapshots from version 3 MUST NOT be loaded by version 4 binaries. */
/* Bumped 4→5: Added DAWG and SCC structures to snapshot for prediction support.
 * Snapshots from version 4 MUST NOT be loaded by version 5 binaries. */
#define GEMINI_VERSION 6

/**
 * GeminiSnapshotHeader - Fixed-layout file header for zero-copy mmap.
 */
typedef struct {
    uint32_t magic;
    uint32_t version;
    uint32_t node_count;
    uint32_t edge_count;
    uint32_t symbol_count;
    uint32_t max_symbols;   /* Dynamic capacity for symbol table */
    uint32_t scc_count;     /* Version 5: SCC count for loading SCC structures */
    uint32_t current_epoch;
    uint64_t current_step;
    uint32_t token_count;
    uint32_t promotion_budget; /* Replaces reserved field for mmap compatibility */

    uint32_t hdr_size;      /* Alignment padding offset to data */
    uint64_t node_offset;   /* From file start */
    uint64_t edge_offset;
    uint64_t edge_nexts_offset; /* CRITICAL FIX CR14: Added edge_nexts for linked-list adjacency */
    uint64_t symbol_offset;
    uint64_t string_offset;
    uint64_t dawg_nodes_offset;   /* Version 5: DAWG structures for predictions */
    uint64_t dawg_transitions_offset;
    uint64_t scc_nodes_offset;    /* Version 5: SCC structures for predictions */
    uint64_t scc_edges_offset;
    uint64_t transient_nodes_offset; /* Version 5: Transient state for predictions */
    uint64_t node_lifecycles_offset; /* Version 6: Node lifecycle tracking for invariant promotion state */

    uint64_t total_transitions;
    uint32_t dawg_transition_count; /* C-6 FIX: Track actual DAWG transition count for snapshot persistence */
} GeminiSnapshotHeader;


/* PHASE 2A: Edge pool allocator functions */

/**
 * Initialize the global edge pool at engine startup.
 * Allocates pool_size slots in memory for inbound edge storage.
 * Returns 0 on success, -1 on failure.
 */
int edge_pool_init(EngineContext* ctx, uint32_t pool_size);

/**
 * Shut down the edge pool and free backing memory.
 */
void edge_pool_destroy(EngineContext* ctx);

/**
 * Allocate a span of N edge slots in the global pool.
 * Uses bump allocator for O(1) allocation during ingestion.
 * Returns pool offset, or (uint32_t)-1 on exhaustion.
 *
 * Thread-safe: Uses atomic_fetch_add on pool_ptr.
 */
uint32_t edge_pool_allocate(EngineContext* ctx, uint32_t capacity);

/**
 * Initialize a node's EdgeSpan with initial allocation.
 * Called once per node during ingestion; after this, capacity grows via realloc.
 */
int edge_span_init(EngineContext* ctx, NodeID node_id, uint32_t initial_capacity);

/**
 * Add an edge to a node's inbound edge set, reallocating if necessary.
 * Maintains capacity invariant: count <= capacity (amortized O(1)).
 */
int le_add_incoming_edge_pooled(EngineContext* ctx, NodeID target, NodeID source);

/**
 * Report edge pool statistics for monitoring.
 */
void edge_pool_stats(EngineContext* ctx);

/* Promotion logic imported from promotion_layer */
SymbolID promote_scc(EngineContext *ctx, SccID scc_id, float total_sys_mass);
void le_update_dawg_transitions(EngineContext *ctx);
void le_clear_symbol_transitions_safe(EngineContext *ctx, SymbolID sym);
void demote_symbol(EngineContext *ctx, SymbolID sym_id);

/**
 * Dynamically expands the symbol vocabulary capacity when saturation thresholds are met.
 * Returns the new capacity, or 0 on failure.
 */
uint32_t engine_expand_symbols(EngineContext *ctx);

/**
 * Scan for and demote symbols whose cognitive tracking metrics have dropped below operational limits.
 * Returns the total number of symbols demoted during this pass.
 */
uint32_t le_demote_stale_symbols(EngineContext *ctx);

/* Bulk edge addition for optimized prior injection */
void le_add_edge_bulk(EngineContext* ctx, NodeID from, NodeID to, float weight_to_add);

/**
 * Structural Edge Gating Pre-Filter
 * Blocks cross-term bridges (like words trailing punctuation) from linking into core SCC matrices.
 */
bool le_should_gate_structural_edge(EngineContext* ctx, int tok_handle, TokenID from_tok, TokenID to_tok);

/**
 * debug_audit_memory_integrity - Audit Symbol -> Canonical Node -> TokenID -> Text lineage
 *
 * Traces the full memory lineage from Symbol through canonical node to tokenizer
 * string pool to detect UTF-8 corruption, misaligned offsets, and garbage memory.
 * This diagnostic identifies tokenizer serialization misalignment issues where
 * id_to_offset arrays point to invalid memory addresses.
 */
void debug_audit_memory_integrity(EngineContext* ctx, int tok_handle);

#endif /* GEMINI_INTERNAL_H */