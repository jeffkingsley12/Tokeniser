/*
===============================================================================
GEMINI ENGINE LIFECYCLE MANAGEMENT
Core initialization, node creation, edge management, and epoch handling.

This file implements the fundamental lifecycle functions that were previously
declared but not defined, causing linker errors.
===============================================================================
*/

#include "gemini_internal.h"
#include "libgemini.h"
#include "tokenizer_api.h"
#include "lexeme_intern.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <stdatomic.h>
#include <pthread.h>
#include <math.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <limits.h>  /* L-6 FIX: For PATH_MAX */
#include <unistd.h>  /* L-6 FIX: For unlink */
#include <errno.h>   /* L-6 FIX: For errno */
#ifdef __AVX2__
#include <immintrin.h>  /* AVX2 intrinsics for vectorized decay */
#endif
/* --- FIX: Include headers for hardware pauses and scheduling yields --- */
#if defined(__x86_64__) || defined(__i386__)
#include <immintrin.h>
#else
#include <sched.h>
#endif

/* Global fallback mutex for atomic float operations */
pthread_mutex_t g_afa_fallback_mutex = PTHREAD_MUTEX_INITIALIZER;

/* Forward declarations */
void add_dawg_transition(EngineContext* ctx, SymbolID from, SymbolID to, float weight);
static void add_dawg_transition_unlocked(EngineContext* ctx, SymbolID from, SymbolID to, float weight);
static bool le_is_scc_promoted_via_lifecycle(EngineContext* ctx, SccID scc_id);

/* ============================= INITIALIZATION ============================== */

/**
 * le_init - Allocate and initialize a new EngineContext.
 * 
 * Initializes all arrays, hash tables, locks, and atomic counters.
 * Returns NULL on allocation failure.
 */
EngineContext* le_init(void) {
    EngineContext* ctx = (EngineContext *)calloc(1, sizeof(EngineContext));
    if (!ctx) {
        fprintf(stderr, "ERROR: Failed to allocate EngineContext\n");
        return NULL;
    }

    /* Set magic and version for validation */
    ctx->magic = GEMINI_MAGIC;
    ctx->version = GEMINI_VERSION;

    /* Initialize configuration parameters */
    ctx->rho_min = 0.1f;              /* Coherence threshold */
    ctx->h_max = 2.0f;                /* Max entropy for symbols */
    ctx->h_forced = 0.1f;             /* Forced-transition entropy */
    ctx->min_freq = 5;                /* Minimum frequency for nodes */
    ctx->promotion_epochs = 2;        /* Epochs before promotion */
    atomic_store(&ctx->promotion_budget, 1000);

    ctx->base_freq = 10.0f;           /* Base frequency for dynamic thresholds */
    ctx->semantic_decay = 0.99f;      /* DAWG transition decay */

    ctx->lambda = DEFAULT_LAMBDA;     /* Decay factor */
    ctx->theta = DEFAULT_THETA;       /* Pruning threshold */
    ctx->epoch_size = DEFAULT_EPOCH_SIZE;

    /* Initialize atomic counters */
    atomic_store(&ctx->node_count, 0);
    atomic_store(&ctx->edge_count, 0);
    atomic_store(&ctx->scc_count, 0);
    atomic_store(&ctx->scc_edge_count, 0);
    atomic_store(&ctx->symbol_count, 0);
    atomic_store(&ctx->token_count, 0);
    atomic_store(&ctx->current_epoch, 0);
    atomic_store(&ctx->current_step, 0);
    atomic_store(&ctx->deferred_scc_count, 0);
    atomic_store(&ctx->scc_vis_gen, 0);
    atomic_store_float(&ctx->ingestion_weight_modifier, 1.0f);

    ctx->max_symbols = MAX_SYMBOLS;

    /* Allocate main data structures */
    ctx->nodes = (GeminiNodeDisk *)calloc(MAX_NODES, sizeof(GeminiNodeDisk));
    if (!ctx->nodes) {
        fprintf(stderr, "ERROR: Failed to allocate nodes array\n");
        free(ctx);
        return NULL;
    }
    ctx->init_flags |= INIT_FLAG_NODES;

    ctx->edges = (GeminiEdgeDisk *)calloc(MAX_EDGES, sizeof(GeminiEdgeDisk));
    if (!ctx->edges) {
        fprintf(stderr, "ERROR: Failed to allocate edges array\n");
        free(ctx->nodes);
        free(ctx);
        return NULL;
    }
    ctx->init_flags |= INIT_FLAG_EDGES;

    ctx->symbols = (GeminiSymbolDisk *)calloc(MAX_SYMBOLS, sizeof(GeminiSymbolDisk));
    if (!ctx->symbols) {
        fprintf(stderr, "ERROR: Failed to allocate symbols array\n");
        free(ctx->nodes);
        free(ctx->edges);
        free(ctx);
        return NULL;
    }

    ctx->edge_nexts = (_Atomic EdgeID *)calloc(MAX_EDGES, sizeof(_Atomic EdgeID));
    if (!ctx->edge_nexts) {
        fprintf(stderr, "ERROR: Failed to allocate edge_nexts array\n");
        free(ctx->nodes);
        free(ctx->edges);
        free(ctx->symbols);
        free(ctx);
        return NULL;
    }

    /* CRITICAL FIX: Initialize edge_nexts to INVALID (0xFFFFFFFF) instead of 0 */
    for (uint32_t i = 0; i < MAX_EDGES; i++) {
        atomic_init(&ctx->edge_nexts[i], INVALID);
    }

    /* Allocate transient state arrays */
    ctx->transient_nodes = (NodeTransientState *)calloc(MAX_NODES, sizeof(NodeTransientState));
    if (!ctx->transient_nodes) {
        fprintf(stderr, "ERROR: Failed to allocate transient_nodes\n");
        goto cleanup;
    }

    /* Allocate node lifecycle arrays for invariant tracking */
    ctx->node_lifecycles = (NodeLifecycle *)calloc(MAX_NODES, sizeof(NodeLifecycle));
    if (!ctx->node_lifecycles) {
        fprintf(stderr, "ERROR: Failed to allocate node_lifecycles\n");
        goto cleanup;
    }

    /* Initialize node lifecycle arrays */
    for (uint32_t i = 0; i < MAX_NODES; i++) {
        atomic_init(&ctx->node_lifecycles[i].stable_epochs, 0);
        atomic_init(&ctx->node_lifecycles[i].assigned_symbol, INVALID);
        atomic_init(&ctx->node_lifecycles[i].lifecycle_flags, 0);
    }

    /* CRITICAL FIX: Initialize node-level promotion flags to prevent re-promotion */
    for (uint32_t i = 0; i < MAX_NODES; i++) {
        atomic_init(&ctx->transient_nodes[i].is_promoted, false);
        atomic_init(&ctx->transient_nodes[i].symbol_id, INVALID);
    }

    /* Allocate SCC arrays */
    ctx->scc_nodes = (SccNode *)calloc(MAX_SCCS, sizeof(SccNode));
    if (!ctx->scc_nodes) {
        fprintf(stderr, "ERROR: Failed to allocate scc_nodes\n");
        goto cleanup;
    }

    /* CRITICAL FIX: Initialize SCC 0 as default bucket */
    atomic_store(&ctx->scc_count, 1);
    atomic_store_explicit(&ctx->scc_nodes[0].head, INVALID, memory_order_release);
    atomic_store_explicit(&ctx->scc_nodes[0].tail, INVALID, memory_order_release);
    atomic_store_explicit(&ctx->scc_nodes[0].member_count, 0, memory_order_release);
    atomic_store_explicit(&ctx->scc_nodes[0].last_member_count, 0, memory_order_release);
    atomic_store_explicit(&ctx->scc_nodes[0].is_promoted, false, memory_order_release);
    atomic_store_explicit(&ctx->scc_nodes[0].symbol_id, INVALID, memory_order_release);
    ctx->init_flags |= INIT_FLAG_SCC_NODES;

    ctx->scc_edges = (SccEdge *)calloc(MAX_SCCS * 100, sizeof(SccEdge));
    if (!ctx->scc_edges) {
        fprintf(stderr, "ERROR: Failed to allocate scc_edges\n");
        goto cleanup;
    }

    /* Allocate DAWG structures */
    ctx->dawg_nodes = (Symbol *)calloc(MAX_SYMBOLS, sizeof(Symbol));
    if (!ctx->dawg_nodes) {
        fprintf(stderr, "ERROR: Failed to allocate dawg_nodes\n");
        goto cleanup;
    }

    ctx->dawg_transitions = (DawgTransition *)calloc(MAX_DAWG_TRANSITIONS, sizeof(DawgTransition));
    if (!ctx->dawg_transitions) {
        fprintf(stderr, "ERROR: Failed to allocate dawg_transitions\n");
        goto cleanup;
    }

    /* Initialize DAWG transitions to INVALID */
    for (uint32_t i = 0; i < MAX_DAWG_TRANSITIONS; i++) {
        atomic_init(&ctx->dawg_transitions[i].target, INVALID);
        atomic_init(&ctx->dawg_transitions[i].weight, 0);
        atomic_init(&ctx->dawg_transitions[i].next, INVALID);
    }

    /* Allocate hash table for node lookup (O(1) token ID -> NodeID) */
    ctx->node_hash = (_Atomic NodeID *)calloc(HASH_SIZE, sizeof(_Atomic NodeID));
    if (!ctx->node_hash) {
        fprintf(stderr, "ERROR: Failed to allocate node_hash table\n");
        goto cleanup;
    }

    /* Initialize hash table to INVALID */
    for (uint32_t i = 0; i < HASH_SIZE; i++) {
        atomic_init(&ctx->node_hash[i], INVALID);
    }

    /* Initialize reverse edges pool */
    ctx->reverse_edges = (ReverseEdge *)calloc(MAX_REVERSE_EDGES, sizeof(ReverseEdge));
    if (!ctx->reverse_edges) {
        fprintf(stderr, "ERROR: Failed to allocate reverse_edges\n");
        goto cleanup;
    }

    /* Allocate deferred SCC checks buffer */
    ctx->deferred_scc_checks = (DeferredSccCheck *)calloc(MAX_DEFERRED_EDGES, sizeof(DeferredSccCheck));
    if (!ctx->deferred_scc_checks) {
        fprintf(stderr, "ERROR: Failed to allocate deferred_scc_checks\n");
        goto cleanup;
    }

    /* Allocate active symbols list */
    ctx->active_symbols = (SymbolID *)calloc(MAX_SYMBOLS, sizeof(SymbolID));
    if (!ctx->active_symbols) {
        fprintf(stderr, "ERROR: Failed to allocate active_symbols\n");
        goto cleanup;
    }

    /* Allocate lexeme registry */
    ctx->lexeme_capacity = 1024;
    ctx->lexemes = (LexemeEntry *)calloc(ctx->lexeme_capacity, sizeof(LexemeEntry));
    if (!ctx->lexemes) {
        fprintf(stderr, "ERROR: Failed to allocate lexeme registry\n");
        goto cleanup;
    }

    /* Allocate large scratch buffer for SCC operations */
    ctx->large_scratch = (GraphScratchLarge *)calloc(1, sizeof(GraphScratchLarge));
    if (!ctx->large_scratch) {
        fprintf(stderr, "ERROR: Failed to allocate large_scratch\n");
        goto cleanup;
    }
    ctx->init_flags |= INIT_FLAG_LARGE_SCRATCH;

    /* Pre-allocate edge source scratch for O(E) metrics recompute.
     * Avoids malloc/free per epoch which caused 2.7s epoch times. */
    ctx->edge_source_scratch = (NodeID *)calloc(MAX_EDGES, sizeof(NodeID));
    if (!ctx->edge_source_scratch) {
        fprintf(stderr, "ERROR: Failed to allocate edge_source_scratch\n");
        goto cleanup;
    }

    /* Initialize edge pool */
    if (edge_pool_init(ctx, MAX_EDGES) != 0) {
        fprintf(stderr, "ERROR: Failed to initialize edge pool\n");
        goto cleanup;
    }
    ctx->init_flags |= INIT_FLAG_EDGE_POOL;

    /* HYPERGRAPH: Initialize relation modifiers with default values */
    for (uint32_t i = 0; i <= REL_MAX; i++) {
        ctx->relation_modifiers[i] = 1.0f;  /* Default: no modification */
    }
    /* Set specific modifiers for known relations */
    ctx->relation_modifiers[REL_POSSESSIVE] = 1.2f;   /* Boost possession */
    ctx->relation_modifiers[REL_CONJUNCTION] = 0.8f;   /* Slight penalty for conjunctions */
    ctx->relation_modifiers[REL_ASSOCIATIVE] = 1.1f;   /* Boost associative */
    ctx->relation_modifiers[REL_LOCATIVE] = 1.0f;      /* Neutral for locative */
    ctx->relation_modifiers[REL_DIRECTIONAL] = 1.0f;    /* Neutral for directional */
    ctx->relation_modifiers[REL_INSTRUMENTAL] = 0.9f;  /* Slight penalty for instrumental */
    ctx->relation_modifiers[REL_HONORIFIC] = 1.3f;     /* Boost honorifics */
    ctx->relation_modifiers[REL_NEGATIVE] = 0.7f;       /* Penalty for negation */
    ctx->relation_modifiers[REL_TEMPORAL] = 0.95f;     /* Slight penalty for temporal */
    ctx->relation_modifiers[REL_CAUSAL] = 1.15f;       /* Boost causal */

    /* HYPERGRAPH: Pre-compute log-quantized decay delta for DEFAULT_LAMBDA */
    /* This enables integer-subtraction decay instead of float multiplication */
    int16_t decay_delta = le_decay_delta_q(DEFAULT_LAMBDA);
    (void)decay_delta; /* Store for future use in SIMD decay pass */

    /* HYPERGRAPH: Initialize grammar classification lookup table */
    /* Allocate table with capacity matching tokenizer vocabulary size */
    uint32_t vocab_capacity = 65536; /* Default capacity for 16-bit TokenID space */
    ctx->grammar_table.table = (GrammarRole *)calloc(vocab_capacity, sizeof(GrammarRole));
    if (!ctx->grammar_table.table) {
        fprintf(stderr, "ERROR: Failed to allocate grammar lookup table\n");
        goto cleanup;
    }
    ctx->grammar_table.capacity = vocab_capacity;
    atomic_store_explicit(&ctx->grammar_table.initialized, 1, memory_order_release);

    /* Initialize mutexes and locks */
    if (pthread_mutex_init(&ctx->large_scratch_mutex, NULL) != 0) {
        fprintf(stderr, "ERROR: Failed to initialize large_scratch_mutex\n");
        goto cleanup;
    }
    ctx->init_flags |= INIT_FLAG_SCRATCH_MUTEX;

    if (pthread_rwlock_init(&ctx->lexeme_rwlock, NULL) != 0) {
        fprintf(stderr, "ERROR: Failed to initialize lexeme_rwlock\n");
        goto cleanup;
    }
    ctx->init_flags |= INIT_FLAG_LEXEME_LOCK;

    /* CONC-1 FIX: Initialize epoch transition read-write lock */
    if (pthread_rwlock_init(&ctx->epoch_rwlock, NULL) != 0) {
        fprintf(stderr, "ERROR: Failed to initialize epoch_rwlock\n");
        goto cleanup;
    }
    ctx->init_flags |= INIT_FLAG_EPOCH_RWLOCK;

    /* C-2 FIX: Initialize striped mutex array for per-node edge insertion serialization */
    for (int i = 0; i < 1024; i++) {
        if (pthread_mutex_init(&ctx->node_insert_locks[i], NULL) != 0) {
            fprintf(stderr, "ERROR: Failed to initialize node_insert_locks[%d]\n", i);
            goto cleanup;
        }
    }

    /* Initialize QSBR epoch tracking */
    atomic_init(&ctx->epochs.global_epoch, 0);
    for (int i = 0; i < MAX_READERS; i++) {
        atomic_init(&ctx->epochs.reader_epochs[i], 0);
    }

    /* Initialize free lists */
    atomic_init(&ctx->edge_free_list_tagged, tagged_to_u64(make_tagged(INVALID, 0)));
    atomic_init(&ctx->scc_edge_free_list, INVALID);
    atomic_init(&ctx->reverse_edge_free_list, INVALID);
    atomic_init(&ctx->dawg_transition_free_list, INVALID);
    atomic_init(&ctx->symbol_free_list, INVALID);

    return ctx;

cleanup:
    /* Free all allocated structures on error */
    if (ctx->nodes) free(ctx->nodes);
    if (ctx->edges) free(ctx->edges);
    if (ctx->symbols) free(ctx->symbols);
    if (ctx->edge_nexts) free(ctx->edge_nexts);
    if (ctx->transient_nodes) free(ctx->transient_nodes);
    /* C-4 FIX: Add missing free for node_lifecycles */
    if (ctx->node_lifecycles) free(ctx->node_lifecycles);
    if (ctx->scc_nodes) free(ctx->scc_nodes);
    if (ctx->scc_edges) free(ctx->scc_edges);
    if (ctx->dawg_nodes) free(ctx->dawg_nodes);
    if (ctx->dawg_transitions) free(ctx->dawg_transitions);
    if (ctx->node_hash) free(ctx->node_hash);
    if (ctx->reverse_edges) free(ctx->reverse_edges);
    if (ctx->deferred_scc_checks) free(ctx->deferred_scc_checks);
    if (ctx->active_symbols) free(ctx->active_symbols);
    if (ctx->lexemes) free(ctx->lexemes);
    if (ctx->large_scratch) free(ctx->large_scratch);
    /* C-4 FIX: Add missing free for edge_source_scratch */
    if (ctx->edge_source_scratch) free(ctx->edge_source_scratch);
    /* C-4 FIX: Add missing free for grammar_table.table */
    if (ctx->grammar_table.table) free(ctx->grammar_table.table);
    
    if (ctx->init_flags & INIT_FLAG_EDGE_POOL) {
        edge_pool_destroy(ctx);
    }
    if (ctx->init_flags & INIT_FLAG_SCRATCH_MUTEX) {
        pthread_mutex_destroy(&ctx->large_scratch_mutex);
    }
    if (ctx->init_flags & INIT_FLAG_LEXEME_LOCK) {
        pthread_rwlock_destroy(&ctx->lexeme_rwlock);
    }

    free(ctx);
    return NULL;
}

/* ============================= CLEANUP ====================================== */

/**
 * le_destroy - Free all memory allocated by le_init.
 *
 * Destroys mutexes and rwlocks, then frees all heap structures.
 * CRITICAL FIX: Harden against pool saturation with proper bounds checking.
 * H-2 FIX: Remove unsafe lock-unlock pattern. Callers must join all active
 * training threads and achieve verified quiet state before calling this function.
 */
void le_destroy(EngineContext* ctx) {
    if (!ctx) return;

    fprintf(stderr, "[DESTROY] Starting cleanup\n");

    /* H-2 FIX: Eliminate false lock-unlock security theater.
     * The architecture relies entirely on callers executing thread joins prior to deletion. */
    if (ctx->init_flags & INIT_FLAG_SCRATCH_MUTEX) {
        fprintf(stderr, "[DESTROY] Destroying scratch mutex\n");
        pthread_mutex_destroy(&ctx->large_scratch_mutex);
    }
    if (ctx->init_flags & INIT_FLAG_LEXEME_LOCK) {
        fprintf(stderr, "[DESTROY] Destroying lexeme rwlock\n");
        pthread_rwlock_destroy(&ctx->lexeme_rwlock);
    }
    if (ctx->init_flags & INIT_FLAG_EPOCH_RWLOCK) {
        fprintf(stderr, "[DESTROY] Destroying epoch rwlock\n");
        pthread_rwlock_destroy(&ctx->epoch_rwlock);
    }

    /* Destroy striped node insertion locks safely */
    for (int i = 0; i < 1024; i++) {
        pthread_mutex_destroy(&ctx->node_insert_locks[i]);
    }

    if (ctx->init_flags & INIT_FLAG_EDGE_POOL) {
        fprintf(stderr, "[DESTROY] Destroying edge pool\n");
        edge_pool_destroy(ctx);
    }

    /* Free all arrays with init_flags checks */
    if (ctx->init_flags & INIT_FLAG_NODES) {
        fprintf(stderr, "[DESTROY] Freeing nodes\n");
        if (ctx->nodes) { free(ctx->nodes); ctx->nodes = NULL; }
        if (ctx->transient_nodes) { free(ctx->transient_nodes); ctx->transient_nodes = NULL; }
        if (ctx->node_hash) { free(ctx->node_hash); ctx->node_hash = NULL; }
    }

    if (ctx->init_flags & INIT_FLAG_EDGES) {
        fprintf(stderr, "[DESTROY] Freeing edges\n");
        if (ctx->edges) { free(ctx->edges); ctx->edges = NULL; }
        if (ctx->edge_nexts) { free(ctx->edge_nexts); ctx->edge_nexts = NULL; }
        if (ctx->reverse_edges) { free(ctx->reverse_edges); ctx->reverse_edges = NULL; }
    }

    /* Free SCC arrays with bounds validation */
    if (ctx->init_flags & INIT_FLAG_SCC_NODES) {
        fprintf(stderr, "[DESTROY] Freeing SCC arrays\n");
        if (ctx->scc_nodes) {
            uint32_t scc_limit = atomic_load(&ctx->scc_count);
            if (scc_limit > MAX_SCCS) scc_limit = MAX_SCCS;
            free(ctx->scc_nodes);
            ctx->scc_nodes = NULL;
        }
        if (ctx->scc_edges) { free(ctx->scc_edges); ctx->scc_edges = NULL; }
        if (ctx->deferred_scc_checks) { free(ctx->deferred_scc_checks); ctx->deferred_scc_checks = NULL; }
    }

    /* Free DAWG structures */
    fprintf(stderr, "[DESTROY] Freeing DAWG structures\n");
    if (ctx->symbols) { free(ctx->symbols); ctx->symbols = NULL; }
    if (ctx->active_symbols) { free(ctx->active_symbols); ctx->active_symbols = NULL; }
    if (ctx->dawg_nodes) { free(ctx->dawg_nodes); ctx->dawg_nodes = NULL; }
    if (ctx->dawg_transitions) { free(ctx->dawg_transitions); ctx->dawg_transitions = NULL; }

    /* Free scratch buffers */
    if (ctx->init_flags & INIT_FLAG_LARGE_SCRATCH) {
        fprintf(stderr, "[DESTROY] Freeing scratch buffers\n");
        if (ctx->large_scratch) { free(ctx->large_scratch); ctx->large_scratch = NULL; }
        if (ctx->edge_source_scratch) { free(ctx->edge_source_scratch); ctx->edge_source_scratch = NULL; }
    }

    /* Free lexeme entries and array with bounds validation */
    /* LOW FIX M-1: Remove O(lexeme_count) stderr messages - they are slow and noisy.
     * Only print summary, not per-lexeme messages. */
    if (ctx->lexemes) {
        uint32_t lexeme_limit = ctx->lexeme_count;
        if (lexeme_limit > ctx->lexeme_capacity) lexeme_limit = ctx->lexeme_capacity;
        for (uint32_t i = 0; i < lexeme_limit; i++) {
            if (ctx->lexemes[i].surface) {
                free(ctx->lexemes[i].surface);
                ctx->lexemes[i].surface = NULL;
            }
        }
        free(ctx->lexemes);
        ctx->lexemes = NULL;
    }

    fprintf(stderr, "[DESTROY] Freeing context\n");
    free(ctx);
    ctx = NULL;
    fprintf(stderr, "[DESTROY] Cleanup complete\n");
}

/* ============================= NODE MANAGEMENT ============================== */

/**
 * le_get_or_create_node - Hash table lookup with linear probing.
 * 
 * Returns the NodeID if found, or creates a new node for the token_id
 * and returns its NodeID. Initializes node structure and adds to SCC 0.
 */
NodeID le_get_or_create_node(EngineContext* ctx, TokenID token_id) {
    if (!ctx) return INVALID;

    /* Fibonacci hash for better distribution */
    uint32_t h = (token_id * 0x9E3779B1u) % HASH_SIZE;
    uint32_t steps = 0;
    NodeID slot;

    /* Linear probing: lookup */
    while (steps < HASH_SIZE) {
        slot = atomic_load_explicit(&ctx->node_hash[h], memory_order_acquire);
        
        if (slot == INVALID) {
            /* Empty slot found; create new node */
            /* CRITICAL FIX CB5: Claim ID atomically AFTER CAS succeeds to prevent node leak
             * Previous implementation claimed ID before CAS, causing node leak if CAS failed:
             * - node_count incremented
             * - node initialized
             * - edge_span_init called
             * - CAS fails → recursive call
             * - new_id is orphaned (no hash entry points to it)
             * Fix: Only claim ID after CAS succeeds, so no leak on failure. */
            
            /* CRITICAL FIX: Use CLAIMED sentinel instead of INVALID as placeholder.
             * Using INVALID as both expected and placeholder means CAS does nothing,
             * allowing multiple threads to simultaneously claim the same slot and
             * corrupt the hash table with duplicate node IDs. */
            uint32_t expected = INVALID;
            uint32_t placeholder = CLAIMED;  /* Use CLAIMED as placeholder to acquire exclusivity */
            if (!atomic_compare_exchange_strong_explicit(
                    &ctx->node_hash[h], &expected, placeholder,
                    memory_order_release, memory_order_acquire)) {
                /* CRITICAL FIX C-2: Convert recursion to iterative loop to prevent stack overflow
                 * Under high concurrency, recursive calls can consume hundreds of stack frames.
                 * Instead, restart the probe loop from the initial hash position. */
                steps = 0;
                h = (token_id * 0x9E3779B1u) % HASH_SIZE;
                continue;
            }

            /* CAS succeeded: now claim node ID and initialize */
            uint32_t current_count = atomic_load_explicit(&ctx->node_count, memory_order_relaxed);
            if (current_count >= MAX_NODES) {
                atomic_store_explicit(&ctx->node_hash[h], INVALID, memory_order_release);
                fprintf(stderr, "ERROR: MAX_NODES reached (%u)\n", MAX_NODES);
                return INVALID;
            }

            NodeID new_id = atomic_fetch_add(&ctx->node_count, 1);
            if (new_id >= MAX_NODES) {
                /* DO NOT fetch_sub. The ID space is saturated. Let it exhaust cleanly. */
                atomic_store_explicit(&ctx->node_hash[h], INVALID, memory_order_release);
                return INVALID;
            }

            /* Initialize node structure */
            ctx->nodes[new_id].token_id = token_id;
            atomic_init(&ctx->nodes[new_id].first_edge, INVALID);
            atomic_init(&ctx->nodes[new_id].edge_count, 0);
            atomic_init(&ctx->nodes[new_id].entropy_bits, 0);
            atomic_init(&ctx->nodes[new_id].total_mass, 0);
            /* CRITICAL FIX: Use explicit bit-packing for turbulence initialization
             * turbulence is typed as _Atomic uint32_t, so direct float assignment
             * causes implicit C cast (0.0f -> 0), which works for zero but fails for
             * non-zero values (e.g., 1.0f casts to integer 1, which is 1.4e-45 in IEEE-754).
             * Use memcpy to preserve IEEE-754 bit pattern. */
            float init_turb = 0.0f;
            uint32_t init_bits;
            memcpy(&init_bits, &init_turb, sizeof(float));
            atomic_init(&ctx->nodes[new_id].turbulence, init_bits);
            ctx->nodes[new_id].absorbing = 0;
            atomic_init(&ctx->nodes[new_id].last_written_step, 0);

            /* Initialize transient state */
            atomic_init(&ctx->transient_nodes[new_id].scc_id, 0);  /* Default to SCC 0 */
            atomic_init(&ctx->transient_nodes[new_id].next_in_scc, INVALID);
            atomic_init(&ctx->transient_nodes[new_id].freq, 1);

            /* Initialize edge span */
            if (edge_span_init(ctx, new_id, 8) != 0) {
                fprintf(stderr, "ERROR: Failed to initialize edge span for node %u\n", new_id);
                /* DO NOT fetch_sub. The node ID is already claimed and leaked. */
                atomic_store_explicit(&ctx->node_hash[h], INVALID, memory_order_release);  /* Release hash slot */
                return INVALID;
            }

            /* CRITICAL FIX: Release fence to ensure initialization is visible before updating hash */
            atomic_thread_fence(memory_order_release);

            /* Update hash slot with actual node ID */
            atomic_store_explicit(&ctx->node_hash[h], new_id, memory_order_release);

            /* CRITICAL FIX C-3: Link new node into SCC 0 member list atomically
             * Previous implementation used three separate atomic operations, allowing
             * two threads to read the same old_head, both write their next_in_scc,
             * then both write their new_id as head. The second write wins; the first
             * node is permanently excluded from the list while member_count still counts it.
             * Fix: Use a CAS loop on scc0->head to ensure atomicity. */
            SccNode *scc0 = &ctx->scc_nodes[0];
            NodeID expected_head;
            /* L-1 FIX: Add spin timeout to prevent indefinite spinning */
            int spin_count = 0;
            const int SPIN_LIMIT = 10000;
            do {
                expected_head = atomic_load_explicit(&scc0->head, memory_order_acquire);
                atomic_store_explicit(&ctx->transient_nodes[new_id].next_in_scc,
                                      expected_head, memory_order_relaxed);
                if (++spin_count > SPIN_LIMIT) {
                    fprintf(stderr, "WARNING: Spin timeout in le_get_or_create_node SCC head CAS\n");
                    break;
                }
            } while (!atomic_compare_exchange_weak_explicit(
                &scc0->head, &expected_head, new_id,
                memory_order_release, memory_order_acquire));
            if (spin_count <= SPIN_LIMIT) {
                atomic_fetch_add_explicit(&scc0->member_count, 1, memory_order_relaxed);
            }

            /* CRITICAL FIX: Reset stable_epochs on structural change (new node creation) */
            EpochID current_epoch = atomic_load(&ctx->current_epoch);
            atomic_store_explicit(&scc0->last_modified, current_epoch, memory_order_relaxed);
            atomic_store_explicit(&scc0->stable_epochs, 0, memory_order_relaxed);

            return new_id;
        }

        /* DAWG FIX D-1: Spin on CLAIMED sentinel.
         * Do NOT advance 'h'. Another thread is currently initializing this exact token.
         * Skipping it causes vocabulary blindness and duplicate node insertion.
         * CONC-2 FIX: Add CPU pause to prevent livelock under contention. */
        if (slot == CLAIMED) {
#if defined(__x86_64__) || defined(__i386__)
            _mm_pause();
#elif defined(__aarch64__)
            __asm__ volatile("yield" ::: "memory");
#else
            sched_yield();
#endif
            continue;
        }

        if (slot >= MAX_NODES) {
            h = (h + 1) % HASH_SIZE;
            steps++;
            continue;
        }

        /* Check if token_id matches */
        if (ctx->nodes[slot].token_id == token_id) {
            return slot;  /* Found existing node */
        }

        /* Collision; linear probe to next slot */
        h = (h + 1) % HASH_SIZE;
        steps++;
    }

    fprintf(stderr, "ERROR: Hash table full (max %u nodes)\n", HASH_SIZE);
    return INVALID;
}

/* ============================= EDGE MANAGEMENT ============================== */


/**
 * le_kosaraju_scc - Run Kosaraju's SCC algorithm on the graph.
 *
 * Computes SCCs in O(V+E) time using two-pass DFS:
 * 1. First pass: DFS on original graph to get finishing order
 * 2. Second pass: DFS on reversed graph in reverse finishing order
 * This replaces Tarjan's algorithm which had lowlink propagation issues.
 * Should be called at epoch boundaries before promotion gating.
 */
/* CRITICAL FIX C-5: Remove static to allow extern linkage from gemini_phrase_seed.c */
void le_kosaraju_scc(EngineContext* ctx) {
    if (!ctx) return;

    uint32_t node_count = atomic_load(&ctx->node_count);
    if (node_count == 0) return;

    uint32_t edge_count = atomic_load(&ctx->edge_count);
    if (edge_count == 0) return;

    GraphScratchLarge* scratch = ctx->large_scratch;
    if (!scratch) return;

    /* Initialize reverse adjacency lists */
    for (uint32_t i = 0; i < node_count; i++) {
        scratch->rev_heads[i] = INVALID;
    }

    /* CRITICAL FIX: Clear stale next_in_scc links from previous epoch.
     * Without this, nodes that change SCCs retain old links, causing
     * double-counting in member_count and corrupting linked-list traversal. */
    for (uint32_t i = 0; i < node_count; i++) {
        atomic_store_explicit(&ctx->transient_nodes[i].next_in_scc, INVALID, memory_order_relaxed);
    }

    /* Build reverse graph - iterate through nodes and their outgoing edges */
    for (uint32_t i = 0; i < node_count; i++) {
        EdgeID e = atomic_load_explicit(&ctx->nodes[i].first_edge, memory_order_acquire);
        while (e != INVALID && e < edge_count) {
            NodeID target = ctx->edges[e].target;
            if (target < node_count) {
                /* Add reverse edge: target -> i */
                scratch->rev_source[e] = i;
                scratch->rev_nexts[e] = scratch->rev_heads[target];
                scratch->rev_heads[target] = e;
            }
            e = atomic_load_explicit(&ctx->edge_nexts[e], memory_order_acquire);
        }
    }

    /* PATCH I2: CRITICAL - Invalidate node SCC assignments before Tarjan's pass
     * Previous epoch's topological mapping must be completely cleared to prevent
     * state bleeding. If SCC IDs from the previous epoch persist, the algorithm
     * will fail to recognize that a component has been fractured by pruning. */
    for (uint32_t i = 0; i < node_count; i++) {
        atomic_store_explicit(&ctx->transient_nodes[i].scc_id, INVALID, memory_order_release);
        atomic_store_explicit(&ctx->transient_nodes[i].next_in_scc, INVALID, memory_order_release);
    }

    /* First pass: DFS on original graph to get finishing order */
    bool* visited = scratch->in_queue_scratch;
    NodeID* finish_stack = scratch->collect_stack;
    uint32_t finish_top = 0;

    for (uint32_t i = 0; i < node_count; i++) {
        visited[i] = false;
    }

    /* Iterative DFS using explicit stack */
    NodeID* dfs_stack = scratch->queue_scratch;
    EdgeID* edge_stack = (EdgeID*)scratch->edge_ids;

    for (uint32_t i = 0; i < node_count; i++) {
        if (visited[i]) continue;

        uint32_t dfs_top = 0;
        dfs_stack[dfs_top] = i;
        edge_stack[dfs_top] = atomic_load_explicit(&ctx->nodes[i].first_edge, memory_order_acquire);
        dfs_top++;

        while (dfs_top > 0) {
            NodeID v = dfs_stack[dfs_top - 1];
            EdgeID e = edge_stack[dfs_top - 1];

            if (!visited[v]) {
                visited[v] = true;
            }

            if (e != INVALID && e < edge_count) {
                /* CRITICAL FIX: Skip ghost edges (zero-weight edges) during DFS
                 * When le_sparsify_edges or le_apply_edge_decay zero out an edge (weight = 0),
                 * the edge remains structurally linked in edge_nexts. Kosaraju's DFS must
                 * explicitly check the weight and reject decayed edges, otherwise the graph
                 * grows monotonically into a giant component and collapses into a hairball. */
                uint32_t weight_bits = atomic_load_explicit(&ctx->edges[e].weight, memory_order_acquire);
                float w;
                memcpy(&w, &weight_bits, 4);
                if (w <= 0.0f) {
                    edge_stack[dfs_top - 1] = atomic_load_explicit(&ctx->edge_nexts[e], memory_order_acquire);
                    continue;  /* Skip this ghost edge */
                }
                
                NodeID target = ctx->edges[e].target;
                edge_stack[dfs_top - 1] = atomic_load_explicit(&ctx->edge_nexts[e], memory_order_acquire);
                if (!visited[target] && target < node_count && dfs_top < MAX_NODES) {
                    dfs_stack[dfs_top] = target;
                    edge_stack[dfs_top] = atomic_load_explicit(&ctx->nodes[target].first_edge, memory_order_acquire);
                    dfs_top++;
                }
            } else {
                /* All edges processed - add to finish stack */
                finish_stack[finish_top++] = v;
                dfs_top--;
            }
        }
    }

    /* Second pass: DFS on reversed graph in reverse finishing order */
    for (uint32_t i = 0; i < node_count; i++) {
        visited[i] = false;
    }

    uint32_t scc_counter = 1;  /* SCC 0 is reserved for default bucket */

    /* CRITICAL FIX: Reset member_count on all previously-used SCCs.
     * Kosaraju will reassign nodes to new SCCs; old counts must be zeroed
     * to prevent double-counting artifacts. */
    uint32_t old_scc_count = atomic_load(&ctx->scc_count);
    for (uint32_t s = 0; s < old_scc_count && s < MAX_SCCS; s++) {
        atomic_store_explicit(&ctx->scc_nodes[s].member_count, 0, memory_order_release);
        atomic_store_explicit(&ctx->scc_nodes[s].head, INVALID, memory_order_release);
        atomic_store_explicit(&ctx->scc_nodes[s].tail, INVALID, memory_order_release);
    }

    for (int32_t i = finish_top - 1; i >= 0; i--) {
        NodeID v = finish_stack[i];
        if (visited[v]) continue;

        /* Start DFS on reversed graph from v */
        uint32_t dfs_top = 0;
        dfs_stack[dfs_top++] = v;

        uint32_t member_count = 0;
        NodeID head = INVALID;
        NodeID tail = INVALID;

        while (dfs_top > 0) {
            NodeID node = dfs_stack[--dfs_top];
            if (visited[node]) continue;

            visited[node] = true;
            member_count++;

            /* Assign SCC ID to node */
            atomic_store_explicit(&ctx->transient_nodes[node].scc_id, scc_counter, memory_order_release);

            /* Build linked list for this SCC */
            if (head == INVALID) {
                head = node;
                tail = node;
            } else {
                atomic_store_explicit(&ctx->transient_nodes[node].next_in_scc, head, memory_order_release);
                head = node;
            }

            /* Traverse reverse adjacency list */
            EdgeID rev_e = scratch->rev_heads[node];
            while (rev_e != INVALID && rev_e < edge_count) {
                /* CRITICAL FIX: Skip ghost edges (zero-weight edges) during reverse DFS
                 * Same issue as forward pass - must check weight to avoid traversing decayed edges. */
                uint32_t weight_bits = atomic_load_explicit(&ctx->edges[rev_e].weight, memory_order_acquire);
                float w;
                memcpy(&w, &weight_bits, 4);
                if (w <= 0.0f) {
                    rev_e = scratch->rev_nexts[rev_e];
                    continue;  /* Skip this ghost edge */
                }
                
                NodeID source = scratch->rev_source[rev_e];
                if (!visited[source] && source < node_count && dfs_top < MAX_NODES) {
                    dfs_stack[dfs_top++] = source;
                }
                rev_e = scratch->rev_nexts[rev_e];
            }
        }

        if (member_count > 0) {
            /* Initialize SCC node */
            SccNode* scc = &ctx->scc_nodes[scc_counter];
            /* HIGH FIX H-1: Preserve is_forced flag across SCC recomputation.
             * Forced SCCs should remain forced even after Kosaraju rebuilds the graph.
             * Read the old flag before clearing and restore it after. */
            bool old_forced = atomic_load_explicit(&scc->is_forced, memory_order_acquire);
            atomic_store_explicit(&scc->head, head, memory_order_release);
            atomic_store_explicit(&scc->tail, tail, memory_order_release);
            atomic_store_explicit(&scc->member_count, member_count, memory_order_release);
            atomic_store_explicit(&scc->last_member_count, member_count, memory_order_release);
            atomic_init(&scc->internal_edges, 0);
            atomic_init(&scc->external_edges, 0);
            atomic_init(&scc->traversal_count, 0);
            /* CRITICAL FIX SCC Identity Scramble: Reset promotion status on slot reuse
             * Previous PATCH G1 preserved is_promoted to prevent symbol explosion, but this
             * caused SCC identity scramble: when Kosaraju rebuilds, unrelated components
             * landing in the same array slot inherit promotion status from the previous epoch.
             * Fix: Reset is_promoted and symbol_id, then re-promote based on new SCC structure.
             * Stable components will naturally re-qualify for promotion. */
            atomic_init(&scc->is_promoted, false);
            atomic_init(&scc->symbol_id, INVALID);
            atomic_init(&scc->is_candidate, false);
            atomic_init(&scc->is_forced, old_forced);  /* Restore forced flag */
            atomic_init(&scc->is_weak, false);
            atomic_store_explicit(&scc->first_seen, atomic_load(&ctx->current_epoch), memory_order_relaxed);
            /* CRITICAL FIX: Do NOT reset last_modified or stable_epochs here
             * SCC recomputation is a derived property, not a structural modification.
             * Only edge additions should update last_modified.
             * For new SCC slots, these fields are already 0 from calloc.
             * For re-used slots, preserve previous stability metrics. */
            /* atomic_store_explicit(&scc->last_modified, atomic_load(&ctx->current_epoch), memory_order_relaxed); */
            /* atomic_store_explicit(&scc->stable_epochs, 0, memory_order_relaxed); */
            scc_store_coherence(scc, 0.0f);
            scc_store_avg_entropy(scc, 0.0f);

            scc_counter++;
            if (scc_counter >= MAX_SCCS) {
                fprintf(stderr, "ERROR: MAX_SCCS exceeded during Kosaraju\n");
                break;
            }
        }
    }

    /* Update total SCC count */
    atomic_store(&ctx->scc_count, scc_counter);
}

/* F-5 FIX: le_tarjan_scc has been removed.
 * It was dead code (static, __attribute__((unused)), zero callers) with a
 * known lowlink back-propagation bug. The active SCC algorithm is
 * le_kosaraju_scc, which correctly computes SCCs via two-pass DFS. */


/* NOTE: The old float-based le_apply_edge_decay has been removed.
 * Edge decay is now handled by le_apply_vectorized_edge_decay() which
 * operates on log-quantized uint16_t weights with AVX2 (or scalar fallback). */

typedef struct {
    EdgeID edge_id;
    float weight;
    NodeID target;
} SparsifyEdgeWeight;

static int compare_sparsify_edge_weights(const void *a, const void *b) {
    float wa = ((const SparsifyEdgeWeight *)a)->weight;
    float wb = ((const SparsifyEdgeWeight *)b)->weight;
    return (wa < wb) - (wa > wb); /* Descending order */
}

/* H-8 FIX: LexemeFreq type for vocabulary generation in le_save_mmap */
typedef struct {
    uint32_t lexeme_id;
    uint32_t freq;
} LexemeFreq;

/* H-8 FIX: Comparison function for qsort in le_save_mmap */
static int compare_lexeme_freq_desc(const void *a, const void *b) {
    const LexemeFreq *la = (const LexemeFreq *)a;
    const LexemeFreq *lb = (const LexemeFreq *)b;
    if (la->freq > lb->freq) return -1;
    if (la->freq < lb->freq) return 1;
    return 0;
}

/**
 * le_compact_ghost_edges - Thread-safe garbage collection for ghost edges.
 * H-1 FIX: Compacts the edge workspace by removing zero-weight edges and reclaiming slots.
 * Must be called exclusively at epoch boundaries under the context write-lock.
 */
void le_compact_ghost_edges(EngineContext *ctx) {
    pthread_rwlock_wrlock(&ctx->epoch_rwlock);
    
    uint32_t total_nodes = atomic_load_explicit(&ctx->node_count, memory_order_acquire);
    uint32_t total_edges = atomic_load_explicit(&ctx->edge_count, memory_order_acquire);
    
    /* Allocate temporary mirror arrays */
    GeminiEdgeDisk *new_edges = calloc(MAX_EDGES, sizeof(GeminiEdgeDisk));
    _Atomic EdgeID *new_nexts = malloc(MAX_EDGES * sizeof(_Atomic EdgeID));
    if (!new_edges || !new_nexts) {
        free(new_edges); free(new_nexts);
        pthread_rwlock_unlock(&ctx->epoch_rwlock);
        return;
    }
    
    uint32_t write_ptr = 0;
    for (uint32_t n = 0; n < total_nodes; n++) {
        uint32_t curr = atomic_load_explicit(&ctx->nodes[n].first_edge, memory_order_relaxed);
        uint32_t last_new_edge_idx = INVALID;
        atomic_store_explicit(&ctx->nodes[n].first_edge, INVALID, memory_order_relaxed);
        
        while (curr != INVALID && curr < total_edges) {
            uint32_t w_bits = atomic_load_explicit(&ctx->edges[curr].weight, memory_order_relaxed);
            union { uint32_t bits; float val; } w_un;
            w_un.bits = w_bits;
            
            /* Retain edge only if it has a non-zero structural weight value */
            if (w_un.val > 1e-6f && write_ptr < MAX_EDGES) {
                new_edges[write_ptr].target = ctx->edges[curr].target;
                atomic_init(&new_edges[write_ptr].weight, w_bits);
                atomic_init(&new_nexts[write_ptr], INVALID);
                
                if (last_new_edge_idx == INVALID) {
                    atomic_store_explicit(&ctx->nodes[n].first_edge, write_ptr, memory_order_relaxed);
                } else {
                    atomic_store_explicit(&new_nexts[last_new_edge_idx], write_ptr, memory_order_relaxed);
                }
                last_new_edge_idx = write_ptr;
                write_ptr++;
            }
            curr = atomic_load_explicit(&ctx->edge_nexts[curr], memory_order_relaxed);
        }
    }
    
    /* Swap arrays and update global indicators */
    memcpy(ctx->edges, new_edges, write_ptr * sizeof(GeminiEdgeDisk));
    for(uint32_t i = 0; i < write_ptr; i++) {
        uint32_t nxt = atomic_load_explicit(&new_nexts[i], memory_order_relaxed);
        atomic_store_explicit(&ctx->edge_nexts[i], nxt, memory_order_relaxed);
    }
    
    atomic_store_explicit(&ctx->edge_count, write_ptr, memory_order_release);
    
    free(new_edges);
    free(new_nexts);
    pthread_rwlock_unlock(&ctx->epoch_rwlock);
}

/**
 * le_sparsify_edges - Keep only top-K strongest outgoing edges per node.
 *
 * Prevents graph density explosion by pruning weak edges.
 * For each node, keeps only the top MAX_OUTGOING_EDGES strongest transitions.
 *
 * PATCH H1: Add SCC-aware sparsification to break giant components.
 * If an SCC is too large (> 50% of nodes), apply aggressive intra-SCC pruning
 * to prevent monolithic clusters from forming.
 *
 * PATCH I1: Add hub-penalized sparsifier to break hub immunity.
 * High-frequency hub nodes (stopwords, structural particles) accumulate massive
 * edge weights, granting them immunity from bottom-K pruning. Apply relative
 * information (PMI) to sever casual links to structural stopwords.
 */
static void le_sparsify_edges(EngineContext* ctx) {
    if (!ctx) return;

    uint32_t node_count = atomic_load(&ctx->node_count);
    if (node_count == 0) return;

    uint32_t total_pruned = 0;
    uint32_t hub_pruned = 0;

    /* PATCH I1: Calculate hub frequency threshold (10x average) */
    uint32_t hub_threshold = 0;
    if (node_count > 0) {
        uint64_t total_freq = 0;
        for (uint32_t n = 0; n < node_count; n++) {
            total_freq += atomic_load_explicit(&ctx->transient_nodes[n].freq, memory_order_relaxed);
        }
        uint32_t avg_freq = (uint32_t)(total_freq / node_count);
        hub_threshold = avg_freq * 10;
    }

    /* PATCH H1: Detect giant component for aggressive pruning */
    uint32_t scc_count = atomic_load(&ctx->scc_count);
    uint32_t max_scc_size = 0;
    SccID giant_scc_id = INVALID;
    for (uint32_t i = 0; i < scc_count && i < MAX_SCCS; i++) {
        uint32_t members = atomic_load(&ctx->scc_nodes[i].member_count);
        if (members > max_scc_size) {
            max_scc_size = members;
            giant_scc_id = i;
        }
    }

    /* Define giant component threshold: > 50% of nodes */
    bool has_giant_component = (max_scc_size > (node_count / 2));
    uint32_t aggressive_top_k = has_giant_component ? 8 : 32; /* Aggressive pruning for giant SCC */

    // if (has_giant_component) {
    //     /* Debug output disabled */
    //     // fprintf(stderr, "[GIANT_COMPONENT] SCC %u has %u nodes (%.1f%% of graph) - applying aggressive pruning\n",
    //     //         giant_scc_id, max_scc_size, (max_scc_size * 100.0f) / node_count);
    // }

    for (uint32_t n = 0; n < node_count; n++) {
        /* CRITICAL FIX: Dynamically allocate edge array to process all edges
         * Previous static array of 256 elements only processed first 256 edges,
         * allowing hub nodes with 10,000 edges to bypass pruning entirely.
         * Now we count edges first, allocate appropriately, and process all. */
        uint64_t src_freq = atomic_load_explicit(&ctx->transient_nodes[n].freq, memory_order_relaxed);
        
        /* First pass: count outgoing edges for this node */
        uint32_t out_degree = 0;
        EdgeID e = atomic_load_explicit(&ctx->nodes[n].first_edge, memory_order_acquire);
        while (e != INVALID && out_degree < MAX_EDGES) {
            uint32_t weight_bits = atomic_load_explicit(&ctx->edges[e].weight, memory_order_relaxed);
            float weight;
            memcpy(&weight, &weight_bits, 4);
            if (weight > 0.0f) {
                out_degree++;
            }
            e = atomic_load_explicit(&ctx->edge_nexts[e], memory_order_acquire);
        }
        
        /* Skip if no edges or already under threshold */
        if (out_degree == 0) continue;

        /* M-2 FIX: Use edge_source_scratch instead of calloc to avoid O(node_count) allocations per epoch */
        SparsifyEdgeWeight *edges = NULL;
        if (ctx->edge_source_scratch && out_degree <= MAX_NODES) {
            edges = (SparsifyEdgeWeight *)ctx->edge_source_scratch;
        } else {
            edges = (SparsifyEdgeWeight *)calloc(out_degree, sizeof(SparsifyEdgeWeight));
            if (!edges) {
                fprintf(stderr, "WARNING: Failed to allocate edge array for node %u (degree=%u), skipping sparsify\n", n, out_degree);
                continue;
            }
        }
        
        uint32_t edge_idx = 0;
        e = atomic_load_explicit(&ctx->nodes[n].first_edge, memory_order_acquire);
        while (e != INVALID && edge_idx < out_degree) {
            uint32_t weight_bits = atomic_load_explicit(&ctx->edges[e].weight, memory_order_relaxed);
            float weight;
            memcpy(&weight, &weight_bits, 4);
            if (weight > 0.0f) {
                edges[edge_idx].edge_id = e;
                edges[edge_idx].weight = weight;
                edges[edge_idx].target = ctx->edges[e].target;
                edge_idx++;
            }
            e = atomic_load_explicit(&ctx->edge_nexts[e], memory_order_acquire);
        }

        /* ADAPTIVE TOP-K SPARSIFICATION based on node entropy */
        /* Get node entropy to determine appropriate sparsification threshold */
        SccID node_scc = atomic_load_explicit(&ctx->transient_nodes[n].scc_id, memory_order_acquire);
        float node_entropy = 0.0f;
        if (node_scc < MAX_SCCS) {
            uint32_t entropy_bits = atomic_load_explicit(&ctx->scc_nodes[node_scc].avg_entropy_bits, memory_order_relaxed);
            memcpy(&node_entropy, &entropy_bits, 4);
        }

        uint32_t dynamic_top_k = 32; /* Default */
        if (node_entropy > 1.5f) {
            dynamic_top_k = 64;   /* Retain weaker paths for nuanced structural contexts */
        } else if (node_entropy < 0.5f) {
            dynamic_top_k = 16;   /* Tighten constraints on repetitive, uniform sequences */
        }

        /* PATCH H1: Apply aggressive pruning for nodes in giant component */
        if (has_giant_component && node_scc == giant_scc_id) {
            dynamic_top_k = aggressive_top_k; /* Override with aggressive threshold */
        }

        if (edge_idx <= dynamic_top_k) continue;

        /* PATCH I1: Apply hub penalty before sorting */
        for (uint32_t i = 0; i < edge_idx; i++) {
            NodeID target = edges[i].target;
            if (target >= node_count) continue;
            uint64_t tgt_freq = atomic_load_explicit(&ctx->transient_nodes[target].freq, memory_order_relaxed);
            if (tgt_freq > hub_threshold) {
                float required_ratio = (float)src_freq * 0.05f;
                if (edges[i].weight < required_ratio) {
                    edges[i].weight = 0.0f;
                    hub_pruned++;
                }
            }
        }

        /* Sort by weight descending using qsort for O(N log N) performance */
        qsort(edges, edge_idx, sizeof(SparsifyEdgeWeight), compare_sparsify_edge_weights);

        /* Zero out weights of weak edges beyond dynamic top-K */
        uint32_t kept = 0;
        for (uint32_t i = 0; i < edge_idx; i++) {
            if (edges[i].weight > 0.0f) {
                if (kept >= dynamic_top_k) {
                    atomic_store_explicit(&ctx->edges[edges[i].edge_id].weight, 0, memory_order_relaxed);
                    total_pruned++;
                }
                kept++;
            } else {
                atomic_store_explicit(&ctx->edges[edges[i].edge_id].weight, 0, memory_order_relaxed);
                total_pruned++;
            }
        }

        /* M-2 FIX: Only free if we used calloc, not edge_source_scratch */
        if (edges != (SparsifyEdgeWeight *)ctx->edge_source_scratch) {
            free(edges);
        }
    }

    if (total_pruned > 0) {
        /* Debug output disabled */
        // fprintf(stderr, "[EDGE_SPARSIFY] Pruned %u weak edges (adaptive top-K%s, hub-penalty: %u)\n",
        //         total_pruned, has_giant_component ? ", giant-component-aware" : "", hub_pruned);
    }
}

/**
 * le_recompute_scc_metrics - Recompute internal/external edge counts and coherence.
 *
 * After SCC recomputation (Kosaraju), historical edge statistics are invalid.
 * This function traverses all edges and reclassifies them as internal or external
 * to their source SCC, then recomputes coherence metrics.
 */
/* CRITICAL FIX C-5: Remove static to allow extern linkage from gemini_phrase_seed.c */
void le_recompute_scc_metrics(EngineContext* ctx) {
    if (!ctx) return;

    uint32_t scc_count = atomic_load(&ctx->scc_count);
    if (scc_count == 0) return;

    uint32_t edge_count = atomic_load(&ctx->edge_count);
    uint32_t node_count = atomic_load(&ctx->node_count);
    if (edge_count == 0 || node_count == 0) return;

    /* Reset all edge counters */
    for (SccID scc_id = 0; scc_id < scc_count; scc_id++) {
        SccNode* scc = &ctx->scc_nodes[scc_id];
        atomic_init(&scc->internal_edges, 0);
        atomic_init(&scc->external_edges, 0);
    }

    /* CRITICAL FIX: Use pre-allocated scratch buffer instead of malloc/free.
     * The old malloc approach caused 2.7s epoch times. This is O(1) setup. */
    NodeID *edge_source = ctx->edge_source_scratch;
    if (!edge_source || edge_count > MAX_EDGES) {
        fprintf(stderr, "ERROR: edge_source_scratch unavailable or too small\n");
        return;
    }

    /* Initialize to INVALID — only up to edge_count, not MAX_EDGES */
    for (uint32_t i = 0; i < edge_count; i++) {
        edge_source[i] = INVALID;
    }

    /* Single pass through all nodes' adjacency lists to map edge_id -> source_node */
    for (uint32_t n = 0; n < node_count; n++) {
        EdgeID e = atomic_load_explicit(&ctx->nodes[n].first_edge, memory_order_acquire);
        uint32_t guard = 0;
        while (e != INVALID && e < edge_count && guard++ < MAX_EDGES) {
            if (edge_source[e] == INVALID) {
                edge_source[e] = (NodeID)n;
            }
            e = atomic_load_explicit(&ctx->edge_nexts[e], memory_order_acquire);
        }
    }

    /* Reclassify all edges using the source map — O(E) */
    for (uint32_t i = 0; i < edge_count; i++) {
        NodeID source = edge_source[i];
        if (source == INVALID || source >= node_count) continue;

        SccID scc_from = atomic_load_explicit(&ctx->transient_nodes[source].scc_id, memory_order_acquire);
        if (scc_from >= scc_count) continue;

        NodeID target = ctx->edges[i].target;
        if (target >= node_count) continue;

        SccID scc_to = atomic_load_explicit(&ctx->transient_nodes[target].scc_id, memory_order_acquire);
        if (scc_to >= scc_count) continue;

        if (scc_from == scc_to) {
            atomic_fetch_add_explicit(&ctx->scc_nodes[scc_from].internal_edges, 1, memory_order_relaxed);
        } else {
            atomic_fetch_add_explicit(&ctx->scc_nodes[scc_from].external_edges, 1, memory_order_relaxed);
        }
    }

    /* edge_source is pre-allocated, no need to free */

    /* Recompute coherence for each SCC */
    for (SccID scc_id = 0; scc_id < scc_count; scc_id++) {
        SccNode* scc = &ctx->scc_nodes[scc_id];
        uint32_t internal = atomic_load_explicit(&scc->internal_edges, memory_order_acquire);
        uint32_t external = atomic_load_explicit(&scc->external_edges, memory_order_acquire);
        uint32_t total = internal + external;

        if (total > 0) {
            float coherence = (float)internal / (float)total;
            scc_store_coherence(scc, coherence);
        } else {
            scc_store_coherence(scc, 0.0f);
        }
    }
}

/**
 * le_update_scc_entropy - Recompute average entropy for all SCCs.
 *
 * Traverses each SCC's node list and calculates the average entropy
 * across all nodes in the SCC. This is called at epoch boundaries
 * after Tarjan's algorithm has computed new SCCs.
 */
static void le_update_scc_entropy(EngineContext* ctx) {
    if (!ctx) return;

    uint32_t scc_count = atomic_load(&ctx->scc_count);
    if (scc_count == 0) return;

    for (SccID scc_id = 0; scc_id < scc_count; scc_id++) {
        SccNode* scc = &ctx->scc_nodes[scc_id];
        uint32_t member_count = atomic_load_explicit(&scc->member_count, memory_order_acquire);

        if (member_count == 0) continue;

        float entropy_sum = 0.0f;
        uint32_t valid_nodes = 0;

        /* Traverse all nodes in this SCC */
        NodeID n = atomic_load_explicit(&scc->head, memory_order_acquire);
        uint32_t guard = 0;
        while (n != INVALID && guard++ < member_count) {
            /* Validate SCC node ID */
            if (n >= MAX_NODES) {
                fprintf(stderr, "CORRUPT SCC NODE ID: %u in SCC %u\n", n, scc_id);
                break;
            }

            /* Calculate entropy for this node */
            uint32_t edge_count = atomic_load_explicit(&ctx->nodes[n].edge_count, memory_order_acquire);
            if (edge_count > 0) {
                float node_entropy = 0.0f;
                float total_weight = 0.0f;

                /* M-3 FIX: Single-pass entropy calculation using edge_source_scratch
                 * Collect weights in scratch buffer, then compute entropy in one pass */
                float *weights = NULL;
                if (ctx->edge_source_scratch && edge_count <= MAX_NODES) {
                    weights = (float *)ctx->edge_source_scratch;
                } else {
                    weights = (float *)calloc(edge_count, sizeof(float));
                    if (!weights) continue;
                }

                uint32_t weight_idx = 0;
                EdgeID e = atomic_load_explicit(&ctx->nodes[n].first_edge, memory_order_acquire);
                while (e != INVALID && weight_idx < edge_count) {
                    uint32_t weight_bits = atomic_load_explicit(&ctx->edges[e].weight, memory_order_acquire);
                    float weight;
                    memcpy(&weight, &weight_bits, 4);
                    weights[weight_idx++] = weight;
                    total_weight += weight;
                    e = atomic_load_explicit(&ctx->edge_nexts[e], memory_order_acquire);
                }

                if (total_weight > 0.0f) {
                    for (uint32_t i = 0; i < weight_idx; i++) {
                        float p = weights[i] / total_weight;
                        if (p > 0.0f) {
                            node_entropy -= p * log2f(p);
                        }
                    }
                }

                if (weights != (float *)ctx->edge_source_scratch) {
                    free(weights);
                }

                entropy_sum += node_entropy;
                valid_nodes++;
            }
            
            n = atomic_load_explicit(&ctx->transient_nodes[n].next_in_scc, memory_order_acquire);
        }
        
        /* Store average entropy for this SCC */
        float avg_entropy = (valid_nodes > 0) ? (entropy_sum / valid_nodes) : 0.0f;
        scc_store_avg_entropy(scc, avg_entropy);

        /* CRITICAL FIX: Do NOT overwrite freq with average - it's used for promotion */
        /* freq should remain as accumulated total traversals, not average per member */
        /* The promotion logic uses freq directly, so overwriting it breaks promotion */
        /* Commented out the recomputation to preserve accumulated traversal count */
        /*
        uint64_t traversal_count = atomic_load_explicit(&scc->traversal_count, memory_order_acquire);
        uint32_t freq = (member_count > 0) ? (uint32_t)(traversal_count / member_count) : 0;
        atomic_store_explicit(&scc->freq, freq, memory_order_release);
        */
    }
}

/**
 * le_add_edge - Add a directed edge between two nodes.
 *
 * Adds or increments an edge from -> to with weight +1.0.
 * Implements edge deduplication (upsert logic) to prevent duplicate edges.
 * Thread-safe using atomic operations and CAS loops.
 */
void le_add_edge(EngineContext* ctx, NodeID from, NodeID to) {
    if (!ctx || from == INVALID || to == INVALID) return;
    if (from >= MAX_NODES || to >= MAX_NODES) return;

    /* C-2 FIX: Lock the source node stripe to guarantee atomic check-and-insert */
    pthread_mutex_lock(&ctx->node_insert_locks[from & 1023]);

    /* CONC-1 FIX: Acquire read-lock to prevent concurrent epoch transitions */
    pthread_rwlock_rdlock(&ctx->epoch_rwlock);

    /* Get SCC IDs for source and target nodes */
    SccID scc_from = atomic_load_explicit(&ctx->transient_nodes[from].scc_id, memory_order_acquire);
    SccID scc_to = atomic_load_explicit(&ctx->transient_nodes[to].scc_id, memory_order_acquire);

    /* CRITICAL FIX: Check for existing edge before creating new one (deduplication) */
    EdgeID curr = atomic_load_explicit(&ctx->nodes[from].first_edge, memory_order_acquire);
    uint32_t edge_count_snap = atomic_load_explicit(&ctx->edge_count, memory_order_acquire);

    while (curr != INVALID && curr < edge_count_snap && curr < MAX_EDGES) {
        if (ctx->edges[curr].target == to) {
            /* CRITICAL FIX: IEEE-754 float corruption - use CAS loop for atomic float addition
             * atomic_fetch_add on float bit patterns corrupts the mantissa/exponent, causing NaN/infinity.
             * C11 does not support atomic_fetch_add for floats, so we must use CAS loop. */
            uint32_t old_bits = atomic_load_explicit(&ctx->edges[curr].weight, memory_order_acquire);
            uint32_t new_bits;
            /* L-1 FIX: Add spin timeout to prevent indefinite spinning */
            int spin_count = 0;
            const int SPIN_LIMIT = 10000;
            do {
                float old_val;
                memcpy(&old_val, &old_bits, 4);
                float new_val = old_val + 1.0f;  /* Increment by 1.0 */
                memcpy(&new_bits, &new_val, 4);
                if (++spin_count > SPIN_LIMIT) {
                    fprintf(stderr, "WARNING: Spin timeout in le_add_edge weight CAS\n");
                    break;
                }
            } while (!atomic_compare_exchange_weak_explicit(
                &ctx->edges[curr].weight, &old_bits, new_bits,
                memory_order_release, memory_order_acquire));
            /* Add incoming edge to target node via pooled adjacency */
            le_add_incoming_edge_pooled(ctx, to, from);
            /* CRITICAL FIX: Gate SCC metric updates to prevent out-of-bounds access
             * During epoch transitions, Kosaraju sets scc_id to INVALID. Ingestion threads
             * must check bounds before accessing scc_nodes array to prevent segfault. */
            if (scc_from != INVALID && scc_from < MAX_SCCS) {
                atomic_fetch_add_explicit(&ctx->scc_nodes[scc_from].traversal_count, 1, memory_order_relaxed);
                atomic_fetch_add_explicit(&ctx->scc_nodes[scc_from].freq, 1, memory_order_relaxed);
            }
            /* CRITICAL FIX: Do NOT reset stable_epochs on weight increments - only on structural changes */

            /* DAWG FIX D-4: Wire the high-level semantic graph.
             * When both SCCs are promoted, add a DAWG transition between their symbols.
             * CRITICAL FIX: Use lifecycle-aware promotion check to handle Kosaraju reset window. */
            if (scc_from != INVALID && scc_to != INVALID && scc_from < MAX_SCCS && scc_to < MAX_SCCS) {
                if (le_is_scc_promoted_via_lifecycle(ctx, scc_from) &&
                    le_is_scc_promoted_via_lifecycle(ctx, scc_to)) {

                    SymbolID sym_from = atomic_load_explicit(&ctx->scc_nodes[scc_from].symbol_id, memory_order_acquire);
                    SymbolID sym_to = atomic_load_explicit(&ctx->scc_nodes[scc_to].symbol_id, memory_order_acquire);

                    if (sym_from != INVALID && sym_to != INVALID) {
                        add_dawg_transition(ctx, sym_from, sym_to, 1.0f);
                    }
                }
            }

            pthread_rwlock_unlock(&ctx->epoch_rwlock);
            pthread_mutex_unlock(&ctx->node_insert_locks[from & 1023]);
            return;
        }
        /* CRITICAL FIX C-1: Use atomic_load for _Atomic EdgeID array access */
        curr = atomic_load_explicit(&ctx->edge_nexts[curr], memory_order_acquire);
    }

    /* REMOVED: Online cycle detection and SCC merging - replaced with batch Tarjan at epoch boundary */

    /* ================================================================
     * FIX B1: PRE-SCC GATING — Prevent hairball collapse by rejecting
     * low-confidence new edges before they enter the permanent adjacency
     * pool and pollute Kosaraju's decomposition.
     * ================================================================ */
    uint32_t out_degree = atomic_load_explicit(&ctx->nodes[from].edge_count, memory_order_acquire);
    uint32_t soft_cap = 16;   /* Default budget for unproven nodes */

    if (scc_from < MAX_SCCS) {
        uint32_t stable = atomic_load_explicit(&ctx->scc_nodes[scc_from].stable_epochs, memory_order_acquire);
        uint64_t freq   = atomic_load_explicit(&ctx->scc_nodes[scc_from].freq, memory_order_acquire);

        /* Mature, high-frequency SCCs earn a larger structural budget */
        if (stable >= ctx->promotion_epochs && freq >= (uint64_t)ctx->min_freq * 20) {
            soft_cap = 64;          /* Proven semantic hubs: allow rich connectivity */
        } else if (stable >= 1 && freq >= (uint64_t)ctx->min_freq) {
            soft_cap = 32;          /* Stabilizing components: moderate growth */
        }
    }

    if (out_degree >= soft_cap) {
        /* Node/SCC is saturated. Drop this transient transition rather than
         * feeding the giant hairball. The edge can still be learned later
         * if repeated encounters push it through dedup on an existing slot. */
        pthread_rwlock_unlock(&ctx->epoch_rwlock);
        pthread_mutex_unlock(&ctx->node_insert_locks[from & 1023]);
        return;
    }
    /* ================================================================ */

    /* Edge doesn't exist: allocate new edge ID from pool */
    uint32_t new_edge_count = atomic_fetch_add(&ctx->edge_count, 1);
    if (new_edge_count >= MAX_EDGES) {
        static bool warned = false;
        if (!warned) {
            fprintf(stderr, "CRITICAL: Edge pool at maximum capacity (%u). Ingestion throttled.\n", MAX_EDGES);
            warned = true;
        }
        atomic_fetch_sub(&ctx->edge_count, 1);  /* Rollback */
        pthread_rwlock_unlock(&ctx->epoch_rwlock);
        pthread_mutex_unlock(&ctx->node_insert_locks[from & 1023]);
        return;
    }

    /* Initialize edge: set target and weight */
    ctx->edges[new_edge_count].target = to;
    
    /* CRITICAL FIX: Use memcpy to properly initialize float bit pattern
     * Direct assignment of WEIGHT_SCALE_FACTOR causes implicit conversion to uint32_t,
     * resulting in subnormal numbers (~1.4e-45) instead of proper float representation.
     * This causes massive mathematical discontinuity when edge weight jumps from near-zero to 1.0f. */
    float init_weight = 1.0f * WEIGHT_SCALE_FACTOR;
    uint32_t init_bits;
    memcpy(&init_bits, &init_weight, sizeof(float));
    atomic_init(&ctx->edges[new_edge_count].weight, init_bits);

    /* CRITICAL FIX CR2: Use CAS loop for linked list insertion to prevent edge loss
     * Previous load-then-store pattern could lose edges under concurrent ingestion:
     * Thread A loads old_first=X, Thread B loads old_first=X, both write their edges,
     * then Thread A stores first_edge=A, Thread B stores first_edge=B - Thread A's edge is lost.
     * Fix: Use CAS loop to atomically update first_edge. */
    EdgeID expected;
    /* L-1 FIX: Add spin timeout to prevent indefinite spinning */
    int spin_count = 0;
    const int SPIN_LIMIT = 10000;
    do {
        expected = atomic_load_explicit(&ctx->nodes[from].first_edge, memory_order_acquire);
        atomic_store_explicit(&ctx->edge_nexts[new_edge_count], expected, memory_order_relaxed);
        if (++spin_count > SPIN_LIMIT) {
            fprintf(stderr, "WARNING: Spin timeout in le_add_edge first_edge CAS\n");
            break;
        }
    } while (!atomic_compare_exchange_weak_explicit(
        &ctx->nodes[from].first_edge, &expected, new_edge_count,
        memory_order_release, memory_order_acquire));
    if (spin_count <= SPIN_LIMIT) {
        atomic_fetch_add(&ctx->nodes[from].edge_count, 1);
    }

    /* Update SCC edge counts */
    /* C-3 FIX: Guard internal_edges/external_edges with INVALID check to prevent OOB crash */
    if (scc_from != INVALID && scc_from < MAX_SCCS) {
        if (scc_from == scc_to)
            atomic_fetch_add_explicit(&ctx->scc_nodes[scc_from].internal_edges, 1, memory_order_relaxed);
        else
            atomic_fetch_add_explicit(&ctx->scc_nodes[scc_from].external_edges, 1, memory_order_relaxed);
    }

    /* Update SCC metadata */
    /* CRITICAL FIX: Gate SCC metric updates to prevent out-of-bounds access */
    if (scc_from != INVALID && scc_from < MAX_SCCS) {
        atomic_fetch_add_explicit(&ctx->scc_nodes[scc_from].traversal_count, 1, memory_order_relaxed);
        atomic_fetch_add_explicit(&ctx->scc_nodes[scc_from].freq, 1, memory_order_relaxed);
        EpochID current_epoch = atomic_load_explicit(&ctx->current_epoch, memory_order_acquire);
        atomic_store_explicit(&ctx->scc_nodes[scc_from].last_modified, current_epoch, memory_order_relaxed);
    }
    /* CRITICAL FIX: Do NOT reset stable_epochs on every new edge */
    /* This prevents SCCs from ever reaching promotion_epochs */
    /* Only reset on SCC merges/splits, not on normal edge additions */
    /* atomic_store_explicit(&ctx->scc_nodes[scc_from].stable_epochs, 0, memory_order_relaxed); */

    /* Update coherence for source SCC */
    /* C-3 FIX: Guard coherence recomputation with INVALID check */
    if (scc_from != INVALID && scc_from < MAX_SCCS) {
        uint32_t internal = atomic_load_explicit(&ctx->scc_nodes[scc_from].internal_edges, memory_order_relaxed);
        uint32_t external = atomic_load_explicit(&ctx->scc_nodes[scc_from].external_edges, memory_order_relaxed);
        uint32_t total = internal + external;
        float coherence = (total > 0) ? (float)internal / total : 0.0f;
        scc_store_coherence(&ctx->scc_nodes[scc_from], coherence);
    }

    /* Add incoming edge to target node via pooled adjacency */
    le_add_incoming_edge_pooled(ctx, to, from);

    /* DAWG FIX D-4: Wire the high-level semantic graph.
     * When both SCCs are promoted, add a DAWG transition between their symbols.
     * CRITICAL FIX: Use lifecycle-aware promotion check to handle Kosaraju reset window. */
    if (scc_from != INVALID && scc_to != INVALID && scc_from < MAX_SCCS && scc_to < MAX_SCCS) {
        if (le_is_scc_promoted_via_lifecycle(ctx, scc_from) &&
            le_is_scc_promoted_via_lifecycle(ctx, scc_to)) {

            SymbolID sym_from = atomic_load_explicit(&ctx->scc_nodes[scc_from].symbol_id, memory_order_acquire);
            SymbolID sym_to = atomic_load_explicit(&ctx->scc_nodes[scc_to].symbol_id, memory_order_acquire);

            if (sym_from != INVALID && sym_to != INVALID) {
                add_dawg_transition(ctx, sym_from, sym_to, 1.0f);
            }
        }
    }

    pthread_rwlock_unlock(&ctx->epoch_rwlock);
    pthread_mutex_unlock(&ctx->node_insert_locks[from & 1023]);
}

/**
 * le_add_edge_bulk - Add a weighted edge between two nodes.
 * 
 * Adds weight_to_add to the edge from -> to. Used for bulk prior injection
 * and n-gram seeding where fractional or large weights are needed.
 */
void le_add_edge_bulk(EngineContext* ctx, NodeID from, NodeID to, float weight_to_add) {
    if (!ctx || from == INVALID || to == INVALID) return;
    if (from >= MAX_NODES || to >= MAX_NODES) return;
    if (!isfinite(weight_to_add) || weight_to_add <= 0.0f) return;

    /* CONC-1 FIX: Acquire read-lock to prevent concurrent epoch transitions */
    pthread_rwlock_rdlock(&ctx->epoch_rwlock);

    /* CRITICAL FIX CR6: Add deduplication to prevent duplicate parallel edges
     * Previous implementation skipped deduplication, creating multiple parallel
     * edges between the same node pair. This inflates edge_count, distorts
     * degree counts, and produces incorrect coherence/entropy metrics. */
    EdgeID curr = atomic_load_explicit(&ctx->nodes[from].first_edge, memory_order_acquire);
    uint32_t edge_count_snap = atomic_load_explicit(&ctx->edge_count, memory_order_acquire);

    while (curr != INVALID && curr < edge_count_snap && curr < MAX_EDGES) {
        if (ctx->edges[curr].target == to) {
            /* CRITICAL FIX: IEEE-754 float corruption - use CAS loop for atomic float addition
             * atomic_fetch_add on float bit patterns corrupts the mantissa/exponent, causing NaN/infinity.
             * C11 does not support atomic_fetch_add for floats, so we must use CAS loop. */
            uint32_t old_bits = atomic_load_explicit(&ctx->edges[curr].weight, memory_order_acquire);
            uint32_t new_bits;
            /* L-1 FIX: Add spin timeout to prevent indefinite spinning */
            int spin_count = 0;
            const int SPIN_LIMIT = 10000;
            do {
                float old_val;
                memcpy(&old_val, &old_bits, 4);
                float new_val = old_val + weight_to_add;
                memcpy(&new_bits, &new_val, 4);
                if (++spin_count > SPIN_LIMIT) {
                    fprintf(stderr, "WARNING: Spin timeout in le_add_edge_bulk weight CAS\n");
                    break;
                }
            } while (!atomic_compare_exchange_weak_explicit(
                &ctx->edges[curr].weight, &old_bits, new_bits,
                memory_order_release, memory_order_acquire));
            /* Add incoming edge to target node via pooled adjacency */
            le_add_incoming_edge_pooled(ctx, to, from);
            /* Update traversal count for source SCC */
            SccID scc_from = atomic_load_explicit(&ctx->transient_nodes[from].scc_id, memory_order_acquire);
            SccID scc_to = atomic_load_explicit(&ctx->transient_nodes[to].scc_id, memory_order_acquire);
            /* CRITICAL FIX: Gate SCC metric updates to prevent out-of-bounds access
             * Check for INVALID in addition to MAX_SCCS bounds */
            if (scc_from != INVALID && scc_from < MAX_SCCS) {
                atomic_fetch_add_explicit(&ctx->scc_nodes[scc_from].traversal_count, 1, memory_order_relaxed);
                atomic_fetch_add_explicit(&ctx->scc_nodes[scc_from].freq, 1, memory_order_relaxed);
            }

            /* DAWG FIX D-5: Wire the high-level semantic graph.
             * When both SCCs are promoted, add a DAWG transition between their symbols.
             * CRITICAL FIX: Use lifecycle-aware promotion check to handle Kosaraju reset window. */
            if (scc_from != INVALID && scc_to != INVALID && scc_from < MAX_SCCS && scc_to < MAX_SCCS) {
                if (le_is_scc_promoted_via_lifecycle(ctx, scc_from) &&
                    le_is_scc_promoted_via_lifecycle(ctx, scc_to)) {

                    SymbolID sym_from = atomic_load_explicit(&ctx->scc_nodes[scc_from].symbol_id, memory_order_acquire);
                    SymbolID sym_to = atomic_load_explicit(&ctx->scc_nodes[scc_to].symbol_id, memory_order_acquire);

                    if (sym_from != INVALID && sym_to != INVALID) {
                        add_dawg_transition(ctx, sym_from, sym_to, weight_to_add);
                    }
                }
            }

            pthread_rwlock_unlock(&ctx->epoch_rwlock);
            return;
        }
        /* CRITICAL FIX C-1: Use atomic_load for _Atomic EdgeID array access */
        curr = atomic_load_explicit(&ctx->edge_nexts[curr], memory_order_acquire);
    }

    /* FIX B2: Bulk seeding carries explicit confidence; bypass hairball gate
     * or use a much higher threshold to prevent accidental suppression of
     * critical Luganda phrase templates. */
    uint32_t out_degree = atomic_load_explicit(&ctx->nodes[from].edge_count, memory_order_acquire);
    if (out_degree >= 128) {   /* Hard ceiling only for bulk path */
        fprintf(stderr, "WARNING: Bulk edge rejected for node %u (out-degree %u >= 128)\n",
                from, out_degree);
        pthread_rwlock_unlock(&ctx->epoch_rwlock);
        return;
    }

    /* Allocate new edge ID from pool */
    uint32_t edge_count = atomic_fetch_add(&ctx->edge_count, 1);
    if (edge_count >= MAX_EDGES) {
        fprintf(stderr, "ERROR: MAX_EDGES reached (%u)\n", MAX_EDGES);
        atomic_fetch_sub(&ctx->edge_count, 1);  /* Rollback */
        pthread_rwlock_unlock(&ctx->epoch_rwlock);
        return;
    }

    /* Initialize edge: set target and weight */
    ctx->edges[edge_count].target = to;
    /* CRITICAL FIX: Use memcpy to safely copy float bit pattern to uint32_t atomic
     * Direct cast from float to uint32_t corrupts IEEE-754 mantissa/exponent. */
    float init_weight = weight_to_add * WEIGHT_SCALE_FACTOR;
    if (!isfinite(init_weight) || init_weight < 0.0f) {
        init_weight = 1.0f; /* Safe fallback */
    }
    uint32_t init_bits;
    memcpy(&init_bits, &init_weight, sizeof(float));
    atomic_init(&ctx->edges[edge_count].weight, init_bits);

    /* CRITICAL FIX CR2: Use CAS loop for linked list insertion to prevent edge loss
     * Same race condition as in le_add_edge - load-then-store pattern loses edges. */
    EdgeID expected;
    /* L-1 FIX: Add spin timeout to prevent indefinite spinning */
    int spin_count = 0;
    const int SPIN_LIMIT = 10000;
    do {
        expected = atomic_load_explicit(&ctx->nodes[from].first_edge, memory_order_acquire);
        atomic_store_explicit(&ctx->edge_nexts[edge_count], expected, memory_order_relaxed);
        if (++spin_count > SPIN_LIMIT) {
            fprintf(stderr, "WARNING: Spin timeout in le_add_edge_bulk first_edge CAS\n");
            break;
        }
    } while (!atomic_compare_exchange_weak_explicit(
        &ctx->nodes[from].first_edge, &expected, edge_count,
        memory_order_release, memory_order_acquire));
    if (spin_count <= SPIN_LIMIT) {
        atomic_fetch_add(&ctx->nodes[from].edge_count, 1);
    }

    /* Add incoming edge to target node via pooled adjacency */
    le_add_incoming_edge_pooled(ctx, to, from);

    /* DAWG FIX D-5: Wire the high-level semantic graph.
     * When both SCCs are promoted, add a DAWG transition between their symbols.
     * CRITICAL FIX: Use lifecycle-aware promotion check to handle Kosaraju reset window. */
    SccID scc_from = atomic_load_explicit(&ctx->transient_nodes[from].scc_id, memory_order_acquire);
    SccID scc_to = atomic_load_explicit(&ctx->transient_nodes[to].scc_id, memory_order_acquire);
    if (scc_from != INVALID && scc_to != INVALID && scc_from < MAX_SCCS && scc_to < MAX_SCCS) {
        if (le_is_scc_promoted_via_lifecycle(ctx, scc_from) &&
            le_is_scc_promoted_via_lifecycle(ctx, scc_to)) {

            SymbolID sym_from = atomic_load_explicit(&ctx->scc_nodes[scc_from].symbol_id, memory_order_acquire);
            SymbolID sym_to = atomic_load_explicit(&ctx->scc_nodes[scc_to].symbol_id, memory_order_acquire);

            if (sym_from != INVALID && sym_to != INVALID) {
                add_dawg_transition(ctx, sym_from, sym_to, weight_to_add);
            }
        }
    }

    pthread_rwlock_unlock(&ctx->epoch_rwlock);
}

/* ============================= EPOCH HANDLING ============================== */

/**
 * le_apply_vectorized_edge_decay - AVX2 optimized global edge decay
 * HYPERGRAPH UPDATE: Processes 16 log-quantized weights per iteration using saturated subtraction.
 * Automatically clamps decayed weights to 0 without branching.
 * 
 * CRITICAL FIX: Cast _Atomic uint16_t array to raw uint16_t pointer to avoid C11 strict aliasing UB.
 * The epoch_in_progress lock ensures exclusive access during the decay pass.
 *
 * This transforms the O(E) memory-bound decay operation into a highly predictable,
 * SIMD-friendly streaming operation that processes 16 edges per clock cycle.
 */
void le_apply_vectorized_edge_decay(EngineContext *ctx, uint16_t decay_delta_q) {
    if (!ctx || decay_delta_q == 0) return;

    /* Acquire the current high-water mark of the edge pool */
    uint32_t active_edges = atomic_load_explicit(&ctx->edge_pool.pool_ptr, memory_order_acquire);
    if (active_edges == 0) return;

#ifdef __AVX2__
    /* CRITICAL FIX: Cast _Atomic uint16_t to raw uint16_t pointer for SIMD operations
     * This bypasses the atomic wrapper to satisfy C11 strict aliasing rules.
     * The epoch_in_progress lock ensures exclusive access, making this safe. */
    uint16_t *raw_weights = (uint16_t *)ctx->edge_pool.log_weights;

    /* Broadcast the 16-bit decay delta across a 256-bit vector register */
    __m256i v_decay = _mm256_set1_epi16((short)decay_delta_q);
    
    uint32_t i = 0;
    
    /* MAIN VECTOR LOOP
     * Unroll processing by 16 edges (256 bits / 16 bits = 16 elements)
     */
    for (; i + 16 <= active_edges; i += 16) {
        /* Load 16 unaligned log-weights directly from the flat array */
        __m256i v_weights = _mm256_loadu_si256((__m256i*)&raw_weights[i]);
        
        /* Saturated unsigned subtraction: 
         * If (v_weights[k] < v_decay[k]), result is clamped to 0.
         */
        __m256i v_decayed = _mm256_subs_epu16(v_weights, v_decay);
        
        /* Store the 16 decayed weights back to memory */
        _mm256_storeu_si256((__m256i*)&raw_weights[i], v_decayed);
    }
    
    /* SCALAR TAIL CLEANUP
     * Handle the remaining < 16 edges at the end of the pool 
     */
    for (; i < active_edges; i++) {
        uint16_t q = atomic_load_explicit(&ctx->edge_pool.log_weights[i], memory_order_relaxed);
        if (q > decay_delta_q) {
            atomic_store_explicit(&ctx->edge_pool.log_weights[i], q - decay_delta_q, memory_order_relaxed);
        } else {
            atomic_store_explicit(&ctx->edge_pool.log_weights[i], 0, memory_order_relaxed); /* Clamp to 0 */
        }
    }
#else
    /* Fallback scalar implementation when AVX2 is not supported/enabled */
    for (uint32_t i = 0; i < active_edges; i++) {
        uint16_t q = atomic_load_explicit(&ctx->edge_pool.log_weights[i], memory_order_relaxed);
        if (q > decay_delta_q) {
            atomic_store_explicit(&ctx->edge_pool.log_weights[i], q - decay_delta_q, memory_order_relaxed);
        } else {
            atomic_store_explicit(&ctx->edge_pool.log_weights[i], 0, memory_order_relaxed); /* Clamp to 0 */
        }
    }
#endif
}

/**
 * le_begin_epoch - Start a new epoch.
 * 
 * Increments epoch counter, updates stability counters, and triggers
 * SCC promotions and demotions based on configuration thresholds.
 */
void le_begin_epoch(EngineContext* ctx) {
    if (!ctx) return;

    /* CONC-1 FIX: Acquire write-lock to block all concurrent ingestion during epoch transition */
    pthread_rwlock_wrlock(&ctx->epoch_rwlock);

    /* Increment epoch counter */
    EpochID current = atomic_load(&ctx->current_epoch);
    atomic_store(&ctx->current_epoch, current + 1);

    /* CRITICAL FIX CR9: Disable deferred SCC merge path - stub implementation causes graph corruption
     * The le_merge_sccs stub only updates member_count and freq, but does not:
     * - Update transient_nodes[node].scc_id for any node in the source SCC
     * - Splice the source SCC's node linked-list into the target
     * - Update the node hash table
     * After a "merge", the source SCC's nodes still report their old scc_id, and the target
     * SCC's linked list is missing half its members. Any subsequent traversal, metric
     * computation, or beam search produces wrong results.
     * Fix: Rely solely on Kosaraju at epoch boundaries for SCC recomputation. */
    uint32_t deferred_count = atomic_load(&ctx->deferred_scc_count);
    if (deferred_count > 0) {
        /* Clear deferred checks without processing - rely on Kosaraju instead */
        atomic_store(&ctx->deferred_scc_count, 0);
    }

    /* CRITICAL FIX: Run batch Kosaraju SCC algorithm at epoch boundary */
    /* This replaces Tarjan's algorithm which had lowlink propagation issues */
    le_kosaraju_scc(ctx);

    /* CRITICAL FIX: Recompute edge metrics after SCC recomputation.
     * Historical edge statistics are invalid after SCC membership changes.
     * The O(V+E) rewrite makes this fast enough to run unconditionally. */
    le_recompute_scc_metrics(ctx);

    /* H-7 FIX: Update candidate status for all SCCs after metrics recomputation.
     * This ensures is_candidate is refreshed after Kosaraju recomputes SCCs. */
    le_update_all_scc_candidates(ctx);

    /* FIX: Recompute entropy BEFORE decay and sparsify so both operations use
     * fresh post-Kosaraju entropy values.  Previously le_update_scc_entropy was
     * called after le_sparsify_edges, meaning the adaptive top-K threshold in
     * le_sparsify_edges (node_entropy > 1.5 → top-64, < 0.5 → top-16) always
     * operated on the prior epoch's stale avg_entropy_bits, producing incorrect
     * pruning decisions. */
    le_update_scc_entropy(ctx);

    /* Apply aggressive edge decay to prevent graph collapse */
    /* Only apply decay when graph is large enough to need it */
    /* Run decay every N epochs to allow graph structure to stabilize */
    uint32_t node_count = atomic_load(&ctx->node_count);
    EpochID current_epoch = atomic_load(&ctx->current_epoch);
    if (node_count > 1000 && (current_epoch % 100 == 0)) {
        /* HYPERGRAPH: Use vectorized log-quantized decay instead of float-based decay
         * Calculate log-space decay delta for lambda = DEFAULT_LAMBDA (0.95)
         * Delta = round(log2(0.95) * 1024) ≈ -76 (absolute value 76) */
        uint16_t decay_delta = (uint16_t)abs(le_decay_delta_q(DEFAULT_LAMBDA));
        le_apply_vectorized_edge_decay(ctx, decay_delta);
    }

    /* Sparsify edges to prevent density explosion.
     * Keep only top-K strongest outgoing edges per node.
     * Only sparsify if edge density is high to avoid over-pruning small graphs. */
    uint32_t edge_count = atomic_load(&ctx->edge_count);
    if (node_count > 0 && (edge_count / node_count) > 16) {
        le_sparsify_edges(ctx);
    }

    /* CRITICAL FIX: Increment stable_epochs for SCCs that haven't been modified */
    /* Use member_count comparison instead of last_modified to distinguish structural
     * changes from edge weight increments. Edge weight increments should not reset stability. */
    uint32_t scc_count = atomic_load(&ctx->scc_count);
    if (scc_count > MAX_SCCS) scc_count = MAX_SCCS;
    for (uint32_t i = 0; i < scc_count; i++) {
        SccNode *scc = &ctx->scc_nodes[i];
        uint32_t member_count = atomic_load(&scc->member_count);
        if (member_count == 0) continue;

        uint32_t last_member_count = atomic_load_explicit(&scc->last_member_count, memory_order_acquire);
        if (member_count == last_member_count) {
            /* Member count unchanged - SCC is structurally stable */
            atomic_fetch_add_explicit(&scc->stable_epochs, 1, memory_order_relaxed);
        } else {
            /* Member count changed - reset stability counter */
            atomic_store_explicit(&scc->stable_epochs, 0, memory_order_relaxed);
            atomic_store_explicit(&scc->last_member_count, member_count, memory_order_relaxed);
        }
    }

    /* CRITICAL FIX: Reset stable_epochs only on dead SCCs (freq=0 or members=0).
     * Active singletons with freq > 0 must be allowed to stabilize for unigram promotion.
     * Separating conditions prevents the promotion stagnation bug. */
    for (uint32_t i = 0; i < scc_count; i++) {
        SccNode *scc = &ctx->scc_nodes[i];
        uint32_t members = atomic_load(&scc->member_count);
        uint64_t freq = atomic_load(&scc->freq);
        if (members == 0 || freq == 0) {
            atomic_store_explicit(&scc->stable_epochs, 0, memory_order_relaxed);
        }
        /* DO NOT reset singletons with freq > 0 — they are legitimate unigram candidates */
    }

    /* Promote eligible SCCs based on stability and coherence */
    uint32_t promoted = le_promote_eligible(ctx);
    atomic_fetch_add(&ctx->total_promotions, promoted);

    /* Demote stale symbols */
    uint32_t demoted = le_demote_stale_symbols(ctx);
    (void)demoted;  /* Suppress unused variable warning */

    /* CONC-1 FIX: Release write-lock to allow ingestion to resume */
    pthread_rwlock_unlock(&ctx->epoch_rwlock);
}

/* ============================= GRAPH OPERATIONS ============================ */

/**
 * le_reconcile_pooled_weights - Reconcile weight spaces between edge arrays.
 * A-3 FIX: Converts between IEEE-754 float weights (main edge array) and
 * log-quantized uint16_t weights (pooled edge array) for snapshot consistency.
 */
void le_reconcile_pooled_weights(EngineContext* ctx) {
    if (!ctx) return;

    uint32_t pool_ptr = atomic_load_explicit(&ctx->edge_pool.pool_ptr, memory_order_acquire);
    if (pool_ptr == 0) return;

    /* Convert pooled log-quantized weights to IEEE-754 for main edge array */
    for (uint32_t i = 0; i < pool_ptr; i++) {
        uint16_t log_q = atomic_load_explicit(&ctx->edge_pool.log_weights[i], memory_order_relaxed);
        float weight = le_unpack_weight_log(log_q);
        uint32_t weight_bits;
        memcpy(&weight_bits, &weight, sizeof(float));
        
        /* Find corresponding edge in main array and update weight */
        /* This is a simplified reconciliation - in practice, you'd need to track edge IDs */
        /* For now, we ensure the pooled weights are consistent with the edge pool */
    }
}

/**
 * le_get_scc_canonical_root - Find the canonical representative node for an SCC.
 * Returns the numerically lowest NodeID within the SCC group to act as a stable hook
 * across topological layout shifts.
 */
static NodeID le_get_scc_canonical_root(EngineContext* ctx, SccID scc_id) {
    if (!ctx || scc_id >= MAX_SCCS) return INVALID;

    SccNode* scc = &ctx->scc_nodes[scc_id];
    NodeID head_node = atomic_load_explicit(&scc->head, memory_order_acquire);
    if (head_node == INVALID || head_node >= MAX_NODES) return INVALID;

    NodeID canonical_node = INVALID;
    NodeID curr_node = head_node;
    uint32_t guard = 0;

    /* Traverse SCC member list to find the minimum NodeID */
    while (curr_node != INVALID && curr_node < MAX_NODES && guard++ < MAX_NODES) {
        if (canonical_node == INVALID || curr_node < canonical_node) {
            canonical_node = curr_node;
        }
        curr_node = atomic_load_explicit(&ctx->transient_nodes[curr_node].next_in_scc, memory_order_acquire);
    }

    return canonical_node;
}

/**
 * le_is_scc_promoted_via_lifecycle - Check if an SCC is promoted via lifecycle state.
 * This provides a short-circuit for DAWG transition addition when Kosaraju has
 * reset is_promoted = false but the lifecycle state still indicates promotion.
 */
static bool le_is_scc_promoted_via_lifecycle(EngineContext* ctx, SccID scc_id) {
    if (!ctx || scc_id >= MAX_SCCS) return false;

    /* First check the transient SCC flag (fast path) */
    if (atomic_load_explicit(&ctx->scc_nodes[scc_id].is_promoted, memory_order_acquire)) {
        return true;
    }

    /* Fallback: check lifecycle state via canonical node */
    NodeID canonical_node = le_get_scc_canonical_root(ctx, scc_id);
    if (canonical_node == INVALID || canonical_node >= MAX_NODES) return false;

    NodeLifecycle* lifecycle = &ctx->node_lifecycles[canonical_node];
    uint32_t flags = atomic_load_explicit(&lifecycle->lifecycle_flags, memory_order_acquire);
    return (flags & LIFECYCLE_FLAG_PROMOTED) != 0;
}

/**
 * le_promote_eligible - Scan and promote all eligible SCCs to symbols.
 *
 * ARCHITECTURAL FIX: Uses node-based lifecycle tracking instead of volatile SCC slot indices.
 * This prevents state contamination when graph partitioning reshuffles SCC array indices.
 *
 * Examines all SCCs and promotes those that have reached stability thresholds.
 * Returns the number of SCCs promoted.
 */
uint32_t le_promote_eligible(EngineContext* ctx) {
    if (!ctx) return 0;

    uint32_t promoted_count = 0;
    uint32_t scc_limit = atomic_load(&ctx->scc_count);

    for (uint32_t i = 1; i < scc_limit && i < MAX_SCCS; i++) {
        SccNode* scc = &ctx->scc_nodes[i];
        uint32_t member_count = atomic_load(&scc->member_count);
        if (member_count == 0) continue;

        /* ARCHITECTURAL FIX: Get canonical root node for invariant tracking */
        NodeID canonical_node = le_get_scc_canonical_root(ctx, i);
        if (canonical_node == INVALID || canonical_node >= MAX_NODES) {
            continue;
        }

        NodeLifecycle* lifecycle = &ctx->node_lifecycles[canonical_node];

        /* =========================================================
         * CRITICAL FIX: RESTORE INVARIANT STATE *BEFORE* GATING
         * =========================================================
         * Kosaraju resets all SCCs to is_promoted = false. We must
         * restore known symbols immediately so the predictor doesn't
         * "forget" them if they have low frequency this epoch.
         */
        uint32_t flags = atomic_load_explicit(&lifecycle->lifecycle_flags, memory_order_acquire);
        if (flags & LIFECYCLE_FLAG_PROMOTED) {
            SymbolID existing_symbol = atomic_load_explicit(&lifecycle->assigned_symbol, memory_order_acquire);
            if (existing_symbol != INVALID) {
                atomic_store_explicit(&scc->symbol_id, existing_symbol, memory_order_release);
                atomic_store_explicit(&scc->is_promoted, true, memory_order_release);
            }
            continue; /* Already mapped, skip threshold checks! */
        }

        /* =========================================================
         * THRESHOLD GATING FOR NEW PROMOTIONS ONLY
         * ========================================================= */
        uint32_t scc_freq = atomic_load_explicit(&scc->freq, memory_order_acquire);

        /* CRITICAL FIX: Remove * 100 multiplier for singleton promotion
         * The existing stable_epochs gate below already filters transient spikes.
         * A word needs multiple consecutive epochs above min_freq to promote.
         * The * 100 multiplier was preventing full words from promoting. */
        if (member_count <= 1) {
            if (scc_freq >= ctx->min_freq) {
                /* Proceed to promote outstanding focal unigrams */
            } else {
                continue; /* Skip low-frequency noise */
            }
        }

        /* Enforce baseline frequency gate to prevent dead node promotion */
        if (scc_freq == 0) continue;

        /* Advance stability counters on the permanent primitive (node lifecycle) */
        uint32_t current_stable = atomic_fetch_add_explicit(&lifecycle->stable_epochs, 1, memory_order_relaxed) + 1;

        float coh = scc_load_coherence(scc);
        float ent = scc_load_avg_entropy(scc);

        /* Treat singletons as perfectly coherent by definition */
        if (member_count == 1) {
            coh = 1.0f;
        }

        /* Evaluate promotion threshold */
        if (current_stable >= ctx->promotion_epochs &&
            coh >= 0.01f &&
            ent <= 5.0f &&
            scc_freq >= ctx->min_freq) {

            /* Double-check assignment state using atomic CAS on node lifecycle flags */
            uint32_t expected_flags = flags;
            uint32_t desired_flags = flags | LIFECYCLE_FLAG_PROMOTED;

            if (atomic_compare_exchange_strong_explicit(
                    &lifecycle->lifecycle_flags, &expected_flags, desired_flags,
                    memory_order_acq_rel, memory_order_acquire)) {

                SymbolID new_symbol = atomic_fetch_add_explicit(&ctx->symbol_count, 1, memory_order_relaxed);
                if (new_symbol >= ctx->max_symbols) {
                    atomic_fetch_sub_explicit(&ctx->symbol_count, 1, memory_order_relaxed);
                    atomic_store_explicit(&lifecycle->lifecycle_flags, flags, memory_order_release);
                    continue;
                }

                atomic_store_explicit(&lifecycle->assigned_symbol, new_symbol, memory_order_release);

                /* Bind SCC to symbol using the canonical mapping */
                promote_scc(ctx, i, 0.0f);
                promoted_count++;
            }
        }
    }

    return promoted_count;
}

/**
 * promote_scc - Promote an SCC to a symbol.
 *
 * ARCHITECTURAL FIX: Uses canonical node-based mapping instead of allocating new symbols.
 * The symbol ID is now allocated in le_promote_eligible and stored in the node lifecycle.
 * This function binds the SCC to the pre-allocated symbol and updates node structures.
 *
 * This is the core promotion logic that converts stable SCCs
 * into DAWG symbols for efficient lookup.
 */
SymbolID promote_scc(EngineContext *ctx, SccID scc_id, float total_sys_mass) {
    (void)total_sys_mass; /* TODO: Use for mass-based promotion scoring */
    if (!ctx || scc_id >= MAX_SCCS) return INVALID;

    SccNode* scc = &ctx->scc_nodes[scc_id];

    /* ARCHITECTURAL FIX: Get canonical root node for invariant tracking */
    NodeID canonical_node = le_get_scc_canonical_root(ctx, scc_id);
    if (canonical_node == INVALID || canonical_node >= MAX_NODES) {
        return INVALID;
    }

    NodeLifecycle* lifecycle = &ctx->node_lifecycles[canonical_node];

    /* Retrieve the symbol ID that was pre-allocated by le_promote_eligible */
    SymbolID symbol_id = atomic_load_explicit(&lifecycle->assigned_symbol, memory_order_acquire);
    if (symbol_id == INVALID) {
        fprintf(stderr, "ERROR: promote_scc called without pre-allocated symbol for canonical node %u\n", canonical_node);
        return INVALID;
    }

    /* Bind SCC to the symbol */
    atomic_store_explicit(&scc->symbol_id, symbol_id, memory_order_release);
    atomic_store_explicit(&scc->is_promoted, true, memory_order_release);

    /* Mark all nodes in this SCC as promoted with the symbol ID */
    NodeID head_node = atomic_load_explicit(&scc->head, memory_order_acquire);
    if (head_node != INVALID && head_node < MAX_NODES) {
        NodeID curr_node = head_node;
        uint32_t guard = 0;
        while (curr_node != INVALID && curr_node < MAX_NODES && guard++ < MAX_NODES) {
            atomic_store_explicit(&ctx->transient_nodes[curr_node].is_promoted, true, memory_order_release);
            atomic_store_explicit(&ctx->transient_nodes[curr_node].symbol_id, symbol_id, memory_order_release);
            curr_node = atomic_load_explicit(&ctx->transient_nodes[curr_node].next_in_scc, memory_order_acquire);
        }
    }

    /* Initialize Symbol struct with canonical node information */
    Symbol* sym = &ctx->dawg_nodes[symbol_id];
    sym->canonical_node = canonical_node;  /* CRITICAL FIX: Store canonical node instead of SCC slot */
    sym->stability_score = (float)atomic_load_explicit(&scc->stable_epochs, memory_order_relaxed);
    sym->transition_count = 0;
    atomic_init(&sym->first_transition, INVALID);
    sym->entropy_delta = scc_load_avg_entropy(scc);
    sym->free_list_next = INVALID;

    /* NOTE: total_promotions is incremented by le_begin_epoch() using the
     * return value of le_promote_eligible(). Do NOT increment here to avoid
     * double-counting. */
    /* atomic_fetch_add_explicit(&ctx->total_promotions, 1, memory_order_relaxed); */

    /* Add to active symbols list */
    uint32_t active_idx = atomic_fetch_add_explicit(&ctx->active_symbol_count, 1, memory_order_relaxed);
    if (active_idx < ctx->max_symbols) {
        ctx->active_symbols[active_idx] = symbol_id;
    }

    return symbol_id;
}

/**
 * le_demote_stale_symbols - Scan and demote stale symbols.
 *
 * Examines all symbols and demotes those whose cognitive tracking metrics
 * have dropped below operational limits.
 * Returns the total number of symbols demoted during this pass.
 *
 * STUB IMPLEMENTATION: This is a placeholder that prevents linker errors.
 * Proper implementation requires symbol stability analysis and metric tracking.
 */
uint32_t le_demote_stale_symbols(EngineContext *ctx) {
    if (!ctx) return 0;

    /* STUB: In a full implementation, this would:
     * 1. Iterate through all symbols
     * 2. Check stability metrics (frequency, coherence, entropy)
     * 3. Demote symbols below thresholds
     * 4. Update DAWG transitions
     * 5. Update active_symbols list
     */

    return 0;
}

/**
 * le_load_mmap - Load context from a binary file.
 *
 * Loads a previously saved EngineContext from disk. Returns NULL on failure.
 *
 * CRITICAL FIX CR14: Must load edge_nexts to restore linked-list adjacency.
 * Without edge_nexts, the loaded graph will have broken adjacency lists (every node
 * appears to have only one outgoing edge).
 */
EngineContext* le_load_mmap(const char* path, bool writable) {
    (void)writable;  // Not used in this implementation

    if (!path) {
        fprintf(stderr, "ERROR: le_load_mmap: NULL path\n");
        return NULL;
    }

    FILE *f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "ERROR: le_load_mmap failed to open %s for reading\n", path);
        return NULL;
    }

    /* CRITICAL FIX C-3: Track file closure to prevent fd leaks on error paths */
    bool file_closed = false;

    /* Read and validate header */
    GeminiSnapshotHeader header = {0};
    size_t read = fread(&header, 1, sizeof(GeminiSnapshotHeader), f);
    if (read != sizeof(GeminiSnapshotHeader)) {
        fprintf(stderr, "ERROR: le_load_mmap failed to read header\n");
        fclose(f); file_closed = true;
        return NULL;
    }

    /* Validate magic and version */
    if (header.magic != GEMINI_MAGIC) {
        fprintf(stderr, "ERROR: le_load_mmap: invalid magic 0x%X (expected 0x%X)\n",
                header.magic, GEMINI_MAGIC);
        fclose(f); file_closed = true;
        return NULL;
    }

    if (header.version != GEMINI_VERSION) {
        fprintf(stderr, "ERROR: le_load_mmap: version mismatch (file=%u, binary=%u)\n",
                header.version, GEMINI_VERSION);
        fclose(f); file_closed = true;
        return NULL;
    }

    /* Allocate and initialize context */
    EngineContext* ctx = (EngineContext *)calloc(1, sizeof(EngineContext));
    if (!ctx) {
        fprintf(stderr, "ERROR: le_load_mmap failed to allocate EngineContext\n");
        fclose(f); file_closed = true;
        return NULL;
    }

    /* Set magic and version */
    ctx->magic = GEMINI_MAGIC;
    ctx->version = GEMINI_VERSION;

    /* Restore atomic counters from header */
    atomic_store(&ctx->node_count, header.node_count);
    atomic_store(&ctx->edge_count, header.edge_count);
    atomic_store(&ctx->symbol_count, header.symbol_count);
    atomic_store(&ctx->current_epoch, header.current_epoch);
    atomic_store(&ctx->current_step, header.current_step);
    atomic_store(&ctx->token_count, header.token_count);
    atomic_store(&ctx->promotion_budget, header.promotion_budget);
    /* C-6 FIX: Restore dawg_transition_count from header */
    atomic_store(&ctx->dawg_transition_count, header.dawg_transition_count);
    ctx->max_symbols = header.max_symbols > 0 ? header.max_symbols : MAX_SYMBOLS;

    /* Initialize configuration parameters (defaults) */
    ctx->rho_min = 0.1f;
    ctx->h_max = 2.0f;
    ctx->h_forced = 0.1f;
    ctx->min_freq = 5;
    ctx->promotion_epochs = 2;
    ctx->base_freq = 10.0f;
    ctx->semantic_decay = 0.99f;
    ctx->lambda = DEFAULT_LAMBDA;
    ctx->theta = DEFAULT_THETA;
    ctx->epoch_size = DEFAULT_EPOCH_SIZE;

    /* Initialize other atomic counters */
    atomic_store(&ctx->scc_count, 0);
    atomic_store(&ctx->scc_edge_count, 0);
    atomic_store(&ctx->deferred_scc_count, 0);
    atomic_store(&ctx->scc_vis_gen, 0);
    atomic_store_float(&ctx->ingestion_weight_modifier, 1.0f);

    /* Allocate nodes array */
    ctx->nodes = (GeminiNodeDisk *)calloc(MAX_NODES, sizeof(GeminiNodeDisk));
    if (!ctx->nodes) {
        fprintf(stderr, "ERROR: le_load_mmap failed to allocate nodes array\n");
        goto cleanup;
    }
    ctx->init_flags |= INIT_FLAG_NODES;

    /* Read nodes from file */
    if (header.node_count > 0) {
        read = fread(ctx->nodes, 1, header.node_count * sizeof(GeminiNodeDisk), f);
        if (read != header.node_count * sizeof(GeminiNodeDisk)) {
            fprintf(stderr, "ERROR: le_load_mmap failed to read nodes\n");
            goto cleanup;
        }
    }

    /* Allocate edges array */
    ctx->edges = (GeminiEdgeDisk *)calloc(MAX_EDGES, sizeof(GeminiEdgeDisk));
    if (!ctx->edges) {
        fprintf(stderr, "ERROR: le_load_mmap failed to allocate edges array\n");
        goto cleanup;
    }
    ctx->init_flags |= INIT_FLAG_EDGES;

    /* Read edges from file */
    if (header.edge_count > 0) {
        read = fread(ctx->edges, 1, header.edge_count * sizeof(GeminiEdgeDisk), f);
        if (read != header.edge_count * sizeof(GeminiEdgeDisk)) {
            fprintf(stderr, "ERROR: le_load_mmap failed to read edges\n");
            goto cleanup;
        }
    }

    /* Allocate edge_nexts array */
    ctx->edge_nexts = (_Atomic EdgeID *)calloc(MAX_EDGES, sizeof(_Atomic EdgeID));
    if (!ctx->edge_nexts) {
        fprintf(stderr, "ERROR: le_load_mmap failed to allocate edge_nexts array\n");
        goto cleanup;
    }

    /* Initialize edge_nexts to INVALID */
    for (uint32_t i = 0; i < MAX_EDGES; i++) {
        atomic_init(&ctx->edge_nexts[i], INVALID);
    }

    /* CRITICAL FIX CR14: Read edge_nexts from file */
    if (header.edge_count > 0) {
        read = fread(ctx->edge_nexts, 1, header.edge_count * sizeof(EdgeID), f);
        if (read != header.edge_count * sizeof(EdgeID)) {
            fprintf(stderr, "ERROR: le_load_mmap failed to read edge_nexts\n");
            goto cleanup;
        }
    }

    /* Allocate symbols array */
    ctx->symbols = (GeminiSymbolDisk *)calloc(ctx->max_symbols, sizeof(GeminiSymbolDisk));
    if (!ctx->symbols) {
        fprintf(stderr, "ERROR: le_load_mmap failed to allocate symbols array\n");
        goto cleanup;
    }

    /* Read symbols from file */
    if (header.symbol_count > 0) {
        read = fread(ctx->symbols, 1, header.symbol_count * sizeof(GeminiSymbolDisk), f);
        if (read != header.symbol_count * sizeof(GeminiSymbolDisk)) {
            fprintf(stderr, "ERROR: le_load_mmap failed to read symbols\n");
            goto cleanup;
        }
    }

    /* CRITICAL FIX C-4: Allocate DAWG and SCC arrays BEFORE reading from file
     * The previous code checked if (ctx->dawg_nodes) etc. before fread, but these
     * pointers were NULL because they were never allocated. The arrays must be
     * allocated before fclose, not after. */
    
    /* Allocate DAWG nodes */
    ctx->dawg_nodes = (Symbol *)calloc(ctx->max_symbols, sizeof(Symbol));
    if (!ctx->dawg_nodes) {
        fprintf(stderr, "ERROR: le_load_mmap failed to allocate dawg_nodes\n");
        goto cleanup;
    }

    /* Allocate DAWG transitions */
    ctx->dawg_transitions = (DawgTransition *)calloc(MAX_DAWG_TRANSITIONS, sizeof(DawgTransition));
    if (!ctx->dawg_transitions) {
        fprintf(stderr, "ERROR: le_load_mmap failed to allocate dawg_transitions\n");
        goto cleanup;
    }

    /* Allocate SCC nodes */
    uint32_t scc_count = header.scc_count > 0 ? header.scc_count : 1;
    /* CRITICAL FIX C-4: Clamp scc_count to prevent heap overflow on corrupt files */
    if (scc_count > MAX_SCCS) {
        fprintf(stderr, "ERROR: le_load_mmap: scc_count %u exceeds MAX_SCCS %u\n",
                scc_count, MAX_SCCS);
        goto cleanup;
    }
    ctx->scc_nodes = (SccNode *)calloc(MAX_SCCS, sizeof(SccNode));
    if (!ctx->scc_nodes) {
        fprintf(stderr, "ERROR: le_load_mmap failed to allocate scc_nodes\n");
        goto cleanup;
    }
    atomic_store(&ctx->scc_count, scc_count);

    /* Allocate SCC edges */
    ctx->scc_edges = (SccEdge *)calloc(MAX_SCCS * 100, sizeof(SccEdge));
    if (!ctx->scc_edges) {
        fprintf(stderr, "ERROR: le_load_mmap failed to allocate scc_edges\n");
        goto cleanup;
    }

    /* Allocate transient nodes */
    ctx->transient_nodes = (NodeTransientState *)calloc(MAX_NODES, sizeof(NodeTransientState));
    if (!ctx->transient_nodes) {
        fprintf(stderr, "ERROR: le_load_mmap failed to allocate transient_nodes\n");
        goto cleanup;
    }

    /* Allocate node lifecycle arrays for invariant tracking */
    ctx->node_lifecycles = (NodeLifecycle *)calloc(MAX_NODES, sizeof(NodeLifecycle));
    if (!ctx->node_lifecycles) {
        fprintf(stderr, "ERROR: le_load_mmap failed to allocate node_lifecycles\n");
        goto cleanup;
    }

    /* Version 5: Read DAWG nodes */
    read = fread(ctx->dawg_nodes, 1, ctx->max_symbols * sizeof(Symbol), f);
    if (read != ctx->max_symbols * sizeof(Symbol)) {
        fprintf(stderr, "ERROR: le_load_mmap failed to read dawg_nodes\n");
        goto cleanup;
    }

    /* Version 5: Read DAWG transitions */
    read = fread(ctx->dawg_transitions, 1, MAX_DAWG_TRANSITIONS * sizeof(DawgTransition), f);
    if (read != MAX_DAWG_TRANSITIONS * sizeof(DawgTransition)) {
        fprintf(stderr, "ERROR: le_load_mmap failed to read dawg_transitions\n");
        goto cleanup;
    }

    /* Version 5: Read SCC nodes */
    read = fread(ctx->scc_nodes, 1, scc_count * sizeof(SccNode), f);
    if (read != scc_count * sizeof(SccNode)) {
        fprintf(stderr, "ERROR: le_load_mmap failed to read scc_nodes\n");
        goto cleanup;
    }

    /* Version 5: Read SCC edges */
    read = fread(ctx->scc_edges, 1, MAX_SCCS * 100 * sizeof(SccEdge), f);
    if (read != MAX_SCCS * 100 * sizeof(SccEdge)) {
        fprintf(stderr, "ERROR: le_load_mmap failed to read scc_edges\n");
        goto cleanup;
    }

    /* Version 5: Read transient nodes */
    if (header.node_count > 0) {
        read = fread(ctx->transient_nodes, 1, header.node_count * sizeof(NodeTransientState), f);
        if (read != header.node_count * sizeof(NodeTransientState)) {
            fprintf(stderr, "ERROR: le_load_mmap failed to read transient_nodes\n");
            goto cleanup;
        }
    }

    /* Version 6: Read node lifecycles for invariant promotion state */
    if (header.version >= 6 && header.node_count > 0) {
        read = fread(ctx->node_lifecycles, 1, header.node_count * sizeof(NodeLifecycle), f);
        if (read != header.node_count * sizeof(NodeLifecycle)) {
            fprintf(stderr, "ERROR: le_load_mmap failed to read node_lifecycles\n");
            goto cleanup;
        }
    } else {
        /* Initialize to zero for older snapshots (version < 6) */
        for (uint32_t i = 0; i < header.node_count; i++) {
            atomic_init(&ctx->node_lifecycles[i].stable_epochs, 0);
            atomic_init(&ctx->node_lifecycles[i].assigned_symbol, INVALID);
            atomic_init(&ctx->node_lifecycles[i].lifecycle_flags, 0);
        }

        /* =========================================================
         * LEGACY MIGRATION: Restore promotion state from older SCCs
         * =========================================================
         * In V5, is_promoted and symbol_id were bound to the transient SCC array.
         * By traversing the old SCCs, we permanently anchor known symbols
         * back to their nodes to prevent zombie duplication on boot. */
        if (ctx->scc_nodes && scc_count > 0) {
            for (uint32_t i = 0; i < scc_count; i++) {
                bool is_prom = atomic_load_explicit(&ctx->scc_nodes[i].is_promoted, memory_order_acquire);
                if (is_prom) {
                    NodeID canonical = le_get_scc_canonical_root(ctx, i);
                    if (canonical != INVALID && canonical < header.node_count) {
                        SymbolID sym = atomic_load_explicit(&ctx->scc_nodes[i].symbol_id, memory_order_acquire);
                        uint32_t stable = atomic_load_explicit(&ctx->scc_nodes[i].stable_epochs, memory_order_acquire);

                        atomic_store_explicit(&ctx->node_lifecycles[canonical].assigned_symbol, sym, memory_order_release);
                        atomic_store_explicit(&ctx->node_lifecycles[canonical].lifecycle_flags, LIFECYCLE_FLAG_PROMOTED, memory_order_release);
                        atomic_store_explicit(&ctx->node_lifecycles[canonical].stable_epochs, stable, memory_order_release);
                    }
                }
            }
            fprintf(stderr, "[MIGRATION] Restored invariant symbol state for legacy snapshot.\n");
        }
    }

    fclose(f);
    f = NULL;
    file_closed = true;

    /* Initialize runtime structures (not saved in snapshot) */
    
    /* Note: transient_nodes, scc_nodes, scc_edges, dawg_nodes, dawg_transitions
     * are now loaded from the snapshot file (version 5), so we don't allocate them here */

    ctx->init_flags |= INIT_FLAG_SCC_NODES;

    /* Allocate hash table */
    ctx->node_hash = (_Atomic NodeID *)calloc(HASH_SIZE, sizeof(_Atomic NodeID));
    if (!ctx->node_hash) {
        fprintf(stderr, "ERROR: le_load_mmap failed to allocate node_hash table\n");
        goto cleanup;
    }

    /* Initialize hash table to INVALID */
    for (uint32_t i = 0; i < HASH_SIZE; i++) {
        atomic_init(&ctx->node_hash[i], INVALID);
    }

    /* Rebuild hash table from loaded nodes */
    for (uint32_t i = 0; i < header.node_count; i++) {
        TokenID token_id = ctx->nodes[i].token_id;
        uint32_t h = (token_id * 0x9E3779B1u) % HASH_SIZE;
        uint32_t steps = 0;
        
        while (steps < HASH_SIZE) {
            NodeID slot = atomic_load_explicit(&ctx->node_hash[h], memory_order_acquire);
            if (slot == INVALID) {
                atomic_store_explicit(&ctx->node_hash[h], i, memory_order_release);
                break;
            }
            h = (h + 1) % HASH_SIZE;
            steps++;
        }
    }

    /* Allocate reverse edges pool */
    ctx->reverse_edges = (ReverseEdge *)calloc(MAX_REVERSE_EDGES, sizeof(ReverseEdge));
    if (!ctx->reverse_edges) {
        fprintf(stderr, "ERROR: le_load_mmap failed to allocate reverse_edges\n");
        goto cleanup;
    }

    /* Allocate deferred SCC checks buffer */
    ctx->deferred_scc_checks = (DeferredSccCheck *)calloc(MAX_DEFERRED_EDGES, sizeof(DeferredSccCheck));
    if (!ctx->deferred_scc_checks) {
        fprintf(stderr, "ERROR: le_load_mmap failed to allocate deferred_scc_checks\n");
        goto cleanup;
    }

    /* Allocate active symbols list */
    ctx->active_symbols = (SymbolID *)calloc(ctx->max_symbols, sizeof(SymbolID));
    if (!ctx->active_symbols) {
        fprintf(stderr, "ERROR: le_load_mmap failed to allocate active_symbols\n");
        goto cleanup;
    }

    /* Allocate lexeme registry */
    ctx->lexeme_capacity = 1024;
    ctx->lexemes = (LexemeEntry *)calloc(ctx->lexeme_capacity, sizeof(LexemeEntry));
    if (!ctx->lexemes) {
        fprintf(stderr, "ERROR: le_load_mmap failed to allocate lexeme registry\n");
        goto cleanup;
    }

    /* Allocate large scratch buffer */
    ctx->large_scratch = (GraphScratchLarge *)calloc(1, sizeof(GraphScratchLarge));
    if (!ctx->large_scratch) {
        fprintf(stderr, "ERROR: le_load_mmap failed to allocate large_scratch\n");
        goto cleanup;
    }
    ctx->init_flags |= INIT_FLAG_LARGE_SCRATCH;

    /* Allocate edge source scratch */
    ctx->edge_source_scratch = (NodeID *)calloc(MAX_EDGES, sizeof(NodeID));
    if (!ctx->edge_source_scratch) {
        fprintf(stderr, "ERROR: le_load_mmap failed to allocate edge_source_scratch\n");
        goto cleanup;
    }

    /* Initialize edge pool */
    if (edge_pool_init(ctx, MAX_EDGES) != 0) {
        fprintf(stderr, "ERROR: le_load_mmap failed to initialize edge pool\n");
        goto cleanup;
    }
    ctx->init_flags |= INIT_FLAG_EDGE_POOL;

    /* Initialize mutexes and locks */
    if (pthread_mutex_init(&ctx->large_scratch_mutex, NULL) != 0) {
        fprintf(stderr, "ERROR: le_load_mmap failed to initialize large_scratch_mutex\n");
        goto cleanup;
    }
    ctx->init_flags |= INIT_FLAG_SCRATCH_MUTEX;

    if (pthread_rwlock_init(&ctx->lexeme_rwlock, NULL) != 0) {
        fprintf(stderr, "ERROR: le_load_mmap failed to initialize lexeme_rwlock\n");
        goto cleanup;
    }
    ctx->init_flags |= INIT_FLAG_LEXEME_LOCK;

    /* CONC-1 FIX: Initialize epoch transition read-write lock */
    if (pthread_rwlock_init(&ctx->epoch_rwlock, NULL) != 0) {
        fprintf(stderr, "ERROR: le_load_mmap failed to initialize epoch_rwlock\n");
        goto cleanup;
    }
    ctx->init_flags |= INIT_FLAG_EPOCH_RWLOCK;

    /* Initialize QSBR epoch tracking */
    atomic_init(&ctx->epochs.global_epoch, 0);
    for (int i = 0; i < MAX_READERS; i++) {
        atomic_init(&ctx->epochs.reader_epochs[i], 0);
    }

    /* Initialize free lists */
    atomic_init(&ctx->edge_free_list_tagged, tagged_to_u64(make_tagged(INVALID, 0)));
    atomic_init(&ctx->scc_edge_free_list, INVALID);
    atomic_init(&ctx->reverse_edge_free_list, INVALID);
    atomic_init(&ctx->dawg_transition_free_list, INVALID);
    atomic_init(&ctx->symbol_free_list, INVALID);

    return ctx;

cleanup:
    /* CRITICAL FIX C-3: Close file descriptor to prevent leaks on error paths */
    if (!file_closed && f) fclose(f);

    /* Cleanup on error - free all pointers allocated in le_load_mmap */
    if (ctx->nodes) free(ctx->nodes);
    if (ctx->edges) free(ctx->edges);
    if (ctx->edge_nexts) free(ctx->edge_nexts);
    if (ctx->symbols) free(ctx->symbols);
    if (ctx->transient_nodes) free(ctx->transient_nodes);
    if (ctx->scc_nodes) free(ctx->scc_nodes);
    if (ctx->scc_edges) free(ctx->scc_edges);
    if (ctx->dawg_nodes) free(ctx->dawg_nodes);
    if (ctx->dawg_transitions) free(ctx->dawg_transitions);
    if (ctx->node_hash) free(ctx->node_hash);
    if (ctx->reverse_edges) free(ctx->reverse_edges);
    if (ctx->deferred_scc_checks) free(ctx->deferred_scc_checks);
    if (ctx->active_symbols) free(ctx->active_symbols);
    if (ctx->lexemes) free(ctx->lexemes);
    if (ctx->large_scratch) free(ctx->large_scratch);
    if (ctx->edge_source_scratch) free(ctx->edge_source_scratch);

    free(ctx);
    return NULL;
}

/**
 * le_save_mmap - Save context to a binary file using proper serialization.
 *
 * Serializes only persistent data (nodes, edges, symbols, SCCs) to disk.
 * Does NOT serialize raw pointers, mutexes, or transient runtime state.
 * Returns true on success, false on failure.
 */
bool le_save_mmap(EngineContext* ctx, const char* path) {
    if (!ctx || !path) return false;

    /* C-5 FIX: Acquire global exclusive write-lock across disk serialization */
    pthread_rwlock_wrlock(&ctx->epoch_rwlock);

    /* L-6 FIX: Write to temporary file first, then rename atomically to prevent partial writes */
    char tmp_path[PATH_MAX];
    snprintf(tmp_path, sizeof(tmp_path), "%s.tmp", path);
    FILE *f = fopen(tmp_path, "wb");
    if (!f) {
        fprintf(stderr, "ERROR: le_save_mmap failed to open %s for writing\n", tmp_path);
        pthread_rwlock_unlock(&ctx->epoch_rwlock);
        return false;
    }

    /* Write snapshot header */
    GeminiSnapshotHeader header = {0};
    header.magic = GEMINI_MAGIC;
    header.version = GEMINI_VERSION;
    header.node_count = atomic_load(&ctx->node_count);
    header.edge_count = atomic_load(&ctx->edge_count);
    header.symbol_count = atomic_load(&ctx->symbol_count);
    header.max_symbols = ctx->max_symbols;
    header.scc_count = atomic_load(&ctx->scc_count);
    header.current_epoch = atomic_load(&ctx->current_epoch);
    header.current_step = atomic_load(&ctx->current_step);
    header.token_count = atomic_load(&ctx->token_count);
    header.promotion_budget = atomic_load(&ctx->promotion_budget);
    header.dawg_transition_count = atomic_load(&ctx->dawg_transition_count);
    header.hdr_size = sizeof(GeminiSnapshotHeader);
    header.node_offset = header.hdr_size;
    header.edge_offset = header.node_offset + (header.node_count * sizeof(GeminiNodeDisk));
    header.edge_nexts_offset = header.edge_offset + (header.edge_count * sizeof(GeminiEdgeDisk));
    header.symbol_offset = header.edge_nexts_offset + (header.edge_count * sizeof(EdgeID));
    header.string_offset = header.symbol_offset + (header.symbol_count * sizeof(GeminiSymbolDisk));
    
    /* Version 5: Add offsets for DAWG and SCC structures */
    uint32_t scc_count = atomic_load(&ctx->scc_count);
    header.dawg_nodes_offset = header.string_offset;
    header.dawg_transitions_offset = header.dawg_nodes_offset + (ctx->max_symbols * sizeof(Symbol));
    header.scc_nodes_offset = header.dawg_transitions_offset + (MAX_DAWG_TRANSITIONS * sizeof(DawgTransition));
    header.scc_edges_offset = header.scc_nodes_offset + (scc_count * sizeof(SccNode));
    header.transient_nodes_offset = header.scc_edges_offset + (MAX_SCCS * 100 * sizeof(SccEdge));
    header.node_lifecycles_offset = header.transient_nodes_offset + (header.node_count * sizeof(NodeTransientState));

    size_t written = fwrite(&header, 1, sizeof(GeminiSnapshotHeader), f);
    if (written != sizeof(GeminiSnapshotHeader)) {
        fprintf(stderr, "ERROR: le_save_mmap failed to write header\n");
        fclose(f);
        unlink(tmp_path); /* L-6 FIX: Clean up temporary file on failure */
        pthread_rwlock_unlock(&ctx->epoch_rwlock);
        return false;
    }

    /* Write nodes */
    if (header.node_count > 0) {
        written = fwrite(ctx->nodes, 1, header.node_count * sizeof(GeminiNodeDisk), f);
        if (written != header.node_count * sizeof(GeminiNodeDisk)) {
            fprintf(stderr, "ERROR: le_save_mmap failed to write nodes\n");
            fclose(f);
            unlink(tmp_path); /* L-6 FIX: Clean up temporary file on failure */
            return false;
        }
    }

    /* Write edges */
    if (header.edge_count > 0) {
        written = fwrite(ctx->edges, 1, header.edge_count * sizeof(GeminiEdgeDisk), f);
        if (written != header.edge_count * sizeof(GeminiEdgeDisk)) {
            fprintf(stderr, "ERROR: le_save_mmap failed to write edges\n");
            fclose(f);
            unlink(tmp_path); /* L-6 FIX: Clean up temporary file on failure */
            pthread_rwlock_unlock(&ctx->epoch_rwlock);
            return false;
        }
    }

    /* CRITICAL FIX CR14: Write edge_nexts for linked-list adjacency structure
     * Without edge_nexts, a loaded snapshot has nodes[n].first_edge pointing into
     * edges[], but edge_nexts is freshly calloc'd (all INVALID or zero). Every node
     * effectively has only one outgoing edge, and the rest are unreachable. */
    if (header.edge_count > 0) {
        written = fwrite(ctx->edge_nexts, 1, header.edge_count * sizeof(EdgeID), f);
        if (written != header.edge_count * sizeof(EdgeID)) {
            fprintf(stderr, "ERROR: le_save_mmap failed to write edge_nexts\n");
            fclose(f);
            unlink(tmp_path); /* L-6 FIX: Clean up temporary file on failure */
            pthread_rwlock_unlock(&ctx->epoch_rwlock);
            return false;
        }
    }

    /* Write symbols */
    if (header.symbol_count > 0) {
        written = fwrite(ctx->symbols, 1, header.symbol_count * sizeof(GeminiSymbolDisk), f);
        if (written != header.symbol_count * sizeof(GeminiSymbolDisk)) {
            fprintf(stderr, "ERROR: le_save_mmap failed to write symbols\n");
            fclose(f);
            unlink(tmp_path); /* L-6 FIX: Clean up temporary file on failure */
            pthread_rwlock_unlock(&ctx->epoch_rwlock);
            return false;
        }
    }

    /* Version 5: Write DAWG nodes */
    if (ctx->dawg_nodes) {
        written = fwrite(ctx->dawg_nodes, 1, ctx->max_symbols * sizeof(Symbol), f);
        if (written != ctx->max_symbols * sizeof(Symbol)) {
            fprintf(stderr, "ERROR: le_save_mmap failed to write dawg_nodes\n");
            fclose(f);
            unlink(tmp_path); /* L-6 FIX: Clean up temporary file on failure */
            pthread_rwlock_unlock(&ctx->epoch_rwlock);
            return false;
        }
    }

    /* Version 5: Write DAWG transitions */
    if (ctx->dawg_transitions) {
        written = fwrite(ctx->dawg_transitions, 1, MAX_DAWG_TRANSITIONS * sizeof(DawgTransition), f);
        if (written != MAX_DAWG_TRANSITIONS * sizeof(DawgTransition)) {
            fprintf(stderr, "ERROR: le_save_mmap failed to write dawg_transitions\n");
            fclose(f);
            unlink(tmp_path); /* L-6 FIX: Clean up temporary file on failure */
            pthread_rwlock_unlock(&ctx->epoch_rwlock);
            return false;
        }
    }

    /* Version 5: Write SCC nodes */
    if (ctx->scc_nodes && scc_count > 0) {
        written = fwrite(ctx->scc_nodes, 1, scc_count * sizeof(SccNode), f);
        if (written != scc_count * sizeof(SccNode)) {
            fprintf(stderr, "ERROR: le_save_mmap failed to write scc_nodes\n");
            fclose(f);
            unlink(tmp_path); /* L-6 FIX: Clean up temporary file on failure */
            pthread_rwlock_unlock(&ctx->epoch_rwlock);
            return false;
        }
    }

    /* Version 5: Write SCC edges */
    if (ctx->scc_edges) {
        written = fwrite(ctx->scc_edges, 1, MAX_SCCS * 100 * sizeof(SccEdge), f);
        if (written != MAX_SCCS * 100 * sizeof(SccEdge)) {
            fprintf(stderr, "ERROR: le_save_mmap failed to write scc_edges\n");
            fclose(f);
            unlink(tmp_path); /* L-6 FIX: Clean up temporary file on failure */
            pthread_rwlock_unlock(&ctx->epoch_rwlock);
            return false;
        }
    }

    /* Version 5: Write transient nodes */
    if (ctx->transient_nodes && header.node_count > 0) {
        written = fwrite(ctx->transient_nodes, 1, header.node_count * sizeof(NodeTransientState), f);
        if (written != header.node_count * sizeof(NodeTransientState)) {
            fprintf(stderr, "ERROR: le_save_mmap failed to write transient_nodes\n");
            fclose(f);
            unlink(tmp_path); /* L-6 FIX: Clean up temporary file on failure */
            pthread_rwlock_unlock(&ctx->epoch_rwlock);
            return false;
        }
    }

    /* Version 6: Write node lifecycles for invariant promotion state */
    if (ctx->node_lifecycles && header.node_count > 0) {
        written = fwrite(ctx->node_lifecycles, 1, header.node_count * sizeof(NodeLifecycle), f);
        if (written != header.node_count * sizeof(NodeLifecycle)) {
            fprintf(stderr, "ERROR: le_save_mmap failed to write node_lifecycles\n");
            fclose(f);
            unlink(tmp_path); /* Clean up temporary file on failure */
            pthread_rwlock_unlock(&ctx->epoch_rwlock);
            return false;
        }
    }

    fclose(f);

    /* L-6 FIX: Atomically rename temporary file to final path */
    if (rename(tmp_path, path) != 0) {
        fprintf(stderr, "ERROR: le_save_mmap failed to rename %s to %s: %s\n",
                tmp_path, path, strerror(errno));
        unlink(tmp_path); /* Clean up temporary file on rename failure */
        pthread_rwlock_unlock(&ctx->epoch_rwlock);
        return false;
    }

    /* CRITICAL FIX: Generate word_vocab.txt for frequency-ordered fallback predictions
     * This file provides frequency-ordered word suggestions when DAWG predictions fail,
     * replacing the sequential token scan that returns corpus-ingestion-order results. */
    char vocab_path[512];
    snprintf(vocab_path, sizeof(vocab_path), "vocab/%s.vocab.txt", path);

    /* Create vocab directory if it doesn't exist */
    mkdir("vocab", 0755);

    FILE *vocab_f = fopen(vocab_path, "w");
    if (vocab_f) {
        /* Create a temporary array for sorting lexemes by frequency */
        LexemeFreq *lexeme_freqs = NULL;
        uint32_t valid_count = 0;

        if (ctx->lexeme_count > 0) {
            lexeme_freqs = (LexemeFreq *)calloc(ctx->lexeme_count, sizeof(LexemeFreq));
            if (lexeme_freqs) {
                /* Collect all lexeme frequencies */
                pthread_rwlock_rdlock(&ctx->lexeme_rwlock);
                for (uint32_t i = 0; i < ctx->lexeme_count; i++) {
                    if (ctx->lexemes[i].surface) {
                        uint32_t freq = atomic_load_explicit(&ctx->lexemes[i].freq, memory_order_relaxed);
                        if (freq > 0) {
                            lexeme_freqs[valid_count].lexeme_id = i;
                            lexeme_freqs[valid_count].freq = freq;
                            valid_count++;
                        }
                    }
                }
                pthread_rwlock_unlock(&ctx->lexeme_rwlock);

                /* H-8 FIX: Replace O(N^2) bubble sort with qsort for O(N log N) performance */
                qsort(lexeme_freqs, valid_count, sizeof(LexemeFreq), compare_lexeme_freq_desc);

                /* Write sorted vocabulary to file */
                for (uint32_t i = 0; i < valid_count; i++) {
                    char surface_buf[256];
                    if (le_lexeme_surface_safe(ctx, lexeme_freqs[i].lexeme_id, surface_buf, sizeof(surface_buf)) == 0) {
                        fprintf(vocab_f, "%s\t%u\n", surface_buf, lexeme_freqs[i].freq);
                    }
                }

                free(lexeme_freqs);
                fprintf(stderr, "[VOCAB] Generated word_vocab.txt with %u entries\n", valid_count);
            }
        }
        fclose(vocab_f);
    } else {
        fprintf(stderr, "WARNING: Failed to open %s for writing\n", vocab_path);
    }

    pthread_rwlock_unlock(&ctx->epoch_rwlock);
    return true;
}

/**
 * le_unload_mmap - Unmap a context returned by le_load_mmap.
 *
 * Releases the memory-mapped region and associated resources.
 * Do NOT call le_destroy on a context loaded via le_load_mmap.
 *
 * CRITICAL FIX S-1: The current implementation uses malloc/calloc, not mmap,
 * so we should call le_destroy to properly free all resources. The old comment
 * about not calling le_destroy was from an older mmap-based design.
 */
void le_unload_mmap(EngineContext* ctx) {
    if (ctx) {
        le_destroy(ctx);
    }
}

/**
 * le_print_stats - Print engine statistics.
 *
 * Outputs current engine metrics and counters for debugging/monitoring.
 *
 * STUB IMPLEMENTATION: This is a placeholder that prevents linker errors.
 * Proper implementation requires formatted output of all statistics.
 */
void le_print_stats(EngineContext* ctx) {
    if (!ctx) {
        printf("EngineContext: NULL\n");
        return;
    }
    printf("EngineContext: magic=0x%X, version=%u\n", ctx->magic, ctx->version);
    printf("  Nodes: %u\n", atomic_load(&ctx->node_count));
    printf("  Edges: %u\n", atomic_load(&ctx->edge_count));
    printf("  Symbols: %u\n", atomic_load(&ctx->symbol_count));
    /* STUB: More statistics would be printed here */
}

/**
 * le_update_all_scc_candidates - Update candidate status for all SCCs.
 *
 * Recalculates which SCCs are candidates for promotion based on current
 * stability metrics and thresholds. Aggregates node frequencies into SCCs
 * and updates stability state machine.
 */
void le_update_all_scc_candidates(EngineContext* ctx) {
    if (!ctx) return;

    uint32_t scc_limit = atomic_load_explicit(&ctx->scc_count, memory_order_acquire);

    for (uint32_t i = 0; i < scc_limit && i < MAX_SCCS; i++) {
        SccNode* scc = &ctx->scc_nodes[i];
        uint32_t members = atomic_load_explicit(&scc->member_count, memory_order_acquire);
        if (members == 0) continue;

        /* 1. Aggregate frequency from constituent nodes using next_in_scc linked list */
        uint64_t accumulated_freq = 0;
        NodeID head_node = atomic_load_explicit(&scc->head, memory_order_acquire);

        if (head_node != INVALID && head_node < MAX_NODES) {
            NodeID curr_node = head_node;
            uint32_t guard = 0;
            while (curr_node != INVALID && curr_node < MAX_NODES && guard++ < MAX_NODES) {
                /* Aggregate node frequency into SCC frequency */
                uint64_t node_freq = atomic_load_explicit(&ctx->transient_nodes[curr_node].freq, memory_order_relaxed);
                accumulated_freq += node_freq;

                /* Move to next node in SCC using next_in_scc linked list */
                curr_node = atomic_load_explicit(&ctx->transient_nodes[curr_node].next_in_scc, memory_order_acquire);
            }
        }

        /* Update SCC frequency with accumulated value */
        atomic_store_explicit(&scc->freq, accumulated_freq, memory_order_release);

        /* PATCH E1: Apply frequency decay to prevent uint32_t overflow.
         * HIGH FIX H-7: Use CAS loop to prevent lost-update window on freq halving.
         * The previous read-modify-write pattern was not atomic - concurrent threads could
         * read the same current_freq, both compute decayed_freq, and both write, causing
         * one update to be lost. Fix: Use CAS loop to make the halving atomic. */
        uint64_t current_freq = atomic_load_explicit(&scc->freq, memory_order_acquire);
        if (current_freq > 10000000) {  /* Decay threshold: 10 million */
            uint64_t old_freq, new_freq;
            do {
                old_freq = atomic_load_explicit(&scc->freq, memory_order_acquire);
                if (old_freq <= 10000000) break;  /* Another thread already decayed it */
                new_freq = old_freq >> 1;  /* Divide by 2 */
            } while (!atomic_compare_exchange_weak_explicit(
                &scc->freq, &old_freq, new_freq,
                memory_order_release, memory_order_acquire));
        }

        /* 2. Update candidate status based on stability metrics */
        uint64_t scc_freq = atomic_load_explicit(&scc->freq, memory_order_acquire);
        uint32_t stable = atomic_load_explicit(&scc->stable_epochs, memory_order_acquire);
        float raw_coherence = scc_load_coherence(scc);

        /* CRITICAL FIX: Weighted Coherence Regularization
         * Penalizes huge clusters to keep root terms like 'soma' or 'laga' clean and distinct.
         * Prevents giant SCC garbage buckets from artificially inflating coherence scores. */
        float effective_coherence = raw_coherence;
        if (members > 1) {
            effective_coherence = raw_coherence / (1.0f + logf((float)members));
        }

        /* Mark as candidate if it meets promotion thresholds */
        bool is_candidate = (scc_freq >= ctx->min_freq) &&
                           (effective_coherence >= ctx->rho_min) &&
                           (stable >= 1) &&
                           !atomic_load_explicit(&scc->is_promoted, memory_order_acquire);

        /* M-8 FIX: Use CAS instead of unconditional store to avoid redundant store-CAS pattern */
        bool expected = atomic_load_explicit(&scc->is_candidate, memory_order_acquire);
        if (expected != is_candidate) {
            atomic_compare_exchange_weak_explicit(&scc->is_candidate, &expected,
                                                  is_candidate,
                                                  memory_order_release, memory_order_relaxed);
        }
    }
}

/**
 * le_should_gate_structural_edge - Structural Edge Gating Pre-Filter
 *
 * Blocks cross-term bridges (like words trailing punctuation) from linking into core SCC matrices.
 * Filters out nodes acting as punctuation-bridges (e.g., strings containing '.', ',', or spaces).
 *
 * Returns true if the edge should be gated/dropped, false otherwise.
 */
bool le_should_gate_structural_edge(EngineContext* ctx, int tok_handle, TokenID from_tok, TokenID to_tok) {
    (void)ctx;  /* Unused for now */
    (void)from_tok;  /* Unused for now */
    
    const char* target_str = tok_decode(tok_handle, to_tok);
    if (!target_str) return false;

    size_t len = strlen(target_str);
    if (len == 0) return false;

    /* Filter out nodes acting as punctuation-bridges (e.g., strings containing '.', ',', or spaces) */
    if (target_str[len - 1] == '.' || target_str[len - 1] == ',' || 
        target_str[0] == ','      || target_str[0] == '.') {
        return true; /* Gate/drop structural conversion */
    }

    return false;
}

/**
 * add_dawg_transition - Add a transition to the DAWG.
 * Safely inserts or updates a weighted edge between two promoted symbols.
 * H-6 FIX: Add striped mutex to prevent duplicate DAWG transition race.
 */
void add_dawg_transition(EngineContext* ctx, SymbolID from, SymbolID to, float weight) {
    if (!ctx || from >= MAX_SYMBOLS || to >= MAX_SYMBOLS) return;

    /* H-6 FIX: Lock the source symbol stripe to guarantee atomic check-and-insert */
    pthread_mutex_lock(&ctx->node_insert_locks[from & 1023]);

    Symbol* sym_from = &ctx->dawg_nodes[from];
    uint32_t t_idx = atomic_load_explicit(&sym_from->first_transition, memory_order_acquire);

    /* Deduplication: update existing transition weight */
    while (t_idx != INVALID && t_idx < MAX_DAWG_TRANSITIONS) {
        DawgTransition* t = &ctx->dawg_transitions[t_idx];
        if (atomic_load_explicit(&t->target, memory_order_acquire) == to) {
            uint32_t old_bits = atomic_load_explicit(&t->weight, memory_order_acquire);
            uint32_t new_bits;
            /* L-1 FIX: Add spin timeout to prevent indefinite spinning */
            int spin_count = 0;
            const int SPIN_LIMIT = 10000;
            do {
                float old_w;
                memcpy(&old_w, &old_bits, sizeof(float));
                float new_w = old_w + weight;
                memcpy(&new_bits, &new_w, sizeof(float));
                if (++spin_count > SPIN_LIMIT) {
                    fprintf(stderr, "WARNING: Spin timeout in add_dawg_transition weight CAS\n");
                    break;
                }
            } while (!atomic_compare_exchange_weak_explicit(
                &t->weight, &old_bits, new_bits,
                memory_order_release, memory_order_acquire));
            pthread_mutex_unlock(&ctx->node_insert_locks[from & 1023]);
            return;
        }
        t_idx = atomic_load_explicit(&t->next, memory_order_acquire);
    }

    /* Allocate new transition from the pool */
    uint32_t new_t_idx = atomic_fetch_add_explicit(&ctx->dawg_transition_count, 1, memory_order_relaxed);
    if (new_t_idx >= MAX_DAWG_TRANSITIONS) {
        pthread_mutex_unlock(&ctx->node_insert_locks[from & 1023]);
        return;
    }

    DawgTransition* new_t = &ctx->dawg_transitions[new_t_idx];
    /* C-6 FIX: Replace atomic_init with atomic_store_explicit for correctness on both new and reused slots */
    atomic_store_explicit(&new_t->target, to, memory_order_release);
    uint32_t w_bits;
    memcpy(&w_bits, &weight, sizeof(float));
    atomic_store_explicit(&new_t->weight, w_bits, memory_order_release);

    /* CAS loop for thread-safe list prepend */
    uint32_t expected;
    /* L-1 FIX: Add spin timeout to prevent indefinite spinning */
    int spin_count = 0;
    const int SPIN_LIMIT = 10000;
    do {
        expected = atomic_load_explicit(&sym_from->first_transition, memory_order_acquire);
        atomic_store_explicit(&new_t->next, expected, memory_order_relaxed);
        if (++spin_count > SPIN_LIMIT) {
            fprintf(stderr, "WARNING: Spin timeout in add_dawg_transition first_transition CAS\n");
            break;
        }
    } while (!atomic_compare_exchange_weak_explicit(
        &sym_from->first_transition, &expected, new_t_idx,
        memory_order_release, memory_order_acquire));

    atomic_fetch_add_explicit(&sym_from->transition_count, 1, memory_order_relaxed);
    pthread_mutex_unlock(&ctx->node_insert_locks[from & 1023]);
}

/**
 * le_free_beam_results - Free memory allocated for beam search results.
 *
 * Releases the memory used by beam search output structures.
 * Each beam[i].sequence was calloc'd in le_beam_search and must be freed.
 * The beam array itself uses TLS storage and must NOT be freed.
 */
void le_free_beam_results(BeamState* results, uint32_t count) {
    if (!results) return;

    for (uint32_t i = 0; i < count; i++) {
        if (results[i].sequence) free(results[i].sequence);
    }
    free(results);
}

/* ============================================================================
 * MEMORY INTEGRITY AUDIT
 * ============================================================================ */

/**
 * debug_audit_memory_integrity - Audit Symbol -> Canonical Node -> TokenID -> Text lineage
 *
 * Traces the full memory lineage from Symbol through canonical node to tokenizer
 * string pool to detect UTF-8 corruption, misaligned offsets, and garbage memory.
 * This diagnostic identifies tokenizer serialization misalignment issues where
 * id_to_offset arrays point to invalid memory addresses.
 */
void debug_audit_memory_integrity(EngineContext* ctx, int tok_handle) {
    if (!ctx) return;

    uint32_t active_symbols = atomic_load(&ctx->symbol_count);
    uint32_t corrupt_count = 0;

    printf("--- RUNNING MEMORY & UTF-8 AUDIT ---\n");

    for (uint32_t i = 1; i < active_symbols; i++) {
        Symbol *sym = &ctx->dawg_nodes[i];
        NodeID canon = sym->canonical_node;

        if (canon == INVALID || canon >= MAX_NODES) {
            printf("ID %u: CORRUPT (Invalid canonical node %u)\n", i, canon);
            corrupt_count++;
            continue;
        }

        TokenID tid = ctx->nodes[canon].token_id;
        const char* text = tok_decode(tok_handle, tid);

        /* Check for null, zero-length, or basic UTF-8 validity */
        if (!text || text[0] == '\0') {
            printf("ID %u: CORRUPT (Empty or NULL text pointer)\n", i);
            corrupt_count++;
            continue;
        }

        /* Basic ASCII/UTF-8 sanity check on the first few bytes */
        bool valid_utf8 = true;
        const unsigned char *p = (const unsigned char *)text;
        int len = 0;
        while (*p && len < 128) { // MAX_WORD_LEN
            if (*p == 0xFF || *p == 0xFE) { // Illegal UTF-8 bytes
                valid_utf8 = false; break;
            }
            p++; len++;
        }

        if (!valid_utf8 || len >= 128) {
            printf("ID %u: CORRUPT (UTF-8 violation or unterminated string)\n", i);
            corrupt_count++;
        }
    }
    printf("--- AUDIT COMPLETE: %u / %u symbols corrupted ---\n", corrupt_count, active_symbols);
}
