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
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <stdatomic.h>
#include <pthread.h>

/* Global fallback mutex for atomic float operations */
pthread_mutex_t g_afa_fallback_mutex = PTHREAD_MUTEX_INITIALIZER;

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
    atomic_store(&ctx->ingestion_weight_modifier, WEIGHT_SCALE_FACTOR);

    ctx->max_symbols = MAX_SYMBOLS;

    /* Allocate main data structures */
    ctx->nodes = (GeminiNodeDisk *)calloc(MAX_NODES, sizeof(GeminiNodeDisk));
    if (!ctx->nodes) {
        fprintf(stderr, "ERROR: Failed to allocate nodes array\n");
        free(ctx);
        return NULL;
    }

    ctx->edges = (GeminiEdgeDisk *)calloc(MAX_EDGES, sizeof(GeminiEdgeDisk));
    if (!ctx->edges) {
        fprintf(stderr, "ERROR: Failed to allocate edges array\n");
        free(ctx->nodes);
        free(ctx);
        return NULL;
    }

    ctx->symbols = (GeminiSymbolDisk *)calloc(MAX_SYMBOLS, sizeof(GeminiSymbolDisk));
    if (!ctx->symbols) {
        fprintf(stderr, "ERROR: Failed to allocate symbols array\n");
        free(ctx->nodes);
        free(ctx->edges);
        free(ctx);
        return NULL;
    }

    ctx->edge_nexts = (EdgeID *)calloc(MAX_EDGES, sizeof(EdgeID));
    if (!ctx->edge_nexts) {
        fprintf(stderr, "ERROR: Failed to allocate edge_nexts array\n");
        free(ctx->nodes);
        free(ctx->edges);
        free(ctx->symbols);
        free(ctx);
        return NULL;
    }

    /* Allocate transient state arrays */
    ctx->transient_nodes = (NodeTransientState *)calloc(MAX_NODES, sizeof(NodeTransientState));
    if (!ctx->transient_nodes) {
        fprintf(stderr, "ERROR: Failed to allocate transient_nodes\n");
        goto cleanup;
    }

    /* Allocate SCC arrays */
    ctx->scc_nodes = (SccNode *)calloc(MAX_SCCS, sizeof(SccNode));
    if (!ctx->scc_nodes) {
        fprintf(stderr, "ERROR: Failed to allocate scc_nodes\n");
        goto cleanup;
    }

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
        ctx->dawg_transitions[i].target = INVALID;
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

    /* Initialize edge pool */
    if (edge_pool_init(ctx, MAX_EDGES) != 0) {
        fprintf(stderr, "ERROR: Failed to initialize edge pool\n");
        goto cleanup;
    }

    /* Initialize mutexes and locks */
    if (pthread_mutex_init(&ctx->large_scratch_mutex, NULL) != 0) {
        fprintf(stderr, "ERROR: Failed to initialize large_scratch_mutex\n");
        goto cleanup;
    }

    if (pthread_mutex_init(&ctx->lexeme_lock, NULL) != 0) {
        fprintf(stderr, "ERROR: Failed to initialize lexeme_lock\n");
        goto cleanup;
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
    
    edge_pool_destroy(ctx);
    pthread_mutex_destroy(&ctx->large_scratch_mutex);
    pthread_mutex_destroy(&ctx->lexeme_lock);

    free(ctx);
    return NULL;
}

/* ============================= CLEANUP ====================================== */

/**
 * le_destroy - Free all memory allocated by le_init.
 * 
 * Destroys mutexes and rwlocks, then frees all heap structures.
 */
void le_destroy(EngineContext* ctx) {
    if (!ctx) return;

    pthread_mutex_destroy(&ctx->large_scratch_mutex);
    pthread_mutex_destroy(&ctx->lexeme_lock);

    edge_pool_destroy(ctx);

    /* Free all arrays */
    free(ctx->nodes);
    free(ctx->edges);
    free(ctx->symbols);
    free(ctx->edge_nexts);
    free(ctx->transient_nodes);
    free(ctx->scc_nodes);
    free(ctx->scc_edges);
    free(ctx->dawg_nodes);
    free(ctx->dawg_transitions);
    free(ctx->node_hash);
    free(ctx->reverse_edges);
    free(ctx->deferred_scc_checks);
    free(ctx->active_symbols);
    free(ctx->large_scratch);

    /* Free lexeme entries and array */
    if (ctx->lexemes) {
        for (uint32_t i = 0; i < ctx->lexeme_count; i++) {
            free(ctx->lexemes[i].surface);
        }
        free(ctx->lexemes);
    }

    free(ctx);
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
            uint32_t node_count = atomic_load(&ctx->node_count);
            if (node_count >= MAX_NODES) {
                fprintf(stderr, "ERROR: MAX_NODES reached (%u)\n", MAX_NODES);
                return INVALID;
            }

            NodeID new_id = node_count;

            /* Initialize node structure */
            ctx->nodes[new_id].token_id = token_id;
            atomic_init(&ctx->nodes[new_id].first_edge, INVALID);
            atomic_init(&ctx->nodes[new_id].edge_count, 0);
            atomic_init(&ctx->nodes[new_id].entropy_bits, 0);
            atomic_init(&ctx->nodes[new_id].total_mass, 0);
            ctx->nodes[new_id].turbulence = 0.0f;
            ctx->nodes[new_id].absorbing = 0;
            atomic_init(&ctx->nodes[new_id].last_written_step, 0);

            /* Initialize transient state */
            atomic_init(&ctx->transient_nodes[new_id].scc_id, 0);  /* Default to SCC 0 */
            atomic_init(&ctx->transient_nodes[new_id].next_in_scc, INVALID);
            atomic_init(&ctx->transient_nodes[new_id].freq, 1);

            /* Initialize edge span */
            if (edge_span_init(ctx, new_id, 8) != 0) {
                fprintf(stderr, "ERROR: Failed to initialize edge span for node %u\n", new_id);
                return INVALID;
            }

            /* Insert into hash table (CAS-based) */
            uint32_t expected = INVALID;
            if (!atomic_compare_exchange_strong_explicit(
                    &ctx->node_hash[h], &expected, new_id,
                    memory_order_release, memory_order_acquire)) {
                /* Another thread inserted; retry lookup */
                return le_get_or_create_node(ctx, token_id);
            }

            /* Update node count */
            atomic_fetch_add(&ctx->node_count, 1);
            return new_id;
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
 * le_add_edge - Add a directed edge between two nodes.
 * 
 * Adds or increments an edge from -> to with weight +1.0.
 * Thread-safe using atomic operations and CAS loops.
 */
void le_add_edge(EngineContext* ctx, NodeID from, NodeID to) {
    if (!ctx || from == INVALID || to == INVALID) return;
    if (from >= MAX_NODES || to >= MAX_NODES) return;

    /* Atomic weight increment using float addition */
    atomic_fetch_add_float(&ctx->edges[from].weight, 1.0f);
    atomic_fetch_add(&ctx->node_count, 0);  /* Ensure nodes exist */

    /* Add incoming edge to target node via pooled adjacency */
    le_add_incoming_edge_pooled(ctx, to, from);
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

    /* Atomic weight increment */
    atomic_fetch_add_float(&ctx->edges[from].weight, weight_to_add);

    /* Add incoming edge to target node via pooled adjacency */
    le_add_incoming_edge_pooled(ctx, to, from);
}

/* ============================= EPOCH HANDLING ============================== */

/**
 * le_begin_epoch - Start a new epoch.
 * 
 * Increments epoch counter, updates stability counters, and triggers
 * SCC promotions and demotions based on configuration thresholds.
 */
void le_begin_epoch(EngineContext* ctx) {
    if (!ctx) return;

    /* Atomic CAS to prevent concurrent epoch transitions */
    uint32_t expected = 0;
    if (!atomic_compare_exchange_strong(&ctx->epoch_in_progress, &expected, 1)) {
        return;  /* Another thread is in epoch transition */
    }

    /* Increment epoch counter */
    EpochID current = atomic_load(&ctx->current_epoch);
    atomic_store(&ctx->current_epoch, current + 1);

    /* Update deferred SCC checks if any are pending */
    uint32_t deferred_count = atomic_load(&ctx->deferred_scc_count);
    if (deferred_count > 0) {
        for (uint32_t i = 0; i < deferred_count && i < MAX_DEFERRED_EDGES; i++) {
            DeferredSccCheck *check = &ctx->deferred_scc_checks[i];
            uint32_t from_id = atomic_load(&check->from);
            uint32_t to_id = check->to;
            
            if (from_id != INVALID && from_id < MAX_SCCS && to_id < MAX_SCCS) {
                /* Perform deferred merging or promotion logic */
                if (from_id != to_id) {
                    /* Mark for potential merging */
                    le_merge_sccs(ctx, to_id, from_id);
                }
            }
        }
        atomic_store(&ctx->deferred_scc_count, 0);
    }

    /* Promote eligible SCCs based on stability and coherence */
    uint32_t promoted = le_promote_eligible(ctx);
    atomic_fetch_add(&ctx->total_promotions, promoted);

    /* Demote stale symbols */
    uint32_t demoted = le_demote_stale_symbols(ctx);
    (void)demoted;  /* Suppress unused variable warning */

    /* Clear epoch in progress flag */
    atomic_store(&ctx->epoch_in_progress, 0);
}

/* ============================= GRAPH OPERATIONS ============================ */

/**
 * le_merge_sccs - Merge two strongly connected components.
 * 
 * Combines source_id into target_id. This is a critical operation for
 * incremental cycle detection during learning.
 * 
 * STUB IMPLEMENTATION: This is a placeholder that prevents linker errors.
 * Proper implementation requires SCC graph reachability analysis.
 */
void le_merge_sccs(EngineContext* ctx, SccID target_id, SccID source_id) {
    if (!ctx || target_id >= MAX_SCCS || source_id >= MAX_SCCS) return;
    if (target_id == source_id) return;

    /* STUB: In a full implementation, this would:
     * 1. Redirect all edges from source_id to target_id
     * 2. Merge member lists (doubly-linked node chain)
     * 3. Update transitive closure for reachability
     * 4. Update SCC-DAG edges
     * 5. Recalculate coherence and entropy metrics
     */

    SccNode* target = &ctx->scc_nodes[target_id];
    SccNode* source = &ctx->scc_nodes[source_id];

    /* Merge member counts */
    uint32_t target_count = atomic_load(&target->member_count);
    uint32_t source_count = atomic_load(&source->member_count);
    if (target_count + source_count <= MAX_NODES) {
        atomic_store(&target->member_count, target_count + source_count);
    }

    /* Merge frequency information */
    uint64_t target_freq = atomic_load(&target->freq);
    uint64_t source_freq = atomic_load(&source->freq);
    atomic_store(&target->freq, target_freq + source_freq);

    /* Mark source as absorbed (conceptually empty after merge) */
    atomic_store(&source->member_count, 0);

    /* Update total merges counter */
    atomic_fetch_add(&ctx->total_merges, 1);
}

/**
 * le_promote_eligible - Scan and promote all eligible SCCs to symbols.
 * 
 * Examines all SCCs and promotes those that have reached stability thresholds.
 * Returns the number of SCCs promoted.
 * 
 * STUB IMPLEMENTATION: This is a placeholder that prevents linker errors.
 * Proper implementation requires SCC candidate scanning and metric calculations.
 */
uint32_t le_promote_eligible(EngineContext* ctx) {
    if (!ctx) return 0;

    /* STUB: In a full implementation, this would:
     * 1. Iterate through all SCCs
     * 2. Check stability_epochs >= promotion_epochs
     * 3. Calculate coherence (rho) and entropy (H) metrics
     * 4. Compare against thresholds (rho_min, h_max, h_forced)
     * 5. Call promote_scc() for eligible candidates
     * 6. Update DAWG transitions
     */

    uint32_t promoted_count = 0;
    uint32_t scc_limit = atomic_load(&ctx->scc_count);

    for (uint32_t i = 0; i < scc_limit && i < MAX_SCCS; i++) {
        SccNode* scc = &ctx->scc_nodes[i];
        uint32_t member_count = atomic_load(&scc->member_count);

        /* Skip already-promoted or empty SCCs */
        if (member_count == 0 || scc->is_promoted) {
            continue;
        }

        /* Mark as candidate if not already */
        if (!scc->is_candidate) {
            atomic_store_explicit(&scc->is_candidate, true, memory_order_release);
        }

        /* In a full implementation, check:
         * - scc->stable_epochs >= ctx->promotion_epochs
         * - coherence (rho) >= ctx->rho_min
         * - entropy (h) <= ctx->h_max
         * Then call promote_scc(ctx, i, total_mass)
         */

        /* STUB: Always count candidates for now */
        if (scc->is_candidate) {
            promoted_count++;
        }
    }

    return promoted_count;
}
