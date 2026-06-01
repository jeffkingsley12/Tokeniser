#include "gemini_internal.h"
#include <math.h>    /* isfinite — used by callers of SccNode float fields */
#include <stdio.h>   /* fprintf in warnings */

/**
 * Public accessors for EngineContext fields.
 * These provide a stable API for external consumers (like Python bindings).
 */

uint32_t get_node_count(EngineContext* ctx) { 
    if (!ctx) return 0;
    return atomic_load(&ctx->node_count); 
}
uint32_t get_edge_count(EngineContext* ctx) { 
    if (!ctx) return 0;
    return atomic_load(&ctx->edge_count); 
}
uint32_t get_scc_count(EngineContext* ctx) { 
    if (!ctx) return 0;
    /* FIX NOTE: This function now returns the count of *active* SCCs
     * (member_count > 0) rather than the raw atomic scc_count.
     *
     * PERFORMANCE: This is O(scc_count), not O(1). Call only at epoch
     * boundaries or diagnostics, never in hot paths.
     *
     * THREAD SAFETY: member_count is a plain uint32_t (not _Atomic). Reading
     * it concurrently with an SCC merge that writes member_count is a data
     * race. This accessor must only be called when the engine is idle
     * (between epochs), or member_count must be promoted to _Atomic uint32_t. */
    uint32_t active = 0;
    uint32_t limit = atomic_load(&ctx->scc_count);
    for (uint32_t i = 0; i < limit && i < MAX_SCCS; i++) {
        /* H-4 FIX: member_count is _Atomic uint32_t; use atomic_load_explicit
         * rather than relying on implicit seq-cst access, which has full-barrier
         * cost on ARM/POWER and obscures ordering intent. */
        if (atomic_load_explicit(&ctx->scc_nodes[i].member_count, memory_order_acquire) > 0) {
            active++;
        }
    }
    return active; 
}
uint32_t get_symbol_count(EngineContext* ctx) { 
    if (!ctx) return 0;
    return atomic_load(&ctx->symbol_count); 
}
uint32_t get_token_count(EngineContext* ctx) { 
    if (!ctx) return 0;
    return atomic_load(&ctx->token_count); 
}
uint32_t get_current_epoch(EngineContext* ctx) { 
    if (!ctx) return 0;
    return atomic_load(&ctx->current_epoch);  /* CRITICAL FIX: Use atomic_load to prevent data race (Issue #8) */
}
uint64_t get_current_step(EngineContext* ctx) { 
    if (!ctx) return 0;
    return atomic_load(&ctx->current_step); 
}
uint64_t get_total_merges(EngineContext* ctx) { 
    if (!ctx) return 0;
    return atomic_load(&ctx->total_merges);  /* CRITICAL FIX: Now uint64_t storage prevents wrap (Issue #9) */
}
uint64_t get_total_promotions(EngineContext* ctx) { 
    if (!ctx) return 0;
    return atomic_load(&ctx->total_promotions);  /* CRITICAL FIX: Now uint64_t storage prevents wrap (Issue #9) */
}

/* Turbulence & entropy accessors are implemented in gemini_engine.c */

/* ── SCC forced-promotion flag ──────────────────────────────────────────────── */

/*
 * CRITICAL FIX (Issue #10): SccNode fields (is_forced, is_candidate, member_count, head)
 * are accessed here without synchronization. Per C11 semantics, any concurrent write
 * by an engine path during SCC restructuring is undefined behavior.
 * 
 * SYNCHRONIZATION REQUIREMENTS:
 * - These accessors MUST be called only at epoch boundaries when the engine is idle,
 *   OR the SccNode fields must be made _Atomic and accessed with atomic_load/store.
 * - Current code assumes single-reader, engine-thread-writer pattern.
 * - Document this assumption clearly or refactor to use atomics for fields that
 *   are read from user threads while engine runs.
 */

void le_set_scc_forced(EngineContext *ctx, SccID scc_id, bool forced) {
    if (!ctx || scc_id >= MAX_SCCS) return;
    atomic_store_explicit(&ctx->scc_nodes[scc_id].is_forced, forced, memory_order_release);
}

bool le_scc_is_candidate(EngineContext *ctx, SccID scc_id) {
    if (!ctx || scc_id >= MAX_SCCS) return false;
    return atomic_load_explicit(&ctx->scc_nodes[scc_id].is_candidate, memory_order_acquire);
}

void le_set_scc_candidate(EngineContext *ctx, SccID scc_id, bool is_candidate) {
    if (!ctx || scc_id >= MAX_SCCS) return;
    atomic_store_explicit(&ctx->scc_nodes[scc_id].is_candidate, is_candidate, memory_order_release);
}

SccID le_node_scc(EngineContext *ctx, NodeID node_id) {
    if (!ctx || node_id >= MAX_NODES) return (SccID)INVALID;
    /* FIX (Issue #3): scc_id is now _Atomic uint32_t; atomic_load_explicit is correct. */
    return atomic_load_explicit(&ctx->transient_nodes[node_id].scc_id, memory_order_acquire);
}

uint32_t le_scc_member_count(EngineContext *ctx, SccID scc_id) {
    if (!ctx || scc_id >= MAX_SCCS) return 0;
    return atomic_load_explicit(&ctx->scc_nodes[scc_id].member_count, memory_order_acquire);
}

NodeID le_scc_head(EngineContext *ctx, SccID scc_id) {
    if (!ctx || scc_id >= MAX_SCCS) return (NodeID)INVALID;
    /* FIX (Issue #2): head is now _Atomic NodeID. */
    return atomic_load_explicit(&ctx->scc_nodes[scc_id].head, memory_order_acquire);
}

float le_scc_mean_turbulence(EngineContext *ctx, SccID scc_id) {
    if (!ctx || scc_id >= MAX_SCCS) return 0.0f;
    
    /* CRITICAL FIX: Drop const to avoid casting violations later */
    SccNode *scc = &ctx->scc_nodes[scc_id];
    uint32_t member_count = atomic_load_explicit(&scc->member_count, memory_order_acquire);
    if (member_count == 0) return 0.0f;

    double sum = 0.0;
    uint32_t count = 0;
    NodeID cur = atomic_load_explicit(&scc->head, memory_order_acquire);

    while (cur != INVALID && count < member_count) {
        /* FIX (Issue #16): Guard cur against MAX_NODES before indexing nodes[].
         * A corrupted or mid-merge SCC linked list can produce an out-of-bounds
         * cur value; cap it here rather than crashing or reading garbage. */
        if (cur >= MAX_NODES) {
            fprintf(stderr, "WARNING: SCC %u linked list contains out-of-bounds node %u "
                    "(count=%u, member_count=%u)\n", scc_id, cur, count, member_count);
            break;
        }
        
        /* CRITICAL FIX: Safely extract float from atomic bit pattern using memcpy
         * This avoids torn reads on IEEE-754 bit pattern during concurrent mutations */
        uint32_t turb_bits = atomic_load_explicit(&ctx->nodes[cur].turbulence, memory_order_relaxed);
        float turb_val;
        memcpy(&turb_val, &turb_bits, sizeof(float));
        sum += (double)turb_val;
        
        /* FIX (Issue #3): next_in_scc is now _Atomic uint32_t. */
        cur = atomic_load_explicit(&ctx->transient_nodes[cur].next_in_scc, memory_order_acquire);
        count++;
    }

    if (count != member_count) {
        fprintf(stderr, "WARNING: SCC %u linked list mismatch: counted %u but member_count=%u\n",
                scc_id, count, member_count);
    }

    return (count > 0) ? (float)(sum / (double)count) : 0.0f;
}

/* ── Closed-Loop Feedback Ingestion Hooks ───────────────────────────────────── */

/**
 * engine_get_scc_stability - Compute global graph stability.
 * THREAD SAFETY WARNING: 
 * This is an O(scc_count) operation. It performs lock-free acquire loads 
 * across the SCC array. If called concurrently with active ingestion or 
 * epoch transitions, the returned average is "fuzzy" (eventually consistent).
 * For deterministic analytics (e.g., via Python bindings), this MUST be called 
 * while the engine ingestion pipeline is paused or externally read-locked.
 */
float engine_get_scc_stability(EngineContext* ctx) {
    if (!ctx) return 0.0f;
    uint32_t scc_count = atomic_load(&ctx->scc_count);
    if (scc_count == 0) return 0.0f;
    double total_coherence = 0.0;
    uint32_t active_sccs = 0;
    for (uint32_t i = 0; i < scc_count && i < MAX_SCCS; i++) {
        if (atomic_load_explicit(&ctx->scc_nodes[i].member_count, memory_order_acquire) > 0) {
            /* FIX (Issue #2): coherence is now _Atomic uint32_t (bit-cast).
             * Use scc_load_coherence() which issues an acquire load, correctly
             * pairing with the release store in the engine's metric update path. */
            total_coherence += (double)scc_load_coherence(&ctx->scc_nodes[i]);
            active_sccs++;
        }
    }
    return (active_sccs > 0) ? (float)(total_coherence / active_sccs) : 0.0f;
}

float engine_get_entropy_delta(EngineContext* ctx) {
    if (!ctx) return 0.0f;
    float curr = atomic_load_float(&ctx->global_entropy_bits);
    float prev = atomic_load_float(&ctx->prev_global_entropy_bits);
    return curr - prev;
}

uint32_t engine_get_active_region_count(EngineContext* ctx) {
    if (!ctx) return 0;
    uint32_t scc_count = atomic_load(&ctx->scc_count);
    uint32_t active_count = 0;
    for (uint32_t i = 0; i < scc_count && i < MAX_SCCS; i++) {
        SccNode *s = &ctx->scc_nodes[i];
        if (atomic_load_explicit(&s->member_count, memory_order_acquire) > 0
            && !atomic_load_explicit(&s->is_promoted,  memory_order_acquire)
            &&  atomic_load_explicit(&s->is_candidate, memory_order_acquire)) {
            active_count++;
        }
    }
    return active_count;
}

void engine_set_ingestion_weight_modifier(EngineContext* ctx, float modifier) {
    if (!ctx) return;
    if (modifier < 0.0f) modifier = 0.0f;
    atomic_store_float(&ctx->ingestion_weight_modifier, modifier);
}

/* FIX (Issue #17): Update pool_utilization high-water mark after each node token query.
 * Called here because get_node_token is the canonical accessor path used by
 * dawg_predict and the beam search decouple step. */
TokenID get_node_token(EngineContext *ctx, NodeID node) {
    if (!ctx) return INVALID;
    uint32_t total_nodes = atomic_load_explicit(&ctx->node_count, memory_order_acquire);
    if (node == INVALID || node >= total_nodes) return INVALID;

    /* Track peak edge pool utilisation for monitoring (Issue #17 fix). */
    uint32_t pool_ptr = atomic_load_explicit(&ctx->edge_pool.pool_ptr, memory_order_relaxed);
    uint32_t prev_peak = atomic_load_explicit(&ctx->edge_pool.pool_utilization, memory_order_relaxed);
    while (pool_ptr > prev_peak) {
        if (atomic_compare_exchange_weak_explicit(
                &ctx->edge_pool.pool_utilization, &prev_peak, pool_ptr,
                memory_order_relaxed, memory_order_relaxed))
            break;
    }

    return ctx->nodes[node].token_id;
}

/**
 * le_get_symbol_nodes - Returns all NodeIDs belonging to a promoted symbol.
 * Fills out_buf up to max_nodes, returning the total member count.
 */
uint32_t le_get_symbol_nodes(EngineContext *ctx, SymbolID sym, NodeID *out_buf, uint32_t max_nodes) {
    if (!ctx || !out_buf || max_nodes == 0) return 0;

    uint32_t symbol_count = atomic_load_explicit(&ctx->symbol_count, memory_order_acquire);
    if (sym >= symbol_count || sym >= MAX_SYMBOLS) return 0;

    /* Get the Symbol struct */
    Symbol* symbol = &ctx->dawg_nodes[sym];

    /* CRITICAL FIX: Use canonical_node instead of original_scc for stable identity
     * SCC slots are ephemeral across Kosaraju runs, but the canonical node
     * provides stable identity. Look up the current SCC for this node. */
    NodeID canonical_node = symbol->canonical_node;
    if (canonical_node == INVALID || canonical_node >= MAX_NODES) return 0;

    /* Get the current SCC for this canonical node */
    SccID current_scc = atomic_load_explicit(&ctx->transient_nodes[canonical_node].scc_id, memory_order_acquire);
    if (current_scc >= MAX_SCCS) return 0;

    /* Get the SCC struct */
    SccNode* scc = &ctx->scc_nodes[current_scc];
    uint32_t member_count = atomic_load_explicit(&scc->member_count, memory_order_acquire);

    if (member_count == 0) return 0;

    /* Traverse the SCC's member list */
    uint32_t count = 0;
    NodeID node = atomic_load_explicit(&scc->head, memory_order_acquire);
    uint32_t guard = 0;

    while (node != INVALID && guard++ < member_count && count < max_nodes) {
        if (node >= MAX_NODES) break;
        out_buf[count++] = node;
        node = atomic_load_explicit(&ctx->transient_nodes[node].next_in_scc, memory_order_acquire);
    }

    return count;
}