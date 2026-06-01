/*
 * legacy_compat.c - Legacy API compatibility shim for le_process_token
 *
 * This file provides a compatibility shim that translates legacy le_process_token
 * calls to the new enhanced ingestion pipeline. This addresses API drift while
 * maintaining backward compatibility for existing code.
 *
 * DEPRECATED: le_process_token is deprecated. Migrate to gemini_enhanced_process_word
 * or gemini_enhanced_process_text in the enhanced pipeline.
 */


#include "libgemini.h"
#include "gemini_internal.h"
#include "lexeme_intern.h"
#include <stdio.h>

NodeID le_process_token(EngineContext* ctx, TokenID token_id, NodeID prev_node) {
    /* LOW FIX M-4: Return INVALID instead of prev_node when ctx is NULL.
     * Returning prev_node could propagate an invalid state, whereas INVALID
     * clearly indicates an error condition. */
    if (!ctx) return INVALID;

    /* FIX C1: MANDATORY - Intern the canonical surface form to populate lexeme registry */
    /* Note: This requires a tok_handle which is not available in this context.
     * For now, we skip lexeme interning here since the bridge_engine.c path
     * already handles it via tok_decode. If this path is used, the caller
     * should provide the surface form separately. */

    /* 1. Allocate or retrieve the representative node in the base graph */
    NodeID current_node = le_get_or_create_node(ctx, token_id);
    if (current_node == INVALID) {
        return INVALID;
    }

    /* 2. Establish directed adjacency if a valid predecessor exists */
    if (prev_node != INVALID) {
        uint32_t current_node_max = atomic_load_explicit(&ctx->node_count, memory_order_acquire);
        if (prev_node < current_node_max) {
            /* FIX C2: Apply structural gating to prevent hairball formation.
             * The pre-SCC edge gating in le_add_edge (Patch B1) already handles
             * this by checking SCC stability and frequency before accepting
             * new edges. No additional gating needed here. */
            le_add_edge(ctx, prev_node, current_node);
        } else {
            fprintf(stderr, "gemini_compat: WARNING — prev_node (%u) out of bounds (%u)\n",
                    prev_node, current_node_max);
        }
    }

    return current_node;
}
