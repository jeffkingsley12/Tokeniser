#ifndef BRIDGE_ENGINE_H
#define BRIDGE_ENGINE_H

#include "gemini_internal.h"
#include "tokenizer_api.h"

/* BRIDGE_TOKEN_BUFFER_SIZE removed: all bridge functions now use dynamic
 * allocation sized to strlen(text)+1, not a fixed 256-slot buffer. */

/*
 * L-4 FIX: Named constant for the per-prediction string column width used in
 * dawg_predict / dawg_predict_multi_hop output arrays.
 * Callers MUST allocate at least this many bytes per column:
 *   char predictions[N][DAWG_PREDICTION_MAX_LEN];
 * Passing a smaller column silently causes stack/heap corruption.
 */
#define DAWG_PREDICTION_MAX_LEN 256

typedef struct {
    SymbolID input_symbol;
    SccID    from_scc;
    SccID    to_scc;
    NodeID   target_node;
} LineageEntry;

typedef struct {
    LineageEntry *entries;
    uint32_t      count;
    uint32_t      capacity;
} TransitionLineage;

NodeID engine_ingest_text(int tok_handle, EngineContext *ctx, 
                          const char *text, NodeID start_node);

NodeID engine_infer_text(int tok_handle, EngineContext *ctx, 
                         const char *text, NodeID start_node);

NodeID engine_infer_text_direct_id(EngineContext *ctx, NodeID start_node, 
                                  SymbolID sym, SccID *out_from, SccID *out_to);

TransitionLineage* engine_ingest_text_with_lineage(int tok_handle, EngineContext *ctx,
                                                   const char *text, NodeID start_node);
int dawg_predict(int tok_handle, EngineContext *ctx, NodeID current_node,
                 float tau, uint32_t max_predictions,
                 char predictions[][DAWG_PREDICTION_MAX_LEN], float probabilities[]);
int dawg_predict_multi_hop(int tok_handle, EngineContext *ctx, NodeID current_node,
                          float tau, uint32_t max_predictions, uint32_t max_depth,
                          char predictions[][DAWG_PREDICTION_MAX_LEN], float probabilities[]);
int engine_infer_distributed(int tok_handle, EngineContext *ctx, 
                             const char *text, NodeID start_node,
                             ActiveState *out_states, uint32_t max_states);

void engine_lineage_free(TransitionLineage *lineage);
void engine_lineage_print(const TransitionLineage *lineage);

#endif /* BRIDGE_ENGINE_H */
