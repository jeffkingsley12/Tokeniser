#include "../include/tokenizer_api.h"
#include "gemini_internal.h"
#include "bridge_engine.h"
#include "../include/libgemini.h"
#include "lexeme_intern.h"
#include <limits.h>
#include <stdatomic.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#if defined(__x86_64__) || defined(__i386__)
#include <immintrin.h>
#else
#include <sched.h>
#endif

#define MAX_TOKEN_LIMIT 1000000 
#define MAX_ACTIVE_STATES 32 


/* ===============================================================================
 * INTERNAL SEARCH STRUCTURES
 * Bounded state descriptors used for local path tracking during inference.
 * Bounding paths inline avoids heap fragmentation in hot execution paths.
 * =============================================================================== */

typedef struct {
    SymbolID current;                  /* Current symbol node identifier */
    float    score;                    /* Accumulated log-probability score */
    uint32_t length;                   /* Current length of the generated path */
    SymbolID path[MAX_SEARCH_DEPTH];   /* Inline token sequence path buffer */
} PredictState;

typedef struct {
    SymbolID current;                  /* Current symbol node identifier */
    float    score;                    /* Accumulated log-probability score */
    uint32_t length;                   /* Current length of the generated path */
    float    entropy;                  /* Cumulative structural entropy calculation */
    SymbolID path[MAX_SEARCH_DEPTH];   /* Inline token sequence path buffer */
} MultiHopState; 


static int compare_predict_states(const void *a, const void *b) {
    const PredictState *sa = (const PredictState *)a;
    const PredictState *sb = (const PredictState *)b;
    if (sa->score < sb->score) return 1;
    if (sa->score > sb->score) return -1;
    return 0;
}

static int compare_multihop_states(const void *a, const void *b) {
    const MultiHopState *sa = (const MultiHopState *)a;
    const MultiHopState *sb = (const MultiHopState *)b;
    if (sa->score < sb->score) return 1;
    if (sa->score > sb->score) return -1;
    return 0;
}

static NodeID find_node_by_symbol(EngineContext *ctx, SymbolID sym) {
    uint32_t h = (sym * 0x9E3779B1u) % HASH_SIZE;
    uint32_t steps = 0;
    NodeID slot;
    
    while (steps < HASH_SIZE) {
        slot = atomic_load_explicit(&ctx->node_hash[h], memory_order_acquire);
        if (slot == INVALID) {
            return INVALID;
        }
        /* DAWG FIX D-2: Spin on CLAIMED sentinel.
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
            continue; /* Spin until the writing thread commits the NodeID */
        }
        if (slot >= MAX_NODES) {
            h = (h + 1) % HASH_SIZE;
            steps++;
            continue;
        }
        if (ctx->nodes[slot].token_id == sym) {
            return slot;
        }
        h = (h + 1) % HASH_SIZE;
        steps++;
    }
    return INVALID; 
}

static NodeID engine_step(EngineContext *ctx, NodeID current_node, SymbolID input_sym, 
                          SccID *out_from_scc, SccID *out_to_scc) {
    if (!ctx || !ctx->transient_nodes || !ctx->scc_nodes || !ctx->dawg_nodes || !ctx->dawg_transitions) {
        return INVALID;
    }

    if (out_from_scc) *out_from_scc = INVALID;
    if (out_to_scc)   *out_to_scc   = INVALID;

    if (current_node == INVALID) {
        NodeID input_node = find_node_by_symbol(ctx, input_sym);
        if (input_node == INVALID) return INVALID;

        SccID input_scc = atomic_load_explicit(&ctx->transient_nodes[input_node].scc_id, memory_order_acquire);
        if (input_scc >= MAX_SCCS) return INVALID;
        if (!atomic_load_explicit(&ctx->scc_nodes[input_scc].is_promoted, memory_order_acquire))
            return INVALID;

        if (out_to_scc) *out_to_scc = input_scc;
        return atomic_load_explicit(&ctx->scc_nodes[input_scc].head, memory_order_acquire);
    }

    /* Defensive active-bounds check to prevent concurrent OOB reads */
    uint32_t active_nodes = atomic_load_explicit(&ctx->node_count, memory_order_acquire);
    if (current_node >= active_nodes || current_node >= MAX_NODES) return INVALID;

    NodeID input_node = find_node_by_symbol(ctx, input_sym);
    if (input_node == INVALID) return INVALID;

    SccID input_scc = atomic_load_explicit(&ctx->transient_nodes[input_node].scc_id, memory_order_acquire);
    if (input_scc >= MAX_SCCS) return INVALID;
    SccNode *input_scc_node = &ctx->scc_nodes[input_scc];
    
    if (!atomic_load_explicit(&input_scc_node->is_promoted, memory_order_acquire))
        return INVALID;
    SymbolID input_dawg_sym = atomic_load_explicit(&input_scc_node->symbol_id, memory_order_acquire);

    SccID current_scc = atomic_load_explicit(&ctx->transient_nodes[current_node].scc_id, memory_order_acquire);
    if (current_scc >= MAX_SCCS) return INVALID;
    
    if (out_from_scc) *out_from_scc = current_scc;
    
    SccNode *scc = &ctx->scc_nodes[current_scc];
    if (!atomic_load_explicit(&scc->is_promoted, memory_order_acquire))
        return INVALID;
    
    SymbolID current_symbol = atomic_load_explicit(&scc->symbol_id, memory_order_acquire);
    uint32_t active_symbols = atomic_load_explicit(&ctx->symbol_count, memory_order_acquire);
    if (current_symbol >= active_symbols || current_symbol >= MAX_SYMBOLS) return INVALID;
    
    Symbol *sym_obj = &ctx->dawg_nodes[current_symbol];
    uint32_t t_idx = atomic_load_explicit(&sym_obj->first_transition, memory_order_acquire);
    
    while (t_idx != INVALID) {
        /* Defensively guard global transition allocation boundary */
        if (t_idx >= atomic_load_explicit(&ctx->dawg_transition_count, memory_order_acquire) || t_idx >= MAX_DAWG_TRANSITIONS) break;

        DawgTransition *t = &ctx->dawg_transitions[t_idx];
        SymbolID t_target = atomic_load_explicit(&t->target, memory_order_acquire);
        if (t_target == input_dawg_sym) {
            if (t_target >= active_symbols || t_target >= MAX_SYMBOLS) {
                t_idx = atomic_load_explicit(&t->next, memory_order_acquire);
                continue;
            }
            Symbol *target_sym_obj = &ctx->dawg_nodes[t_target];
            NodeID target_canonical = target_sym_obj->canonical_node;
            if (target_canonical != INVALID && target_canonical < MAX_NODES) {
                SccID target_scc = atomic_load_explicit(&ctx->transient_nodes[target_canonical].scc_id, memory_order_acquire);
                if (target_scc != INVALID && target_scc < MAX_SCCS) {
                    if (out_to_scc) *out_to_scc = target_scc;
                    return atomic_load_explicit(&ctx->scc_nodes[target_scc].head, memory_order_acquire);
                }
            }
        }
        t_idx = atomic_load_explicit(&t->next, memory_order_acquire);
    }

    return INVALID;
}

/* ===============================================================================
 * INTERNAL HELPER: Token buffer allocation with stack optimization
 * Allocates a fixed stack buffer for up to 256 tokens, falling back to heap
 * only if the input text produces more tokens. This avoids allocator overhead
 * in the common case of short-to-medium length text.
 * =============================================================================== */
static int encode_tokens_optimized(int tok_handle, const char *text,
                                   uint32_t stack_buffer[256], uint32_t **out_buffer) {
    size_t cap = 256;
    uint32_t *token_buffer = stack_buffer;
    int count = tok_encode(tok_handle, text, token_buffer, cap);

    if (count >= (int)cap) {
        /* Fallback to heap allocation only if stack capacity is exceeded */
        while (1) {
            cap *= 2;
            if (cap > MAX_TOKEN_LIMIT) {
                *out_buffer = stack_buffer;  /* Return stack buffer on overflow */
                return -1;  /* Signal failure */
            }
            
            uint32_t *buf = malloc(cap * sizeof(uint32_t));
            if (!buf) {
                *out_buffer = stack_buffer;
                return -1;  /* Signal failure */
            }
            
            count = tok_encode(tok_handle, text, buf, cap);
            if (count < (int)cap) {
                token_buffer = buf;
                break;
            }
            free(buf);
        }
    }

    *out_buffer = token_buffer;
    return count;
}

NodeID engine_infer_text(int tok_handle, EngineContext *ctx, 
                         const char *text, NodeID start_node) {
    if (!ctx || !text) return start_node;

    uint32_t stack_buffer[256];
    uint32_t *token_buffer = NULL;
    int count = encode_tokens_optimized(tok_handle, text, stack_buffer, &token_buffer);

    if (count <= 0) {
        if (token_buffer != stack_buffer && token_buffer != NULL) {
            free(token_buffer);
        }
        return start_node;
    }

    NodeID current_node = start_node;
    for (int i = 0; i < count; i++) {
        NodeID next = engine_step(ctx, current_node, (SymbolID)token_buffer[i], NULL, NULL);
        if (next == INVALID) break; 
        current_node = next;
    }

    if (token_buffer != stack_buffer) free(token_buffer);
    return current_node;
}

NodeID engine_infer_text_direct_id(EngineContext *ctx, NodeID start_node, 
                                  SymbolID sym, SccID *out_from, SccID *out_to) {
    return engine_step(ctx, start_node, sym, out_from, out_to);
}

NodeID engine_ingest_text(int tok_handle, EngineContext *ctx,
                          const char *text, NodeID start_node) {
    if (!ctx || !text) return start_node;

    uint32_t stack_buffer[256];
    uint32_t *token_buffer = NULL;
    int count = encode_tokens_optimized(tok_handle, text, stack_buffer, &token_buffer);

    if (count <= 0) {
        if (token_buffer != stack_buffer && token_buffer != NULL) {
            free(token_buffer);
        }
        return start_node;
    }
    NodeID current_node = start_node;
    TokenID prev_sym = (start_node != INVALID && start_node < MAX_NODES) ? ctx->nodes[start_node].token_id : INVALID;  /* Track previous token for structural edge gating */
    for (int i = 0; i < count; i++) {
        SymbolID sym = (SymbolID)token_buffer[i];

        /* FIX A1: Intern surface form to populate canonical registry and freq metrics.
         * Without this, le_lexeme_frequency() returns 0 for all SCC candidates,
         * and seeded phrase components starve at epoch boundaries. */
        const char *surface = tok_decode(tok_handle, sym);
        if (surface && surface[0] != '\0') {
            le_intern_lexeme(ctx, surface);
        }

        NodeID next_node = le_get_or_create_node(ctx, sym);
        if (current_node != INVALID && next_node != INVALID) {
            if (current_node < MAX_NODES && next_node < MAX_NODES) {
                /* CRITICAL FIX: Structural edge gating to prevent giant SCC garbage buckets
                 * Filter out punctuation-bridged tokens from linking into core SCC matrices. */
                if (prev_sym != INVALID && !le_should_gate_structural_edge(ctx, tok_handle, prev_sym, sym)) {
                    le_add_edge(ctx, current_node, next_node);
                }
            }
        }
        current_node = next_node;
        prev_sym = sym;
    }

    if (token_buffer != stack_buffer) free(token_buffer);
    return current_node;
}

TransitionLineage* engine_ingest_text_with_lineage(int tok_handle, EngineContext *ctx,
                                                   const char *text, NodeID start_node) {
    if (!ctx || !text) return NULL;

    uint32_t stack_buffer[256];
    uint32_t *token_buffer = NULL;
    int count = encode_tokens_optimized(tok_handle, text, stack_buffer, &token_buffer);

    if (count <= 0) {
        if (token_buffer != stack_buffer && token_buffer != NULL) {
            free(token_buffer);
        }
        return NULL;
    }

    TransitionLineage *lineage = malloc(sizeof(TransitionLineage));
    if (!lineage) {
        /* FIX: Guard against freeing stack allocation */
        if (token_buffer != stack_buffer) {
            free(token_buffer);
        }
        return NULL;
    }

    lineage->capacity = (uint32_t)count;
    lineage->count = 0;
    lineage->entries = malloc(sizeof(LineageEntry) * lineage->capacity);
    if (!lineage->entries) {
        free(lineage);
        /* FIX: Guard against freeing stack allocation */
        if (token_buffer != stack_buffer) {
            free(token_buffer);
        }
        return NULL;
    }

    NodeID current_node = start_node;
    TokenID prev_sym = (start_node != INVALID && start_node < MAX_NODES) ? ctx->nodes[start_node].token_id : INVALID;  /* Track previous token for structural edge gating */
    for (int i = 0; i < count; i++) {
        SymbolID sym = (SymbolID)token_buffer[i];

        /* FIX A1: Canonical lexeme registration */
        const char *surface = tok_decode(tok_handle, sym);
        if (surface && surface[0] != '\0') {
            le_intern_lexeme(ctx, surface);
        }

        NodeID next_node = le_get_or_create_node(ctx, sym);
        if (current_node != INVALID && next_node != INVALID) {
            if (current_node < MAX_NODES && next_node < MAX_NODES) {
                /* CRITICAL FIX: Structural edge gating to prevent giant SCC garbage buckets
                 * Filter out punctuation-bridged tokens from linking into core SCC matrices. */
                if (prev_sym != INVALID && !le_should_gate_structural_edge(ctx, tok_handle, prev_sym, sym)) {
                    le_add_edge(ctx, current_node, next_node);
                }
            }
        }
        
        SccID from_scc = (current_node != INVALID && current_node < MAX_NODES) ? atomic_load_explicit(&ctx->transient_nodes[current_node].scc_id, memory_order_acquire) : INVALID;
        SccID to_scc   = (next_node   != INVALID && next_node < MAX_NODES) ? atomic_load_explicit(&ctx->transient_nodes[next_node].scc_id, memory_order_acquire) : INVALID;

        LineageEntry *entry = &lineage->entries[lineage->count++];
        entry->input_symbol = sym;
        entry->from_scc     = from_scc;
        entry->to_scc       = to_scc;
        entry->target_node  = next_node;
        
        current_node = next_node;
        prev_sym = sym;
    }

    if (token_buffer != stack_buffer && token_buffer != NULL) {
        free(token_buffer);
    }
    return lineage;
}

void engine_lineage_free(TransitionLineage *lineage) {
    if (!lineage) return;
    free(lineage->entries);
    free(lineage);
}

void engine_lineage_print(const TransitionLineage *lineage) {
    if (!lineage) {
        printf("Lineage: (empty)\n");
        return;
    }
    printf("--- Semantic Transition Lineage (%u steps) ---\n", lineage->count);
    for (uint32_t i = 0; i < lineage->count; i++) {
        const LineageEntry *e = &lineage->entries[i];
        if (e->from_scc == INVALID) {
            printf("Step %u: Sym[%u] | SCC INVALID -> ", i, e->input_symbol);
        } else {
            printf("Step %u: Sym[%u] | SCC %u -> ", i, e->input_symbol, e->from_scc);
        }
        if (e->target_node == INVALID) {
            printf("INVALID (stalled)\n");
        } else {
            printf("SCC %u (Node %u)\n", e->to_scc, e->target_node);
        }
    }
    printf("----------------------------------------------\n");
}

int dawg_predict(int tok_handle, EngineContext *ctx, NodeID current_node,
                 float tau, uint32_t max_predictions, char predictions[][DAWG_PREDICTION_MAX_LEN], float probabilities[]) {
    if (!ctx || current_node == INVALID || max_predictions == 0 ||
        !ctx->transient_nodes || !ctx->scc_nodes || !ctx->dawg_nodes || !ctx->dawg_transitions) {
        return 0;
    }

    uint32_t active_nodes = atomic_load_explicit(&ctx->node_count, memory_order_acquire);
    if (current_node >= active_nodes || current_node >= MAX_NODES) {
        return 0;
    }

    #define DAWG_PREDICT_MAX_RESULTS (MAX_BEAM_WIDTH * 10)
    if (max_predictions > DAWG_PREDICT_MAX_RESULTS) {
        max_predictions = DAWG_PREDICT_MAX_RESULTS;
    }

    SccID scc_id = atomic_load_explicit(&ctx->transient_nodes[current_node].scc_id, memory_order_acquire);
    if (scc_id == INVALID || scc_id >= MAX_SCCS || !atomic_load_explicit(&ctx->scc_nodes[scc_id].is_promoted, memory_order_acquire)) {
        return 0;
    }
    SymbolID start_sym = atomic_load_explicit(&ctx->scc_nodes[scc_id].symbol_id, memory_order_acquire);
    if (start_sym >= MAX_SYMBOLS) return 0;

    #define PREDICT_BEAM_WIDTH 10
    #define PREDICT_MAX_DEPTH 2

    PredictState beam[PREDICT_BEAM_WIDTH];
    PredictState next_beam[PREDICT_BEAM_WIDTH * 10];
    memset(beam, 0, sizeof(beam));
    memset(next_beam, 0, sizeof(next_beam));

    beam[0].current = start_sym;
    beam[0].score = 0.0f;
    beam[0].length = 1;
    beam[0].path[0] = start_sym;
    uint32_t beam_size = 1;

    for (uint32_t depth = 0; depth < PREDICT_MAX_DEPTH && beam_size > 0; depth++) {
        uint32_t next_size = 0;
        for (uint32_t i = 0; i < beam_size; i++) {
            PredictState *state = &beam[i];
            if (state->current >= MAX_SYMBOLS) continue;
            Symbol *sym = &ctx->dawg_nodes[state->current];
            uint32_t t_idx = atomic_load_explicit(&sym->first_transition, memory_order_acquire);
            
            while (t_idx != INVALID && next_size < PREDICT_BEAM_WIDTH * 10) {
                if (t_idx >= atomic_load_explicit(&ctx->dawg_transition_count, memory_order_acquire) || t_idx >= MAX_DAWG_TRANSITIONS) break;
                DawgTransition *t = &ctx->dawg_transitions[t_idx];
                SymbolID t_target = atomic_load_explicit(&t->target, memory_order_acquire);
                
                uint32_t active_syms = atomic_load_explicit(&ctx->symbol_count, memory_order_acquire);
                if (t_target >= active_syms || t_target >= MAX_SYMBOLS) {
                    t_idx = atomic_load_explicit(&t->next, memory_order_acquire);
                    continue;
                }
                uint32_t tw_bits = atomic_load_explicit(&t->weight, memory_order_acquire);
                float tw;
                memcpy(&tw, &tw_bits, sizeof(float));
                if (tw >= tau) {
                    PredictState *new_state = &next_beam[next_size++];
                    new_state->current = t_target;
                    new_state->score = state->score + (tw > 0.0f ? logf(tw) : -1000.0f);
                    new_state->length = state->length + 1;
                    memcpy(new_state->path, state->path, state->length * sizeof(SymbolID));
                    new_state->path[state->length] = t_target;
                }
                t_idx = atomic_load_explicit(&t->next, memory_order_acquire);
            }
        }

        if (next_size > 0) {
            qsort(next_beam, next_size, sizeof(PredictState), compare_predict_states);
            beam_size = (next_size < PREDICT_BEAM_WIDTH) ? next_size : PREDICT_BEAM_WIDTH;
            memcpy(beam, next_beam, beam_size * sizeof(PredictState));
        } else {
            break;
        }
    }

    uint32_t pred_count = 0;
    TokenID seen_tids[DAWG_PREDICT_MAX_RESULTS] = {0};

    for (uint32_t i = 0; i < beam_size && pred_count < max_predictions; i++) {
        PredictState *state = &beam[i];
        if (state->length <= 1) continue;

        SymbolID next_sym = state->path[1];
        if (next_sym >= atomic_load_explicit(&ctx->symbol_count, memory_order_acquire) || next_sym >= MAX_SYMBOLS) continue;

        NodeID member_buf[32];
        uint32_t mc = le_get_symbol_nodes(ctx, next_sym, member_buf, 32);
        if (mc > 32) mc = 32;

        for (uint32_t m = 0; m < mc && pred_count < max_predictions; m++) {
            TokenID tid = get_node_token(ctx, member_buf[m]);
            if (tid == INVALID) continue;
            bool dup = false;
            
            for (uint32_t p = 0; p < pred_count; p++) {
                if (seen_tids[p] == tid) {
                    dup = true;
                    float prob = expf(state->score);
                    if (prob > probabilities[p]) probabilities[p] = prob;
                    break;
                }
            }

            if (!dup) {
                const char *decoded = tok_decode(tok_handle, tid);
                /* FIX: Skip corrupted decoded strings - tokenizer string pool may be corrupted */
                if (!decoded || strcmp(decoded, "<?>") == 0 || strlen(decoded) < 1 || strlen(decoded) >= DAWG_PREDICTION_MAX_LEN) continue;
                strncpy(predictions[pred_count], decoded, DAWG_PREDICTION_MAX_LEN - 1);
                predictions[pred_count][DAWG_PREDICTION_MAX_LEN - 1] = '\0';
                probabilities[pred_count] = expf(state->score);
                seen_tids[pred_count] = tid;  
                pred_count++;
            }
        }
    }

    return (int)pred_count;
}

int engine_infer_distributed(int tok_handle, EngineContext *ctx, 
                             const char *text, NodeID start_node,
                             ActiveState *out_states, uint32_t max_states) {
    if (!ctx || !text || !out_states || max_states == 0) return 0;

    uint32_t stack_buffer[256];
    uint32_t *token_buffer = NULL;
    int count = encode_tokens_optimized(tok_handle, text, stack_buffer, &token_buffer);

    if (count <= 0) {
        if (token_buffer != stack_buffer && token_buffer != NULL) {
            free(token_buffer);
        }
        return 0;
    }

    ActiveState active_states[MAX_ACTIVE_STATES];
    uint32_t active_count = 0;
    
    uint32_t active_nodes = atomic_load_explicit(&ctx->node_count, memory_order_acquire);
    if (start_node != INVALID && start_node < active_nodes && start_node < MAX_NODES && active_count < MAX_ACTIVE_STATES) {
        active_states[active_count].node = start_node;
        active_states[active_count].activation = 1.0f;
        active_count++;
    }

    for (int i = 0; i < count && active_count > 0; i++) {
        SymbolID sym = (SymbolID)token_buffer[i];
        ActiveState next_states[MAX_ACTIVE_STATES];
        uint32_t next_count = 0;

        for (uint32_t j = 0; j < active_count; j++) {
            NodeID current = active_states[j].node;
            float activation = active_states[j].activation;

            NodeID next = engine_step(ctx, current, sym, NULL, NULL);
            if (next != INVALID) {
                bool found = false;
                for (uint32_t k = 0; k < next_count; k++) {
                    if (next_states[k].node == next) {
                        next_states[k].activation += activation * 0.9f;
                        found = true;
                        break;
                    }
                }
                
                if (!found && next_count < MAX_ACTIVE_STATES) {
                    next_states[next_count].node = next;
                    next_states[next_count].activation = activation * 0.9f;
                    next_count++;
                }
            }

            /* CORR-3 FIX: Properly update activation weights of already-inserted next_states
             * Traverse edge list without early termination based on next_count.
             * Stop traversal only when edge list is exhausted or we reach MAX_EDGES,
             * not when we've collected MAX_ACTIVE_STATES (which silently drops valid resonances). */
            uint32_t edge_count_snap = atomic_load_explicit(&ctx->edge_count, memory_order_acquire);
            if (current < MAX_NODES) {
                EdgeID e = atomic_load_explicit(&ctx->nodes[current].first_edge, memory_order_acquire);
                uint32_t traversal_count = 0;  /* Count traversals; truncate at MAX_EDGES to prevent infinite loops */
                while (e != INVALID && e < edge_count_snap && e < MAX_EDGES) {
                    traversal_count++;
                    if (traversal_count > MAX_EDGES) break;  /* Prevent unbounded traversal on cyclic edge lists */
                    
                    NodeID alt = ctx->edges[e].target;
                    uint32_t weight_bits = atomic_load_explicit(&ctx->edges[e].weight, memory_order_acquire);
                    float weight;
                    memcpy(&weight, &weight_bits, sizeof(float));
                    
                    NodeID alt_next = engine_step(ctx, alt, sym, NULL, NULL);
                    if (alt_next != INVALID) {
                        bool found = false;
                        for (uint32_t k = 0; k < next_count; k++) {
                            if (next_states[k].node == alt_next) {
                                float resonance = activation * (weight / (weight + 1.0f)) * 0.3f;
                                next_states[k].activation += resonance;  /* UPDATE: Accumulate into existing state */
                                found = true;
                                break;
                            }
                        }
                        
                        if (!found && next_count < MAX_ACTIVE_STATES) {
                            float resonance = activation * (weight / (weight + 1.0f)) * 0.3f;
                            next_states[next_count].node = alt_next;
                            next_states[next_count].activation = resonance;
                            next_count++;
                        }
                    }
                    
                    EdgeID next_e = atomic_load_explicit(&ctx->edge_nexts[e], memory_order_acquire);
                    e = (next_e != INVALID && next_e < edge_count_snap && next_e < MAX_EDGES) ? next_e : INVALID;
                }
            }
        }

        float total_activation = 0.0f;
        for (uint32_t j = 0; j < next_count; j++) {
            total_activation += next_states[j].activation;
        }
        
        if (total_activation > 0.0f) {
            for (uint32_t j = 0; j < next_count; j++) {
                next_states[j].activation /= total_activation;
            }
        }

        active_count = 0;
        /* CORR-4 FIX: Filter low-activation states to prevent probability underflow accumulation */
        for (uint32_t j = 0; j < next_count; j++) {
            if (next_states[j].activation > 0.01f) { 
                active_states[active_count++] = next_states[j];
            }
        }
    }

    uint32_t out_count = (active_count < max_states) ? active_count : max_states;
    for (uint32_t i = 0; i < out_count; i++) {
        out_states[i] = active_states[i];
    }

    if (token_buffer != stack_buffer && token_buffer != NULL) {
        free(token_buffer);
    }
    return (int)out_count;
}

int dawg_predict_multi_hop(int tok_handle, EngineContext *ctx, NodeID current_node,
                          float tau, uint32_t max_predictions, uint32_t max_depth,
                          char predictions[][DAWG_PREDICTION_MAX_LEN], float probabilities[]) {
    if (!ctx || current_node == INVALID || max_predictions == 0 || max_depth == 0 ||
        !ctx->transient_nodes || !ctx->scc_nodes || !ctx->dawg_nodes || !ctx->dawg_transitions) {
        return 0;
    }

    uint32_t active_nodes = atomic_load_explicit(&ctx->node_count, memory_order_acquire);
    if (current_node >= active_nodes || current_node >= MAX_NODES) {
        return 0;
    }

    SccID scc_id = atomic_load_explicit(&ctx->transient_nodes[current_node].scc_id, memory_order_acquire);
    if (scc_id == INVALID || scc_id >= MAX_SCCS || !atomic_load_explicit(&ctx->scc_nodes[scc_id].is_promoted, memory_order_acquire)) {
        return 0;
    }
    SymbolID start_sym = atomic_load_explicit(&ctx->scc_nodes[scc_id].symbol_id, memory_order_acquire);
    if (start_sym >= MAX_SYMBOLS) return 0;

    #define MULTI_HOP_BEAM_WIDTH 20
    #define MULTI_HOP_MAX_DEPTH 5
    #define MULTI_HOP_ENTROPY_CAP 2.0f

    if (max_depth > MULTI_HOP_MAX_DEPTH) max_depth = MULTI_HOP_MAX_DEPTH;

    MultiHopState beam[MULTI_HOP_BEAM_WIDTH];
    MultiHopState next_beam[MULTI_HOP_BEAM_WIDTH * 10];
    memset(beam, 0, sizeof(beam));
    memset(next_beam, 0, sizeof(next_beam));

    beam[0].current = start_sym;
    beam[0].score = 0.0f;
    beam[0].length = 1;
    beam[0].entropy = 0.0f;
    beam[0].path[0] = start_sym;
    uint32_t beam_size = 1;

    for (uint32_t depth = 0; depth < max_depth && beam_size > 0; depth++) {
        uint32_t next_size = 0;
        for (uint32_t i = 0; i < beam_size; i++) {
            MultiHopState *state = &beam[i];
            if (state->current >= MAX_SYMBOLS) continue;
            Symbol *sym = &ctx->dawg_nodes[state->current];
            uint32_t t_idx = atomic_load_explicit(&sym->first_transition, memory_order_acquire);
            
            while (t_idx != INVALID && next_size < MULTI_HOP_BEAM_WIDTH * 10) {
                if (t_idx >= atomic_load_explicit(&ctx->dawg_transition_count, memory_order_acquire) || t_idx >= MAX_DAWG_TRANSITIONS) break;
                DawgTransition *t = &ctx->dawg_transitions[t_idx];
                SymbolID t_target = atomic_load_explicit(&t->target, memory_order_acquire);
                
                uint32_t active_syms = atomic_load_explicit(&ctx->symbol_count, memory_order_acquire);
                if (t_target >= active_syms || t_target >= MAX_SYMBOLS) {
                    t_idx = atomic_load_explicit(&t->next, memory_order_acquire);
                    continue;
                }
                uint32_t tw_bits = atomic_load_explicit(&t->weight, memory_order_acquire);
                float tw;
                memcpy(&tw, &tw_bits, sizeof(float));
                
                if (tw >= tau) {
                    float transition_entropy = -tw * logf(tw + 1e-10f);
                    float new_entropy = state->entropy + transition_entropy;

                    if (new_entropy < MULTI_HOP_ENTROPY_CAP || depth < 2) {
                        /* CRITICAL FIX: Check path array bounds before write to prevent OOB */
                        if (state->length >= MAX_SEARCH_DEPTH - 1) {
                            t_idx = atomic_load_explicit(&t->next, memory_order_acquire);
                            continue;
                        }
                        MultiHopState *new_state = &next_beam[next_size++];
                        new_state->current = t_target;
                        new_state->score = state->score + (tw > 0.0f ? logf(tw) : -1000.0f);
                        new_state->length = state->length + 1;
                        new_state->entropy = new_entropy;
                        memcpy(new_state->path, state->path, state->length * sizeof(SymbolID));
                        new_state->path[state->length] = t_target;
                    }
                }
                t_idx = atomic_load_explicit(&t->next, memory_order_acquire);
            }
        }

        if (next_size > 0) {
            qsort(next_beam, next_size, sizeof(MultiHopState), compare_multihop_states);
            beam_size = (next_size < MULTI_HOP_BEAM_WIDTH) ? next_size : MULTI_HOP_BEAM_WIDTH;
            memcpy(beam, next_beam, beam_size * sizeof(MultiHopState));
        } else {
            break;
        }
    }

    uint32_t pred_count = 0;
    for (uint32_t i = 0; i < beam_size && pred_count < max_predictions; i++) {
        MultiHopState *state = &beam[i];
        if (state->length <= 1) continue;

        SymbolID next_sym = state->path[1];
        if (next_sym >= atomic_load_explicit(&ctx->symbol_count, memory_order_acquire) || next_sym >= MAX_SYMBOLS) continue;

        NodeID member_buf[32];
        memset(member_buf, 0xFF, sizeof(member_buf));
        
        uint32_t mc = le_get_symbol_nodes(ctx, next_sym, member_buf, 32);
        if (mc > 32) mc = 32;

        for (uint32_t m = 0; m < mc && pred_count < max_predictions; m++) {
            NodeID curr_node = member_buf[m];
            if (curr_node == INVALID || curr_node >= atomic_load_explicit(&ctx->node_count, memory_order_acquire) || curr_node >= MAX_NODES) {
                continue;
            }

            TokenID tid = get_node_token(ctx, curr_node);
            if (tid == INVALID) continue;
            const char *decoded = tok_decode(tok_handle, tid);
            /* FIX: Skip corrupted decoded strings - tokenizer string pool may be corrupted */
            if (!decoded || strcmp(decoded, "<?>") == 0 || strlen(decoded) < 1 || strlen(decoded) >= DAWG_PREDICTION_MAX_LEN) continue;

            bool dup = false;
            for (uint32_t p = 0; p < pred_count; p++) {
                if (strcmp(predictions[p], decoded) == 0) {
                    dup = true;
                    float prob = expf(state->score);
                    if (prob > probabilities[p]) {
                        probabilities[p] = prob;
                    }
                    break;
                }
            }

            if (!dup) {
                strncpy(predictions[pred_count], decoded, DAWG_PREDICTION_MAX_LEN - 1);
                predictions[pred_count][DAWG_PREDICTION_MAX_LEN - 1] = '\0'; 
                probabilities[pred_count] = expf(state->score);
                pred_count++;
            }
        }
    }

    return (int)pred_count;
}