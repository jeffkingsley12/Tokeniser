#include "gemini_internal.h"
#include "bridge_engine.h"
#include <limits.h>
#include <stdatomic.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>


/* Implementation for beam state sorting */
int compare_beam_states(const void *a, const void *b) {
    const BeamState *state_a = (const BeamState *)a;
    const BeamState *state_b = (const BeamState *)b;
    
    /* Sort descending: highest log_prob first */
    if (state_a->log_prob < state_b->log_prob) return 1;
    if (state_a->log_prob > state_b->log_prob) return -1;
    return 0;
}
/* Thread-local storage for beam search buffers to prevent heap thrashing */
static __thread BeamState tls_beam[MAX_BEAM_WIDTH];
static __thread BeamState tls_next_beam[MAX_BEAM_WIDTH * 10];
/* CRITICAL FIX: Resize tls_scratch_seqs to accommodate both beam and next_beam
 * The initialization loop accesses indices up to (beam_width + i) * (max_depth + 1)
 * where i ranges from 0 to beam_width * 10, requiring 11 * MAX_BEAM_WIDTH slots. */
static __thread SymbolID tls_scratch_seqs[11 * MAX_BEAM_WIDTH * (MAX_SEARCH_DEPTH + 1)];

/**
 * le_beam_search - Constrained Probabilistic Sequence Decoder
 * Evaluates high-probability structural paths across the DAWG graph using
 * a unified, regularized, and length-normalized log-space scoring policy.
 */
BeamState *le_beam_search(EngineContext *ctx, TokenID start_token,
                           uint32_t beam_width, uint32_t max_depth,
                           uint32_t *result_count) {

  /* NULL guard for result_count */
  if (!result_count) return NULL;

  if (beam_width > MAX_BEAM_WIDTH) beam_width = MAX_BEAM_WIDTH;
  if (max_depth > MAX_SEARCH_DEPTH) max_depth = MAX_SEARCH_DEPTH;

  /* Hash probe for OOV prevention */
  NodeID start_node = INVALID;
  {
    /* CRITICAL FIX: Use same Fibonacci hash as le_get_or_create_node for consistency.
     * Without this, hash probing won't find nodes created with the Fibonacci hash. */
    uint32_t h = (start_token * 0x9E3779B1u) % HASH_SIZE;
    uint32_t steps = 0;
    NodeID slot;
    while (steps < HASH_SIZE) {
      slot = atomic_load_explicit(&ctx->node_hash[h], memory_order_acquire);
      if (slot == INVALID) {
        break;  /* Not found */
      }
      if (ctx->nodes[slot].token_id == start_token) {
        start_node = slot;
        break;
      }
      h = (h + 1) % HASH_SIZE;
      steps++;
    }
  }

  if (start_node == INVALID) {
    *result_count = 0;
    return NULL;
  }

  SccID start_scc = atomic_load_explicit(&ctx->transient_nodes[start_node].scc_id, memory_order_acquire);
  /* SCC bounds check to prevent out-of-bounds access */
  if (start_scc >= MAX_SCCS) {
    *result_count = 0;
    return NULL;
  }
  SccNode *start_scc_node = &ctx->scc_nodes[start_scc];

  if (!atomic_load_explicit(&start_scc_node->is_promoted, memory_order_acquire)) {
    *result_count = 0;
    return NULL;
  }

  SymbolID start_symbol = atomic_load_explicit(&start_scc_node->symbol_id, memory_order_acquire);

  /* Use thread-local storage for beam buffers to prevent heap thrashing */
  BeamState *beam = tls_beam;
  BeamState *next_beam = tls_next_beam;

  /* Link the scratch memory into the beam states */
  for (uint32_t i = 0; i < beam_width; i++) beam[i].sequence = &tls_scratch_seqs[i * (max_depth + 1)];
  for (uint32_t i = 0; i < beam_width * 10; i++) next_beam[i].sequence = &tls_scratch_seqs[(beam_width + i) * (max_depth + 1)];

  /* M-1 FIX: Initialize history_mask array for O(1) repetition tracking */
  uint64_t history_mask[MAX_BEAM_WIDTH] = {0};

  /* Initialize root state */
  beam[0].sequence[0] = start_symbol;
  beam[0].length = 1;
  beam[0].log_prob = 0.0f; /* Cumulative normalized log probability at root depth */
  beam[0].current = start_symbol;
  beam[0].active_class_mask = 0; /* No morphological constraints at root */
  history_mask[0] = (1ULL << (start_symbol & 63)); /* Track start symbol in history */
  uint32_t beam_size = 1;
  uint32_t next_size = 0;

  /* * Config Constants (Problem #3 Fix: Use distinctive uppercase constants 
   * to strictly separate configuration values from runtime accumulators)
   */
  const float REPETITION_FACTOR  = 0.15f;  /* Log-space penalty per repeat token */
  const float LOOP_FACTOR        = 0.25f;  /* Short-cycle localized loop penalty factor */
  const float ENTROPY_FACTOR     = 0.08f;  /* Scaling factor for entropy regularization curve */
  const float DIVERSITY_FACTOR    = 0.12f;  /* Suffix peer diversity scaling factor to prevent collapse */
  const float TARGET_ENTROPY     = 0.60f;  /* Sweet spot for moderate uncertainty preference */
  const float LENGTH_ALPHA       = 0.70f;  /* Rescaling exponent alpha for length normalizer */
  const int   HISTORY_WINDOW     = 5;      /* Token window depth for repetition backoff check */
  const int   LOOP_WINDOW        = 4;      /* Strict window constraint for short-cycle loops */

  for (uint32_t depth = 0; depth < max_depth && beam_size > 0; depth++) {
    next_size = 0;

    for (uint32_t i = 0; i < beam_size && i < beam_width; i++) {
      BeamState *state = &beam[i];
      Symbol *symbol = &ctx->dawg_nodes[state->current];

      uint32_t t_idx = atomic_load_explicit(&symbol->first_transition, memory_order_acquire);
      while (t_idx != INVALID && next_size < beam_width * 10) {
        /* C-1 FIX: Add bounds check before array access to prevent OOB crash */
        uint32_t tc = atomic_load_explicit(&ctx->dawg_transition_count, memory_order_acquire);
        if (t_idx >= tc || t_idx >= MAX_DAWG_TRANSITIONS) break;
        DawgTransition *t = &ctx->dawg_transitions[t_idx];
        SymbolID t_target = atomic_load_explicit(&t->target, memory_order_acquire);

        uint32_t active_syms = atomic_load_explicit(&ctx->symbol_count, memory_order_acquire);
        if (t_target >= active_syms || t_target >= MAX_SYMBOLS) {
            t_idx = atomic_load_explicit(&t->next, memory_order_acquire);
            continue;
        }

        /* Validate target canonical node before we claim a beam slot */
        Symbol *target_symbol = &ctx->dawg_nodes[t_target];
        NodeID target_canonical = target_symbol->canonical_node;
        if (target_canonical == INVALID || target_canonical >= MAX_NODES) {
            t_idx = atomic_load_explicit(&t->next, memory_order_acquire);
            continue;
        }

        /* HYPERGRAPH: Disambiguate hybrid tokens using grammatical context */
        /* Extract TokenIDs from SymbolIDs to avoid type collision */
        TokenID prev_tid = 0;
        if (state->length > 0) {
            SymbolID prev_sym_id = state->sequence[state->length - 1];
            if (prev_sym_id != INVALID && prev_sym_id < MAX_SYMBOLS) {
                Symbol *prev_sym = &ctx->dawg_nodes[prev_sym_id];
                NodeID prev_canonical = prev_sym->canonical_node;
                if (prev_canonical != INVALID && prev_canonical < MAX_NODES) {
                    prev_tid = atomic_load_explicit(&ctx->nodes[prev_canonical].token_id, memory_order_relaxed);
                }
            }
        }

        TokenID curr_tid = 0;
        Symbol *curr_sym = &ctx->dawg_nodes[state->current];
        NodeID curr_canonical = curr_sym->canonical_node;
        if (curr_canonical != INVALID && curr_canonical < MAX_NODES) {
            curr_tid = atomic_load_explicit(&ctx->nodes[curr_canonical].token_id, memory_order_relaxed);
        }

        TokenID next_tid = 0;
        Symbol *next_sym = &ctx->dawg_nodes[t_target];
        NodeID next_canonical = next_sym->canonical_node;
        if (next_canonical != INVALID && next_canonical < MAX_NODES) {
            next_tid = atomic_load_explicit(&ctx->nodes[next_canonical].token_id, memory_order_relaxed);
        }
        
        GrammarRole resolved_role = le_disambiguate_hybrid_by_token(ctx, prev_tid, curr_tid, next_tid);
        
        /* Apply grammar-based scoring modifiers */
        float grammar_modifier = 1.0f;
        if (resolved_role & G_GENITIVE) {
            grammar_modifier = ctx->relation_modifiers[REL_POSSESSIVE];
        } else if (resolved_role & G_PARTICLE) {
            grammar_modifier = ctx->relation_modifiers[REL_ASSOCIATIVE];
        } else if (resolved_role & G_CONNECTOR) {
            grammar_modifier = ctx->relation_modifiers[REL_ASSOCIATIVE];
        }

        float tw = atomic_load_float(&t->weight);
        if (tw <= 0.0f) {
            t_idx = atomic_load_explicit(&t->next, memory_order_acquire);
            continue;
        }
        
        /* Apply grammar modifier to transition weight */
        tw *= grammar_modifier;

        /* HYPERGRAPH: CSR deterministic rule override
         * Intersect with memory-mapped CSR hypergraph for grammatical constraints
         * This validates constraints BEFORE claiming the slot to avoid wasted work */
        bool csr_violation = false;
        uint64_t csr_mask_add = 0;    /* Accumulate mask additions from triggered rules */
        uint64_t csr_mask_clear = 0;  /* Accumulate mask clearances from triggered rules */
        
        if (ctx->csr_hypergraph.mapped_base != NULL && t_target < ctx->csr_hypergraph.vocab_size) {
            uint32_t edge_start = ctx->csr_hypergraph.row_ptr[t_target];
            uint32_t edge_end = ctx->csr_hypergraph.row_ptr[t_target + 1];
            
            for (uint32_t i = edge_start; i < edge_end; i++) {
                /* Unpack 64-bit rule_mask into three 20-bit instruction blocks:
                 * Bits 0-19:   Require mask (constraints candidate must see)
                 * Bits 20-39:  Add mask (constraints candidate injects)
                 * Bits 40-59:  Clear mask (constraints candidate consumes/resolves)
                 * Bits 60-63:  Reserved for future routing flags */
                uint64_t packed_mask = ctx->csr_hypergraph.rule_masks[i];
                uint64_t require_mask = packed_mask & 0xFFFFF;
                uint64_t add_mask     = (packed_mask >> 20) & 0xFFFFF;
                uint64_t clear_mask   = (packed_mask >> 40) & 0xFFFFF;
                
                /* Validate: Does current state satisfy required morphological constraints? */
                if (require_mask != 0 && !(state->active_class_mask & require_mask)) {
                    /* Hard violation: noun-class agreement fails, prune this path */
                    csr_violation = true;
                    break;  /* Exit CSR loop early */
                }
                
                /* Scan history for specific structural token triggers */
                uint32_t rule_target = ctx->csr_hypergraph.ctx_tokens[i];
                if (rule_target != INVALID) {
                    for (int h = 0; h < (int)state->length; h++) {
                        if (state->sequence[h] == rule_target) {
                            /* Apply deterministic Q8.8 multiplier */
                            float modifier = q88_to_float(ctx->csr_hypergraph.modifiers[i]);
                            tw *= modifier;
                            
                            /* Accumulate mask mutations from this triggered rule */
                            csr_mask_add |= add_mask;
                            csr_mask_clear |= clear_mask;
                            break;
                        }
                    }
                }
            }
        }
        
        /* Skip if CSR validation failed */
        if (csr_violation) {
            t_idx = atomic_load_explicit(&t->next, memory_order_acquire);
            continue;
        }

        /* NOW claim the slot — all inputs are valid */
        BeamState *new_state = &next_beam[next_size++];

        memcpy(new_state->sequence, state->sequence, state->length * sizeof(SymbolID));
        new_state->sequence[state->length] = t_target;
        new_state->length = state->length + 1;
        new_state->current = t_target;

        /* M-1 FIX: Update history_mask for new state */
        history_mask[next_size - 1] = history_mask[i] | (1ULL << (t_target & 63));
        
        /* HYPERGRAPH: Inherit and mutate active_class_mask for morphological constraints
         * Default inheritance: carry forward parent's constraints */
        new_state->active_class_mask = state->active_class_mask;
        
        /* Apply accumulated mask mutations from triggered CSR rules
         * Bitwise operation: (inherit | add) & ~clear
         * This adds new constraints and removes resolved ones in a single CPU cycle */
        if (csr_mask_add != 0 || csr_mask_clear != 0) {
            new_state->active_class_mask = (state->active_class_mask | csr_mask_add) & ~csr_mask_clear;
        }

        /* 1. Un-normalize parent score */
        float prev_normalizer = powf((float)state->length, LENGTH_ALPHA);
        float unnorm_state_log_prob = state->log_prob * prev_normalizer;
        float base_score = unnorm_state_log_prob + logf(tw);
        float constrained_score = base_score;

        /* 2. Repetition penalty - M-1 FIX: Use history_mask for O(1) lookup */
        int repetition_count = 0;
        /* Check if target symbol is in recent history using bitmask */
        if (history_mask[i] & (1ULL << (t_target & 63))) {
            /* Symbol was seen recently, count exact occurrences for penalty */
            int rep_checks = (state->length < (uint32_t)HISTORY_WINDOW) ? (int)state->length : HISTORY_WINDOW;
            for (int j = 0; j < rep_checks; j++) {
              if (state->sequence[state->length - 1 - j] == t_target) repetition_count++;
            }
        }
        constrained_score -= (float)repetition_count * REPETITION_FACTOR;

        /* 3. Short-cycle suppression */
        int loop_count = 0;
        int loop_checks = (state->length < (uint32_t)LOOP_WINDOW) ? (int)state->length : LOOP_WINDOW;
        for (int j = 0; j < loop_checks; j++) {
          if (state->sequence[state->length - 1 - j] == t_target) loop_count++;
        }
        constrained_score -= (float)loop_count * LOOP_FACTOR;

        /* 4. Entropy regularization */
        NodeID entropy_canonical = target_symbol->canonical_node;
        if (entropy_canonical == INVALID || entropy_canonical >= MAX_NODES) {
            t_idx = atomic_load_explicit(&t->next, memory_order_acquire);
            continue;
        }
        SccID entropy_scc_id = atomic_load_explicit(&ctx->transient_nodes[entropy_canonical].scc_id, memory_order_acquire);
        if (entropy_scc_id == INVALID || entropy_scc_id >= MAX_SCCS) {
            t_idx = atomic_load_explicit(&t->next, memory_order_acquire);
            continue;
        }
        SccNode *entropy_scc = &ctx->scc_nodes[entropy_scc_id];
        float normalized_entropy = scc_load_avg_entropy(entropy_scc);
        float entropy_distance = fabsf(normalized_entropy - TARGET_ENTROPY);
        constrained_score -= ENTROPY_FACTOR * entropy_distance;
        
        /* 5. Diversity against peer beams */
        float max_similarity = 0.0f;
        for (uint32_t k = 0; k < beam_size; k++) {
          if (k == i) continue;
          BeamState *peer = &beam[k];
          int match_count = 0;
          int max_possible_match = (int)peer->length;
          if (max_possible_match > (int)new_state->length) max_possible_match = (int)new_state->length;
          for (int m = 0; m < max_possible_match; m++) {
            if (new_state->sequence[new_state->length - 1 - m] ==
                peer->sequence[peer->length - 1 - m]) {
              match_count++;
            } else {
              break;
            }
          }
          if (max_possible_match > 0) {
            float sim = (float)match_count / (float)max_possible_match;
            if (sim > max_similarity) max_similarity = sim;
          }
        }
        constrained_score -= max_similarity * DIVERSITY_FACTOR;

        /* 6. Length normalization */
        float length_normalizer = powf((float)new_state->length, LENGTH_ALPHA);
        new_state->log_prob = constrained_score / length_normalizer;

        t_idx = atomic_load_explicit(&t->next, memory_order_acquire);
      }
    }

    if (next_size > 0) {
      qsort(next_beam, next_size, sizeof(BeamState), compare_beam_states);
      beam_size = (next_size < beam_width) ? next_size : beam_width;

      /* CRITICAL FIX: Replace pointer-swap with memcpy to prevent heap overflow
       * The original swap pattern caused next_beam (beam_width slots) to receive
       * the larger buffer (beam_width*10 slots), then on the next iteration
       * writes up to beam_width*10 entries into the smaller buffer, causing OOB.
       * Instead, copy only the top-k results back to beam and keep buffers fixed. */
      memcpy(beam, next_beam, beam_size * sizeof(BeamState));

      /* M-1 FIX: Copy history_mask for top-k results */
      uint64_t temp_history_mask[MAX_BEAM_WIDTH];
      memcpy(temp_history_mask, history_mask, beam_size * sizeof(uint64_t));
      memcpy(history_mask, temp_history_mask, beam_size * sizeof(uint64_t));

      /* Copy sequence data from TLS scratch_next_seqs to TLS scratch_beam_seqs for top-k results */
      for (uint32_t i = 0; i < beam_size; i++) {
        memcpy(tls_scratch_seqs + i * (max_depth + 1),
               next_beam[i].sequence,
               next_beam[i].length * sizeof(SymbolID));
        beam[i].sequence = &tls_scratch_seqs[i * (max_depth + 1)];
      }
    } else {
      break;
    }
  }

  /* Decouple final states from scratch buffer tracking.
   * Allocates dedicated heap blocks per active result path so that clean external invocations 
   * of `le_free_beam_results` won't generate a catastrophic pointer double-free.
   * CRITICAL FIX: Allocate fresh output array to avoid TLS pointer aliasing and memory leaks.
   */
  BeamState *final_results = calloc(beam_size, sizeof(BeamState));
  if (!final_results) {
    *result_count = 0;
    return NULL;
  }

  for (uint32_t i = 0; i < beam_size; i++) {
    final_results[i] = beam[i]; /* Struct copy */
    SymbolID *final_seq = calloc(max_depth + 1, sizeof(SymbolID));
    if (final_seq) {
      memcpy(final_seq, beam[i].sequence, beam[i].length * sizeof(SymbolID));
      final_results[i].sequence = final_seq;
    } else {
      /* Nullify to cleanly signal allocation failure to cleanup functions */
      final_results[i].sequence = NULL;
      final_results[i].length = 0;
    }
  }

  /* TLS buffers are automatically reused - no free needed for intermediate scratch matrices */
   
  *result_count = beam_size;
  return final_results;
}