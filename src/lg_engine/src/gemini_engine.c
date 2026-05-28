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
    uint32_t h = start_token % HASH_SIZE;
    NodeID slot;
    while ((slot = atomic_load_explicit(&ctx->node_hash[h], memory_order_acquire)) != INVALID) {
      if (ctx->nodes[slot].token_id == start_token) {
        start_node = slot;
        break;
      }
      h = (h + 1) % HASH_SIZE;
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

  /* Pre-allocate primary structs */
  BeamState *beam = calloc(beam_width, sizeof(BeamState));
  BeamState *next_beam = calloc(beam_width * 10, sizeof(BeamState));
   
  /* CRITICAL FIX: Pre-allocate flat scratch matrices for sequence tracking to prevent heap thrashing */
  SymbolID *scratch_beam_seqs = calloc(beam_width, (max_depth + 1) * sizeof(SymbolID));
  SymbolID *scratch_next_seqs = calloc(beam_width * 10, (max_depth + 1) * sizeof(SymbolID));

  if (!beam || !next_beam || !scratch_beam_seqs || !scratch_next_seqs) {
    free(beam); free(next_beam); free(scratch_beam_seqs); free(scratch_next_seqs);
    *result_count = 0;
    return NULL;
  }

  /* Link the scratch memory into the beam states */
  for (uint32_t i = 0; i < beam_width; i++) beam[i].sequence = &scratch_beam_seqs[i * (max_depth + 1)];
  for (uint32_t i = 0; i < beam_width * 10; i++) next_beam[i].sequence = &scratch_next_seqs[i * (max_depth + 1)];

  /* Initialize root state */
  beam[0].sequence[0] = start_symbol;
  beam[0].length = 1;
  beam[0].log_prob = 0.0f; /* Cumulative normalized log probability at root depth */
  beam[0].current = start_symbol;
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
        DawgTransition *t = &ctx->dawg_transitions[t_idx];
        BeamState *new_state = &next_beam[next_size++];
        
        /* Direct fast-memory copy within the pre-allocated scratch buffer */
        memcpy(new_state->sequence, state->sequence, state->length * sizeof(SymbolID));
        new_state->sequence[state->length] = t->target;
        new_state->length = state->length + 1;
        new_state->current = t->target;

        /* * 1. Base Probabilistic Score & Un-normalization (Problem #1 & #8 Fix)
         * Before adding the new token transition's log-probability, the parent score 
         * must be un-normalized to get back to clean cumulative log-space coordinates.
         */
        float prev_normalizer = powf((float)state->length, LENGTH_ALPHA);
        float unnorm_state_log_prob = state->log_prob * prev_normalizer;
        float tw = atomic_load_float(&t->weight);
        float base_score = unnorm_state_log_prob + (tw > 0.0f ? logf(tw) : -1000.0f);
        float constrained_score = base_score;

        /* * 2. Repetition Constraint (Problem #2 & #3 Fix)
         * Count frequency over recent window and apply penalty calculation once.
         */
        int repetition_count = 0;
        int rep_checks = (state->length < (uint32_t)HISTORY_WINDOW) ? (int)state->length : HISTORY_WINDOW;
        for (int j = 0; j < rep_checks; j++) {
          if (state->sequence[state->length - 1 - j] == t->target) {
            repetition_count++;
          }
        }
        float repetition_cost = (float)repetition_count * REPETITION_FACTOR;
        constrained_score -= repetition_cost;
        
        /* * 3. Short-Cycle Suppression (Problem #5 Fix)
         * Restrict loop detection to a local lookback window to catch dangerous attractor loops 
         * (e.g., A-B-A or A-B-C-A) without blocking long-distance word recurrences.
         */
        int loop_count = 0;
        int loop_checks = (state->length < (uint32_t)LOOP_WINDOW) ? (int)state->length : LOOP_WINDOW;
        for (int j = 0; j < loop_checks; j++) {
          if (state->sequence[state->length - 1 - j] == t->target) {
            loop_count++;
          }
        }
        float loop_cost = (float)loop_count * LOOP_FACTOR;
        constrained_score -= loop_cost;
        
        /* * 4. Entropy Regularization (Problem #4 Fix)
         * Centered Gaussian-like curve favoring moderate entropy. Penalizes both chaotic 
         * regions (extreme high uncertainty) and rigid deterministic trap loops (zero uncertainty).
         */
        Symbol *target_symbol = &ctx->dawg_nodes[t->target];
        SccNode *target_scc = &ctx->scc_nodes[target_symbol->original_scc];
        float normalized_entropy = scc_load_avg_entropy(target_scc);
        float entropy_distance = fabsf(normalized_entropy - TARGET_ENTROPY);
        float entropy_cost = ENTROPY_FACTOR * entropy_distance;
        constrained_score -= entropy_cost;
        
        /* * 5. Beam Diversity (Problem #6 Fix)
         * Compares trailing sequence similarity using a shared suffix overlap fraction 
         * against active alternative peer beam paths to prevent rapid sequence collapse.
         */
        float max_similarity = 0.0f;
        for (uint32_t k = 0; k < beam_size; k++) {
          if (k == i) continue; /* Skip the active parent index track */
          
          BeamState *peer = &beam[k];
          int match_count = 0;
          int max_possible_match = (int)peer->length;
          if (max_possible_match > (int)new_state->length) {
            max_possible_match = (int)new_state->length;
          }
          
          for (int m = 0; m < max_possible_match; m++) {
            SymbolID cand_tok = new_state->sequence[new_state->length - 1 - m];
            SymbolID peer_tok = peer->sequence[peer->length - 1 - m];
            if (cand_tok == peer_tok) {
              match_count++;
            } else {
              break; /* Suffix coherence boundary broken */
            }
          }
          
          if (max_possible_match > 0) {
            float sim = (float)match_count / (float)max_possible_match;
            if (sim > max_similarity) {
              max_similarity = sim;
            }
          }
        }
        float diversity_cost = max_similarity * DIVERSITY_FACTOR;
        constrained_score -= diversity_cost;

        /* * 6. Global Length Normalization Output (Problem #1 & #8 Fix)
         * Exactly ONE unified tracking value writes directly to the final log_prob state 
         * after being globally normalized by the sequence depth factor.
         */
        float length_normalizer = powf((float)new_state->length, LENGTH_ALPHA);
        new_state->log_prob = constrained_score / length_normalizer;
        
        /* Explicitly use acquire-release memory order boundaries to eliminate data races */
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
      
      /* Copy sequence data from scratch_next_seqs to scratch_beam_seqs for top-k results */
      for (uint32_t i = 0; i < beam_size; i++) {
        memcpy(scratch_beam_seqs + i * (max_depth + 1),
               next_beam[i].sequence,
               next_beam[i].length * sizeof(SymbolID));
        beam[i].sequence = &scratch_beam_seqs[i * (max_depth + 1)];
      }
    } else {
      break;
    }
  }

  /* * Decouple final states from scratch buffer tracking.
   * Allocates dedicated heap blocks per active result path so that clean external invocations 
   * of `le_free_beam_results` won't generate a catastrophic pointer double-free.
   */
  for (uint32_t i = 0; i < beam_size; i++) {
      SymbolID *final_seq = calloc(max_depth + 1, sizeof(SymbolID));
      if (final_seq) {
          memcpy(final_seq, beam[i].sequence, beam[i].length * sizeof(SymbolID));
          beam[i].sequence = final_seq;
      } else {
          /* Nullify to cleanly signal allocation failure to cleanup functions */
          beam[i].sequence = NULL;
          beam[i].length = 0;
      }
  }

  /* Safely release localized intermediate scratch matrices */
  free(next_beam);
  free(scratch_beam_seqs);
  free(scratch_next_seqs);
   
  *result_count = beam_size;
  return beam;
}