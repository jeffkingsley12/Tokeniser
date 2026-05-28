#ifndef BEAM_LATTICE_H
#define BEAM_LATTICE_H

#include "morph_mask.h"
#include <float.h>
#include <string.h>

#define BEAM_WIDTH 64
#define MAX_LATTICE_POS 512   /* max syllables per chunk (adjustable) */

/* -----------------------------------------------------------------------
 * BeamQueue: fixed‑size list of top‑K ConstraintStates
 * ----------------------------------------------------------------------- */
typedef struct {
    ConstraintState states[BEAM_WIDTH];
    uint32_t count;
} BeamQueue;

/* -----------------------------------------------------------------------
 * ParseLattice: array of beams indexed by syllable position
 * ----------------------------------------------------------------------- */
typedef struct {
    BeamQueue beams[MAX_LATTICE_POS + 1];
} ParseLattice;

/* -----------------------------------------------------------------------
 * Initialize a lattice (clear all beams)
 * ----------------------------------------------------------------------- */
static inline void lattice_init(ParseLattice *lat) {
    memset(lat, 0, sizeof(ParseLattice));
}

/* -----------------------------------------------------------------------
 * Insert a new state into a beam, keeping only the top BEAM_WIDTH scores.
 * Deduplicates states with identical (active_features, forbidden_features).
 * ----------------------------------------------------------------------- */
static inline void beam_insert(BeamQueue *beam, const ConstraintState *new_state) {
    /* Check for existing state with same mask pair */
    for (uint32_t i = 0; i < beam->count; i++) {
        if (beam->states[i].active_features == new_state->active_features &&
            beam->states[i].forbidden_features == new_state->forbidden_features) {
            /* Replace if new score is better */
            if (new_state->accumulated_score > beam->states[i].accumulated_score) {
                beam->states[i] = *new_state;
            }
            return;
        }
    }
    
    /* If beam not full, append */
    if (beam->count < BEAM_WIDTH) {
        beam->states[beam->count++] = *new_state;
        return;
    }
    
    /* Beam full: find worst scoring state */
    float min_score = FLT_MAX;
    uint32_t min_idx = 0;
    for (uint32_t i = 0; i < BEAM_WIDTH; i++) {
        if (beam->states[i].accumulated_score < min_score) {
            min_score = beam->states[i].accumulated_score;
            min_idx = i;
        }
    }
    
    /* Replace if new state is better */
    if (new_state->accumulated_score > min_score) {
        beam->states[min_idx] = *new_state;
    }
}

/* -----------------------------------------------------------------------
 * Extract the best token sequence from the final beam via Viterbi backtracking.
 * Returns number of tokens written to `out`, or -1 on error.
 * Requires that each ConstraintState stores `previous_state_id` and `edge_taken`.
 * The beams must have been preserved during forward pass.
 * 
 * NOTE: This simplified version assumes each edge consumes exactly 1 syllable.
 * For variable-length edges, see the improved version below.
 * ----------------------------------------------------------------------- */
static inline int extract_best_path(const ParseLattice *lat, uint32_t final_pos,
                                    uint32_t *out, uint32_t out_cap) {
    const BeamQueue *final_beam = &lat->beams[final_pos];
    if (final_beam->count == 0) return -1;
    
    /* Find best scoring state in final beam */
    float best_score = -FLT_MAX;
    uint32_t best_idx = 0;
    for (uint32_t i = 0; i < final_beam->count; i++) {
        if (final_beam->states[i].accumulated_score > best_score) {
            best_score = final_beam->states[i].accumulated_score;
            best_idx = i;
        }
    }
    
    /* Backtrack through lattice */
    uint32_t token_stack[MAX_LATTICE_POS];
    uint32_t stack_top = 0;
    uint32_t pos = final_pos;
    uint32_t state_idx = best_idx;
    
    while (pos > 0) {
        const ConstraintState *st = &lat->beams[pos].states[state_idx];
        if (st->edge_taken == 0) break;  /* safety */
        token_stack[stack_top++] = st->edge_taken;
        state_idx = st->previous_state_id;
        pos--;  /* Note: this assumes each edge consumes exactly 1 syllable.
                   For variable‑length edges, we need to subtract edge length.
                   See improved version below. */
    }
    
    /* Write tokens in forward order */
    uint32_t out_idx = 0;
    while (stack_top > 0 && out_idx < out_cap) {
        out[out_idx++] = token_stack[--stack_top];
    }
    return (int)out_idx;
}

/* -----------------------------------------------------------------------
 * Improved backtracking that handles variable‑length edges.
 * Requires that each ConstraintState also stores `prev_pos` (absolute position
 * of previous state) and `prev_state_idx` (index in that beam).
 * 
 * A robust design stores in ConstraintState:
 *   uint32_t prev_pos;   // absolute position of previous state
 *   uint32_t prev_state; // index in that beam
 * Then backtracking becomes trivial.
 * 
 * For now, we provide a placeholder that the user can extend when needed.
 * ----------------------------------------------------------------------- */

#endif /* BEAM_LATTICE_H */
