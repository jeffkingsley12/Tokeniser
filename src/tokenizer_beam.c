/*
 * tokenizer_beam.c
 *
 * Beam search tokenizer with morphological constraints for Luganda.
 * Implements Positional Viterbi Lattice to handle variable-length edges
 * and morphological constraint validation using bitwise operations.
 */

#include "tokenizer.h"
#include "morph_constraints.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ========================================================================
 * tokenizer_encode_beam
 *
 * Positional Viterbi lattice with O(1) bitwise constraint filtering.
 *
 * Returns:  >=0  token count
 *           -1   fatal error (no parse, malformed input)
 *           -2   output buffer too small (truncation)
 * ========================================================================
 */
int tokenizer_encode_beam(const Tokenizer *t, const char *text,
                          uint32_t *out, uint32_t out_cap)
{
    if (!t || !text || !out) return -1;

    uint16_t syls[MAX_SYLLABLES];
    int n_syls = syllabify(t->syl, text, syls, MAX_SYLLABLES);
    if (n_syls <= 0) return 0;
    if ((uint32_t)n_syls > MAX_LATTICE_POS) return -1;

    const LOUDS *l = t->louds;
    if (!l || !l->has_csr) return -1;
    if (!t->token_features || !t->token_requires || !t->token_forbids)
        return -1;   /* masks not built — caller should fall back */

    ParseLattice lattice;
    memset(&lattice, 0, sizeof(lattice));

    /* Seed beam 0 */
    lattice.beams[0].count = 1;
    lattice.beams[0].states[0] = (ConstraintState){
        .active_features = 0, .forbidden_features = 0,
        .accumulated_score = 0.0f,
        .backpointer_slot = 0, .backpointer_pos = 0,
        .token_id = 0, .generation = 0
    };
    lattice.backpointers[0][0] = (ViterbiBackpointer){
        .prev_pos = 0, .prev_slot = 0, .token_id = 0,
        .score = 0.0f, .generation = 0
    };

    /* ==================================================================
     * FORWARD SWEEP
     * ================================================================== */
    for (uint32_t pos = 0; pos < (uint32_t)n_syls; pos++) {
        BeamQueue *beam = &lattice.beams[pos];
        if (beam->count == 0) continue;

        for (uint32_t s = 0; s < beam->count; s++) {
            ConstraintState *state = &beam->states[s];

            uint32_t node = 0;
            uint32_t i = pos;

            while (i < (uint32_t)n_syls) {
                uint16_t target = syls[i];
                uint32_t start  = l->row_ptr[node];
                uint32_t end    = l->row_ptr[node + 1];

                /* Binary search on CSR edges */
                uint32_t lo = start, hi = end;
                while (lo < hi) {
                    uint32_t mid = (lo + hi) >> 1;
                    if (l->labels[mid] < target) lo = mid + 1;
                    else hi = mid;
                }
                if (lo >= end || l->labels[lo] != target) break;

                node = l->next_node[lo];
                i++;

                if (l->terminals[node] == 0) continue;

                uint32_t token_id = l->terminals[node] - 1;
                if (token_id >= t->vocab_size) continue;

                MorphEdge edge = {
                    .features     = t->token_features[token_id],
                    .requires     = t->token_requires[token_id],
                    .forbids      = t->token_forbids[token_id],
                    .token_id     = token_id,
                    .syllable_len = i - pos,
                    .log_prob     = 0.0f
                };

                if (!edge_is_valid(state, &edge)) continue;

                ConstraintState ns = transition_state(state, &edge, 0.0f, s, pos, token_id);
                uint32_t slot = beam_insert(&lattice.beams[i], &ns, &lattice, i);

                /* If beam_insert rejected this state (score too low),
                 * we simply continue — no backpointer was recorded. */
                (void)slot;
            }
        }
    }

    /* ==================================================================
     * BACKTRACK through backpointers[][] — NEVER touches beam[] states
     * ================================================================== */
    BeamQueue *final = &lattice.beams[n_syls];
    if (final->count == 0)
        return -1;   /* no complete parse */

    uint32_t best_slot = 0;
    float best_score = final->states[0].accumulated_score;
    for (uint32_t i = 1; i < final->count; i++) {
        if (final->states[i].accumulated_score > best_score) {
            best_score = final->states[i].accumulated_score;
            best_slot = i;
        }
    }

    uint32_t rev[MAX_LATTICE_POS];
    uint32_t rev_len = 0;
    uint32_t pos = (uint32_t)n_syls;
    uint32_t slot = best_slot;
    uint32_t safety = 0;

    while (pos > 0 && safety++ < MAX_LATTICE_POS * 2) {
        ViterbiBackpointer *bp = &lattice.backpointers[pos][slot];

        /* Generation check: detect stale backpointer from beam eviction */
        if (bp->generation != lattice.beams[pos].states[slot].generation) {
            fprintf(stderr, "[beam] generation mismatch at pos=%u slot=%u\n", pos, slot);
            return -1;
        }

        if (bp->token_id != 0)   /* skip dummy root */
            rev[rev_len++] = bp->token_id;

        uint32_t next_pos  = bp->prev_pos;
        uint32_t next_slot = bp->prev_slot;

        if (next_pos >= pos && next_pos != 0) {
            fprintf(stderr, "[beam] backtrack did not make progress\n");
            return -1;
        }
        pos  = next_pos;
        slot = next_slot;
    }

    if (pos != 0) {
        fprintf(stderr, "[beam] backtrack failed to reach root\n");
        return -1;
    }

    if (rev_len > out_cap) return -2;   /* truncation */

    for (uint32_t i = 0; i < rev_len; i++)
        out[i] = rev[rev_len - 1 - i];

    return (int)rev_len;
}
