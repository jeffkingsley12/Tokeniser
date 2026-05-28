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

/* -------------------------------------------------------------------------
 * Forward sweep + Viterbi backtrack.
 * Returns number of tokens written, or -1 on failure / truncation.
 * ------------------------------------------------------------------------- */
int tokenizer_encode_beam(const Tokenizer *t, const char *text,
                          uint32_t *out, uint32_t out_cap)
{
    if (!t || !text || !out) return -1;

    uint16_t syls[MAX_SYLLABLES];
    int n_syls = syllabify(t->syl, text, syls, MAX_SYLLABLES);
    if (n_syls <= 0) return 0;
    if ((uint32_t)n_syls > MAX_LATTICE_POS) return -1;   /* chunk too long */

    const LOUDS *l = t->louds;
    if (!l || !l->has_csr) return -1;

    ParseLattice *lattice = calloc(1, sizeof(ParseLattice));
    if (!lattice) return -1;

    /* Seed beam 0 with the empty parse */
    lattice->beams[0].count = 1;
    lattice->beams[0].states[0] = (ConstraintState){
        .active_features    = 0,
        .forbidden_features = 0,
        .accumulated_score  = 0.0f,
        .previous_state_id  = 0xFFFFFFFF,
        .prev_pos           = 0,
        .token_id           = 0
    };

    /* ==================================================================
     * FORWARD SWEEP
     * ================================================================== */
    for (uint32_t pos = 0; pos < (uint32_t)n_syls; pos++) {
        BeamQueue *beam = &lattice->beams[pos];
        if (beam->count == 0) continue;   /* dead syllable boundary */

        for (uint32_t s = 0; s < beam->count; s++) {
            ConstraintState *state = &beam->states[s];

            /* Walk the DAWG starting at `pos`, emitting every terminal we hit */
            uint32_t node = 0;
            uint32_t i    = pos;

            while (i < (uint32_t)n_syls) {
                uint16_t target = syls[i];
                uint32_t start  = l->row_ptr[node];
                uint32_t end    = l->row_ptr[node + 1];
                uint32_t next   = 0xFFFFFFFF;

                /* Binary search on outbound CSR edges (same as louds_tokenize) */
                uint32_t lo = start, hi = end;
                while (lo < hi) {
                    uint32_t mid = (lo + hi) >> 1;
                    if (l->labels[mid] < target) lo = mid + 1;
                    else hi = mid;
                }
                if (lo < end && l->labels[lo] == target)
                    next = l->next_node[lo];

                if (next == 0xFFFFFFFF) break;
                node = next;
                i++;

                /* If this node is terminal, we have a candidate token */
                if (l->terminals[node] != 0) {
                    uint32_t token_id = l->terminals[node] - 1;
                    if (token_id >= t->vocab_size) continue;

                    /* Skip tokens that have no constraint data (mask == 0).
                     * They are always valid and carry no score. */
                    MorphEdge edge = {
                        .features     = t->token_features[token_id],
                        .requires     = t->token_requires[token_id],
                        .forbids      = t->token_forbids[token_id],
                        .token_id     = token_id,
                        .syllable_len = i - pos,
                        .log_prob     = 0.0f   /* TODO: score from rule freq */
                    };

                    if (!edge_is_valid(state, &edge)) continue;

                    ConstraintState ns = transition_state(state, &edge, edge.log_prob, s, pos, token_id);
                    beam_insert(&lattice->beams[i], &ns);
                }
            }
        }
    }

    /* ==================================================================
     * BACKTRACK (Viterbi)
     * ================================================================== */
    BeamQueue *final = &lattice->beams[n_syls];
    if (final->count == 0) {
        /* No complete parse survived constraints — fallback to greedy */
        free(lattice);
        return louds_tokenize(t->louds, syls, (uint32_t)n_syls, out, out_cap);
    }

    /* Find best-scoring terminal state */
    uint32_t best_idx   = 0;
    float    best_score = final->states[0].accumulated_score;
    for (uint32_t i = 1; i < final->count; i++) {
        if (final->states[i].accumulated_score > best_score) {
            best_score = final->states[i].accumulated_score;
            best_idx   = i;
        }
    }

    /* Reconstruct token sequence by chasing backpointers */
    uint32_t rev[MAX_LATTICE_POS];
    uint32_t rev_len = 0;
    uint32_t pos     = (uint32_t)n_syls;
    uint32_t sid     = best_idx;

    while (pos > 0) {
        ConstraintState *st = &lattice->beams[pos].states[sid];
        rev[rev_len++] = st->token_id;
        sid = st->previous_state_id;
        pos = st->prev_pos;
    }

    free(lattice);

    if (rev_len > out_cap) return -1;   /* truncation */

    for (uint32_t i = 0; i < rev_len; i++)
        out[i] = rev[rev_len - 1 - i];

    return (int)rev_len;
}
