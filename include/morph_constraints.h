#pragma once
#include <stdint.h>
#include <float.h>
#include <stdbool.h>

#define BEAM_WIDTH       64
#define MAX_LATTICE_POS  512   /* syllables; 512 × 64 states ≈ 32 KiB cache */

typedef uint64_t MorphMask;

/* -------------------------------------------------------------------------
 * Feature / role bit definitions (tune to your linguistic model)
 * ------------------------------------------------------------------------- */
#define FEAT_NOUN_CLASS_1    (1ULL << 0)
#define FEAT_NOUN_CLASS_2    (1ULL << 1)
#define FEAT_NOUN_CLASS_3    (1ULL << 2)
#define FEAT_NOUN_CLASS_4    (1ULL << 3)
#define FEAT_NOUN_CLASS_5    (1ULL << 4)
#define FEAT_NOUN_CLASS_6    (1ULL << 5)
#define FEAT_NOUN_CLASS_7    (1ULL << 6)
#define FEAT_NOUN_CLASS_8    (1ULL << 7)
#define FEAT_NOUN_CLASS_9    (1ULL << 8)
#define FEAT_NOUN_CLASS_10   (1ULL << 9)
#define FEAT_NOUN_CLASS_11   (1ULL << 10)
#define FEAT_NOUN_CLASS_13   (1ULL << 11)
#define FEAT_NOUN_CLASS_18   (1ULL << 12)
#define FEAT_NOUN_CLASS_22   (1ULL << 13)
#define FEAT_VERB_ROOT       (1ULL << 14)
#define FEAT_PRESENT_TENSE   (1ULL << 15)
#define FEAT_NEGATION        (1ULL << 16)
#define FEAT_IMPERATIVE_MOOD (1ULL << 17)
#define FEAT_INFINITIVE      (1ULL << 18)   /* class 15 */
#define FEAT_LOCATIVE        (1ULL << 19)
#define FEAT_DIMINUTIVE      (1ULL << 20)
#define FEAT_ABSTRACT        (1ULL << 21)
#define FEAT_AUGMENTATIVE    (1ULL << 22)

/* Structural roles (used in requires / forbids logic) */
#define ROLE_PREFIX          (1ULL << 32)
#define ROLE_ROOT            (1ULL << 33)
#define ROLE_SUFFIX          (1ULL << 34)
#define ROLE_PARTICLE        (1ULL << 35)

/* -------------------------------------------------------------------------
 * 40-byte immutable edge template (cache-line friendly)
 * ------------------------------------------------------------------------- */
typedef struct {
    MorphMask features;   /* 8 */
    MorphMask requires;   /* 8 */
    MorphMask forbids;    /* 8 */
    uint32_t  token_id;   /* 4 */
    uint32_t  syllable_len;/* 4 — how many syllables this edge consumes */
    float     log_prob;   /* 4 — higher is better (log-frequency) */
    uint32_t  reserved;   /* 4 — explicit padding */
} MorphEdge;

/* -------------------------------------------------------------------------
 * Dynamic path state (lives in the beam, not in the graph)
 * ------------------------------------------------------------------------- */
typedef struct {
    MorphMask active_features;
    MorphMask forbidden_features;
    float     accumulated_score;
    uint32_t  backpointer_slot;   /* index of parent state in beams[prev_pos] */
    uint32_t  backpointer_pos;    /* which beam the parent lives in */
    uint32_t  token_id;           /* token emitted by the edge that created this state */
    uint32_t  generation;         /* monotonic counter for stale backpointer detection */
} ConstraintState;

typedef struct {
    ConstraintState states[BEAM_WIDTH];
    uint32_t count;
} BeamQueue;

/* -------------------------------------------------------------------------
 * Viterbi backpointer: immutable record of how a state was reached
 * ------------------------------------------------------------------------- */
typedef struct {
    uint32_t prev_pos;      /* source beam position */
    uint32_t prev_slot;     /* source beam slot */
    uint32_t token_id;      /* token emitted on this edge */
    float    score;         /* accumulated score at this point */
    uint32_t generation;    /* matches state->generation for validation */
} ViterbiBackpointer;

typedef struct {
    BeamQueue beams[MAX_LATTICE_POS + 1];
    ViterbiBackpointer backpointers[MAX_LATTICE_POS + 1][BEAM_WIDTH];
} ParseLattice;

/* -------------------------------------------------------------------------
 * O(1) gatekeeper (inlined into the forward sweep)
 * ------------------------------------------------------------------------- */
static inline bool edge_is_valid(const ConstraintState *state,
                                 const MorphEdge *edge)
{
    if (__builtin_expect((edge->requires & ~state->active_features) != 0, 0))
        return false;                         /* missing prerequisite         */
    if (__builtin_expect((edge->forbids & state->active_features) != 0, 0))
        return false;                         /* edge forbids active history  */
    if (__builtin_expect((state->forbidden_features & edge->features) != 0, 0))
        return false;                         /* history forbids edge offer   */
    return true;
}

static inline ConstraintState transition_state(const ConstraintState *prev,
                                               const MorphEdge *edge,
                                               float score,
                                               uint32_t parent_state_idx,
                                               uint32_t parent_pos,
                                               uint32_t token_id)
{
    return (ConstraintState){
        .active_features    = prev->active_features    | edge->features,
        .forbidden_features = prev->forbidden_features | edge->forbids,
        .accumulated_score  = prev->accumulated_score  + score,
        .backpointer_slot   = parent_state_idx,
        .backpointer_pos    = parent_pos,
        .token_id           = token_id,
        .generation         = prev->generation + 1
    };
}

/* -------------------------------------------------------------------------
 * Fixed-width beam insert: O(K) with K = 64. No mallocs, no pointers.
 * Records backpointer for Viterbi backtrack.
 * ------------------------------------------------------------------------- */
static inline uint32_t beam_insert(BeamQueue *beam, const ConstraintState *st,
                                   ParseLattice *lattice, uint32_t pos)
{
    if (__builtin_expect(beam->count < BEAM_WIDTH, 1)) {
        uint32_t slot = beam->count++;
        beam->states[slot] = *st;
        lattice->backpointers[pos][slot] = (ViterbiBackpointer){
            .prev_pos = st->backpointer_pos,
            .prev_slot = st->backpointer_slot,
            .token_id = st->token_id,
            .score = st->accumulated_score,
            .generation = st->generation
        };
        return slot;
    }

    uint32_t min_idx = 0;
    float    min_score = beam->states[0].accumulated_score;

    for (uint32_t i = 1; i < BEAM_WIDTH; i++) {
        if (beam->states[i].accumulated_score < min_score) {
            min_score = beam->states[i].accumulated_score;
            min_idx   = i;
        }
    }

    if (st->accumulated_score > min_score) {
        beam->states[min_idx] = *st;   /* evict weakest hypothesis */
        lattice->backpointers[pos][min_idx] = (ViterbiBackpointer){
            .prev_pos = st->backpointer_pos,
            .prev_slot = st->backpointer_slot,
            .token_id = st->token_id,
            .score = st->accumulated_score,
            .generation = st->generation
        };
        return min_idx;
    }

    return BEAM_WIDTH;  /* rejected */
}
