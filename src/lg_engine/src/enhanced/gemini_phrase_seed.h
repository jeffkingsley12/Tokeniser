/*
 * gemini_phrase_seed.h — Phrase registry seeding for Gemini
 *
 * Pre-seeds Gemini with high-confidence Luganda phrase pairs from the
 * phrase registry, bypassing the normal epoch-count promotion requirement
 * for culturally critical fixed phrases ("oli otya", "webale nnyo", etc.).
 *
 * Seeds are marked is_forced=true in their SccNode so they survive
 * even in early low-data epochs.
 */

#ifndef GEMINI_PHRASE_SEED_H
#define GEMINI_PHRASE_SEED_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct EngineContext EngineContext;
typedef struct Tokenizer     Tokenizer;

/* ── Built-in phrase table ──────────────────────────────────────────────────── */

/*
 * Phrase categories mirroring luganda_phrase_registry.h
 * (duplicated here so this header compiles standalone).
 */
typedef enum {
    PHRASE_CAT_CONNECTORS  = 0,
    PHRASE_CAT_TEMPORAL,
    PHRASE_CAT_HONORIFICS,
    PHRASE_CAT_AUXILIARIES,
    PHRASE_CAT_IDIOMS,
    PHRASE_CAT_NUMERALS,
    PHRASE_CAT_QUESTION,
    PHRASE_CAT_NEGATION,
    PHRASE_CAT_COUNT
} PhraseSeedCategory;

typedef struct {
    const char       *first_word;
    const char       *second_word;
    float             bonus_multiplier;  /* > 1.0 → extra edge weight */
    PhraseSeedCategory category;
    int               rank;             /* Lower = more common */
} PhraseSeedEntry;

/* ── Result ─────────────────────────────────────────────────────────────────── */

typedef struct {
    int phrases_seeded;
    int nodes_created;
    int edges_created;
    int forced_promotions;   /* SCCs immediately promoted (rank <= threshold) */
} PhraseSeedResult;

/* ── API ────────────────────────────────────────────────────────────────────── */

/**
 * Seed Gemini with the built-in Luganda phrase table.
 *
 * For every (w1, w2) pair:
 *   - Creates/gets TokenIDs and NodeIDs for w1 and w2
 *   - Injects base_weight × bonus_multiplier edge calls
 *   - Marks the resulting SCC is_forced=true for phrases with rank <= force_rank_threshold
 *
 * @param ctx                Initialised EngineContext
 * @param tok                Tokenizer
 * @param base_weight        Base edge weight per phrase (e.g. 5.0)
 * @param force_rank_threshold  Phrases with rank <= this are force-promoted (e.g. 5)
 * @param result             Output stats; may be NULL
 * @return                   Number of phrases seeded, or -1 on fatal error
 */
int gemini_phrase_seed(EngineContext     *ctx,
                       Tokenizer         *tok,
                       float              base_weight,
                       int                force_rank_threshold,
                       PhraseSeedResult  *result);

/**
 * Seed from a custom phrase table instead of the built-in one.
 * entries must be terminated by an entry with first_word == NULL.
 */
int gemini_phrase_seed_custom(EngineContext        *ctx,
                              Tokenizer            *tok,
                              const PhraseSeedEntry *entries,
                              float                 base_weight,
                              int                   force_rank_threshold,
                              PhraseSeedResult     *result);

void gemini_phrase_seed_print(const PhraseSeedResult *result);

#ifdef __cplusplus
}
#endif

#endif /* GEMINI_PHRASE_SEED_H */
