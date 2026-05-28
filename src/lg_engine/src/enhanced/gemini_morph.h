/*
 * gemini_morph.h — Morphology-aware node canonicalization for Gemini
 *
 * Plugs into le_get_or_create_node() to map morphological variants
 * (e.g. omwana, abana, omwana's) to the same canonical stem node,
 * dramatically reducing graph fragmentation.
 */

#ifndef GEMINI_MORPH_H
#define GEMINI_MORPH_H

#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Maximum word and stem lengths */
#define MORPH_MAX_WORD_LEN   128
#define MORPH_MAX_STEM_LEN   64
#define MORPH_MAX_PREFIX_LEN 8
#define MORPH_MAX_SUFFIX_LEN 8

/* ── Noun classes (Luganda) ─────────────────────────────────────────────────── */
typedef enum {
    NC_UNKNOWN = 0,
    NC_1_MU,   /* omu- / omw-  singular person */
    NC_2_BA,   /* aba- / abw-  plural person   */
    NC_3_MU,   /* omu- / omw-  singular tree   */
    NC_4_MI,   /* emi- / em-   plural tree     */
    NC_5_KI,   /* e-            singular thing  */
    NC_6_BI,   /* ama- / am-   plural thing    */
    NC_7_ERI,  /* eki- / eky-  singular lang   */
    NC_8_BI,   /* ebi- / eby-  plural lang     */
    NC_9_N,    /* en- / em-    singular animal */
    NC_10_N,   /* (same prefix, plural)        */
    NC_11_KU,  /* olu- / olw-  singular        */
    NC_12_AKA, /* aka-          diminutive sg   */
    NC_13_OBU, /* obu- / otu-  diminutive pl   */
    NC_14_BU,  /* obu- / obw-  abstract        */
    NC_15_OKU, /* oku- / okw-  infinitive      */
    NC_20_GA,  /* (locative)                   */
    NC_COUNT
} NounClass;

typedef enum {
    NUM_UNKNOWN = 0,
    NUM_SINGULAR,
    NUM_PLURAL
} GrammaticalNumber;

typedef enum {
    POS_UNKNOWN = 0,
    POS_NOUN,
    POS_VERB,
    POS_ADJECTIVE,
    POS_ADVERB,
    POS_CONJUNCTION,
    POS_PRONOUN
} PartOfSpeech;

/* ── Analysis result ────────────────────────────────────────────────────────── */
typedef struct {
    const char     *word;           /* Original word (not owned) */
    const char     *prefix;         /* Matched prefix string (static) */
    const char     *stem;           /* Pointer into word[] past prefix */
    char            canonical[MORPH_MAX_STEM_LEN]; /* Normalized stem copy */
    NounClass       noun_class;
    GrammaticalNumber number;
    PartOfSpeech    pos;
    NounClass       subject_class;  /* For verbs: subject agreement class */
    float           confidence;     /* 0-1 */
} MorphInfo;

/* ── Variant-to-canonical mapping table ─────────────────────────────────────── */
/*
 * Side table that stores the full surface form for each canonical stem ID.
 * This is separate from the TST tokenizer so that:
 *   - The graph collapses variants onto one node (canonical stem)
 *   - The display layer can still recover the original word from the token ID
 */
#define MORPH_VARIANT_TABLE_SIZE 65537  /* Prime > typical vocab */

typedef struct MorphVariantEntry {
    char                  surface[MORPH_MAX_WORD_LEN];  /* e.g. "omwana" */
    char                  canonical[MORPH_MAX_STEM_LEN]; /* e.g. "wana"  */
    uint32_t              canonical_token_id;
    struct MorphVariantEntry *next;
} MorphVariantEntry;

typedef struct {
    MorphVariantEntry *buckets[MORPH_VARIANT_TABLE_SIZE];
    int                entry_count;
} MorphVariantTable;

/* ── Public API ─────────────────────────────────────────────────────────────── */

/**
 * Analyze a word and fill *out_info with the morphological breakdown.
 *
 * The function operates entirely on the caller's stack — no heap allocation
 * is performed and no free() is required.  out_info->stem and out_info->prefix
 * point into static storage; out_info->canonical is a NUL-terminated copy of
 * the normalised stem inside the struct itself.
 *
 * @param word      NUL-terminated UTF-8 word to analyse (not modified)
 * @param out_info  Caller-allocated MorphInfo to populate
 * @return          true on success (including the fallback/no-match path);
 *                  false only if word or out_info is NULL/empty
 */
bool morph_analyze(const char *word, MorphInfo *out_info);

/**
 * Extract the canonical stem from a word.
 * If morphology analysis succeeds with confidence >= min_confidence,
 * writes the canonical stem into out_stem (max_len bytes) and returns true.
 * Otherwise copies the original word unchanged and returns false.
 *
 * out_stem must be at least MORPH_MAX_STEM_LEN bytes.
 */
bool morph_get_canonical_stem(const char *word,
                              float       min_confidence,
                              char       *out_stem,
                              int         max_len);

/**
 * Guess noun class from prefix only (no full analysis).
 * Used for OOV words. Returns NC_UNKNOWN if no match.
 */
NounClass morph_guess_class_from_prefix(const char *word);

/**
 * Validated prefix class guess that also sets a confidence score.
 */
NounClass morph_guess_class_from_prefix_validated(const char *word,
                                                   float      *confidence);

/**
 * Create a variant→canonical side table.
 */
MorphVariantTable *morph_variant_table_create(void);

/**
 * Register a surface→canonical mapping.
 * Returns the canonical token id that was already assigned, or 0 if new.
 */
uint32_t morph_variant_table_register(MorphVariantTable *vt,
                                      const char        *surface,
                                      const char        *canonical,
                                      uint32_t           canonical_token_id);

/**
 * Look up the canonical token id for a surface form.
 * Returns 0 if not found.
 */
uint32_t morph_variant_table_lookup(const MorphVariantTable *vt,
                                    const char              *surface);

/**
 * Look up a canonical string for a token id (reverse).
 * Returns NULL if not found.
 * Caller must NOT free the returned pointer.
 */
const char *morph_variant_table_reverse(const MorphVariantTable *vt,
                                        uint32_t                 canonical_id);

/**
 * Destroy a variant table and free all memory.
 */
void morph_variant_table_destroy(MorphVariantTable *vt);

#ifdef __cplusplus
}
#endif

#endif /* GEMINI_MORPH_H */
