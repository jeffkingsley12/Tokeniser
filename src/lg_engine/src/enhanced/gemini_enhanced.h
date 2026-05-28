/*
 * gemini_enhanced.h — Master header for Luganda-aware Gemini enhancement layer
 *
 * Drop-in over the base Gemini engine. Include this instead of libgemini.h
 * when building with the enhancement layer enabled.
 *
 * Compile-time feature flags (define before including this header, or via -D):
 *   GEMINI_FEAT_MORPH       - Morphological stem canonicalization
 *   GEMINI_FEAT_PHON        - Phonological normalization in tokenizer
 *   GEMINI_FEAT_NGRAM_PRIOR - N-gram model as edge weight prior
 *   GEMINI_FEAT_PHRASE_SEED - Phrase registry forced seeding
 *   GEMINI_FEAT_ATTEST      - Attestation gating on SCC promotion
 *   GEMINI_FEAT_SEMANTIC    - Embedding-based semantic SCC merging
 *   GEMINI_FEAT_EVAL        - Offline evaluation harness
 *
 * Enable all features at once:
 *   #define GEMINI_ALL_FEATURES
 */

#ifndef GEMINI_ENHANCED_H
#define GEMINI_ENHANCED_H

#ifdef __cplusplus
extern "C" {
#endif

#include "libgemini.h"
#include <stdbool.h>
#include <stdint.h>

/* ── Feature shorthand ─────────────────────────────────────────────────────── */
#ifdef GEMINI_ALL_FEATURES
#  define GEMINI_FEAT_MORPH
#  define GEMINI_FEAT_PHON
#  define GEMINI_FEAT_NGRAM_PRIOR
#  define GEMINI_FEAT_PHRASE_SEED
#  define GEMINI_FEAT_ATTEST
#  define GEMINI_FEAT_SEMANTIC
#  define GEMINI_FEAT_EVAL
#endif

/* ── Enhancement configuration ─────────────────────────────────────────────── */
typedef struct {
    /* Morphology */
    bool     morph_enabled;
    float    morph_stem_confidence_min; /* 0-1; stems below this are ignored */

    /* Phonology */
    bool     phon_enabled;

    /* N-gram prior */
    bool     ngram_prior_enabled;
    const char *ngram_prior_path;       /* Path to ngrams.bin */
    float    ngram_prior_scale;         /* Weight multiplier (default 8.0) */

    /* Phrase seeder */
    bool     phrase_seed_enabled;
    float    phrase_seed_base_weight;   /* Initial edge weight for phrases */

    /* Attestation */
    bool     attest_enabled;
    const char *attest_wordlist_path;   /* Path to word list (one per line) */
    float    attest_min_ratio;          /* Fraction of SCC that must be attested */

    /* Semantic merging */
    bool     semantic_enabled;
    const char *embedding_path;         /* Path to embeddings.bin */
    float    semantic_cosine_threshold; /* 0-1; pairs above this may merge */
    float    semantic_turb_threshold;   /* Max turbulence delta to allow merge */

    /* Eval */
    bool     eval_enabled;
    const char *eval_test_csv;          /* Path to test_cases.csv */
} EnhancedConfig;

/* Sensible defaults */
EnhancedConfig gemini_enhanced_default_config(void);

/* ── Opaque handle ──────────────────────────────────────────────────────────── */
typedef struct GeminiEnhanced GeminiEnhanced;

/* ── Lifecycle ──────────────────────────────────────────────────────────────── */

/**
 * Create an enhanced engine from config.
 * Internally calls le_init(), loads all requested subsystems, and returns
 * a GeminiEnhanced wrapper that the caller owns.
 *
 * Returns NULL on fatal error (OOM or mandatory subsystem failure).
 */
GeminiEnhanced *gemini_enhanced_create(const EnhancedConfig *cfg);

/**
 * Destroy the enhanced engine and all subsystems.
 * Also calls le_destroy() on the inner EngineContext.
 */
void gemini_enhanced_destroy(GeminiEnhanced *ge);

/* ── Token ingestion (replaces le_process_token) ───────────────────────────── */

/**
 * Process a raw UTF-8 word through the full enhancement pipeline:
 *   phonology → morphology → canonicalization → graph ingestion
 *
 * Returns the canonical NodeID used internally.
 */
NodeID gemini_enhanced_process_word(GeminiEnhanced *ge,
                                    const char *word,
                                    NodeID prev_node);

/**
 * Process a raw UTF-8 text string (multi-word, sentence etc.)
 * Handles splitting, phonology, morphology, and graph ingestion.
 * Returns the last NodeID produced.
 */
NodeID gemini_enhanced_process_text(GeminiEnhanced *ge,
                                    const char *text,
                                    NodeID prev_node);

/* ── Epoch management (replaces le_begin_epoch) ─────────────────────────────── */

/**
 * Run one enhanced epoch:
 *   1. le_begin_epoch (flush deferred edges, stability counters)
 *   2. Semantic merge pass (if enabled)
 *   3. le_promote_eligible with attestation gating (if enabled)
 *   4. DAWG transition rebuild
 *
 * Returns number of promotions in this epoch.
 */
uint32_t gemini_enhanced_epoch(GeminiEnhanced *ge);

/* ── Accessors to inner engine ──────────────────────────────────────────────── */
EngineContext *gemini_enhanced_ctx(GeminiEnhanced *ge);
struct Tokenizer *gemini_enhanced_tokenizer(GeminiEnhanced *ge);
struct GeminiAttestDB *gemini_enhanced_attest_db(GeminiEnhanced *ge);

/* ── Save / Load ────────────────────────────────────────────────────────────── */
bool gemini_enhanced_save(GeminiEnhanced *ge, const char *path);
GeminiEnhanced *gemini_enhanced_load(const char *path,
                                     const EnhancedConfig *cfg);

/* ── Evaluation ─────────────────────────────────────────────────────────────── */
#include "../validation/gemini_eval.h"

/**
 * Run evaluation suite from CSV and fill *report.
 * CSV format: context_words(space-sep),prefix,expected_word,category,difficulty
 * Returns 0 on success.
 */
int gemini_enhanced_eval(GeminiEnhanced *ge,
                         const char *csv_path,
                         GeminiEvalReport *report);

#ifdef __cplusplus
}
#endif

#endif /* GEMINI_ENHANCED_H */
