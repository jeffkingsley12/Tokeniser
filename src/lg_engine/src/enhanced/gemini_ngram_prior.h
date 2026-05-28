/*
 * gemini_ngram_prior.h — N-gram prior initialisation for Gemini
 *
 * Loads the Luganda n-gram model (ngrams.bin) produced by luganda_train and
 * uses bigram/trigram probabilities to pre-initialise Gemini edge weights.
 * This collapses the cold-start period: the engine has linguistically
 * informed turbulence values from the very first token.
 *
 * File format expected (written by ngram_trainer_save):
 *   Header:  Magic(4) "LNGR" | Version(4) | Count(4)
 *   Entries: Size(1) | Frequency(4) | [Len(2) Word]* × Size
 */

#ifndef GEMINI_NGRAM_PRIOR_H
#define GEMINI_NGRAM_PRIOR_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Forward declarations matching libgemini.h / gemini_internal.h */
typedef struct EngineContext EngineContext;
typedef struct Tokenizer     Tokenizer;

/* ── Configuration ──────────────────────────────────────────────────────────── */

#define NGRAM_PRIOR_FILE_MAGIC   0x47524e4cU   /* "LNGR" — matches luganda_train.c */
#define NGRAM_PRIOR_MAX_ORDER    5
#define NGRAM_PRIOR_MAX_WORD_LEN 128

/* Default weight scale: bigram prob × NGRAM_PRIOR_SCALE = initial edge weight */
#define NGRAM_PRIOR_DEFAULT_SCALE  8.0f

/* ── Result ─────────────────────────────────────────────────────────────────── */

typedef struct {
    int     bigrams_loaded;      /* Total bigram entries read from file */
    int     trigrams_loaded;     /* Total trigram entries read from file */
    int     edges_created;       /* New Gemini edges created */
    int     edges_boosted;       /* Existing edges whose weight was raised */
    int     nodes_created;       /* New Gemini nodes created for OOV priors */
    float   scale_used;          /* Effective scale factor applied */
} NgramPriorResult;

/* ── API ────────────────────────────────────────────────────────────────────── */

/**
 * Load ngrams.bin and inject bigram/trigram probabilities into Gemini as
 * pre-initialised edge weights.
 *
 * @param ctx        Initialised EngineContext (from le_init)
 * @param tok        Tokenizer that maps words → TokenIDs
 * @param path       Path to ngrams.bin produced by luganda_train
 * @param scale      Weight multiplier (use NGRAM_PRIOR_DEFAULT_SCALE or 0 for default)
 * @param result     Output statistics; may be NULL
 * @return           0 on success, -1 on file error, -2 on format mismatch
 */
int gemini_load_ngram_prior(EngineContext     *ctx,
                            Tokenizer         *tok,
                            const char        *path,
                            float              scale,
                            NgramPriorResult  *result);

/**
 * Print a human-readable summary of an NgramPriorResult.
 * Safe to call with result == NULL.
 */
void gemini_ngram_prior_print(const NgramPriorResult *result);

#ifdef __cplusplus
}
#endif

#endif /* GEMINI_NGRAM_PRIOR_H */
