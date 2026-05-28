/*
 * gemini_eval.h — Evaluation harness for the enhanced Gemini engine
 *
 * Adapts the Luganda eval framework (luganda_eval.c) to measure the
 * accuracy of Gemini beam-search predictions against a labelled test set.
 *
 * CSV format (one row per test case):
 *   context_words, prefix, expected_word, category, difficulty
 *
 * Where:
 *   context_words  — space-separated prior words (e.g. "omwana omulungi")
 *   prefix         — partial word being typed (e.g. "a" or "" for next-word)
 *   expected_word  — ground-truth next word
 *   category       — free label ("honorific", "verb", etc.)
 *   difficulty     — float 0-1 (optional; 0.5 if missing)
 *
 * Metrics produced:
 *   Top-1, Top-3, Top-5 accuracy
 *   MRR (Mean Reciprocal Rank)
 *   Average prediction score
 *   Per-category breakdown
 */

#ifndef GEMINI_EVAL_H
#define GEMINI_EVAL_H

#include <stdint.h>
#include <time.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct EngineContext EngineContext;
typedef struct Tokenizer     Tokenizer;

/* ── Test case ──────────────────────────────────────────────────────────────── */

#define GEMINI_EVAL_MAX_CONTEXT  8
#define GEMINI_EVAL_MAX_WORD_LEN 128
#define GEMINI_EVAL_MAX_CAT_LEN  64

typedef struct {
    char    context[GEMINI_EVAL_MAX_CONTEXT][GEMINI_EVAL_MAX_WORD_LEN];
    int     context_size;
    char    prefix[GEMINI_EVAL_MAX_WORD_LEN];
    char    expected[GEMINI_EVAL_MAX_WORD_LEN];
    char    category[GEMINI_EVAL_MAX_CAT_LEN];
    float   difficulty;
} GeminiTestCase;

/* ── Test suite ─────────────────────────────────────────────────────────────── */

typedef struct {
    GeminiTestCase *cases;
    int             count;
    int             capacity;
} GeminiTestSuite;

GeminiTestSuite *gemini_eval_suite_create(int initial_capacity);
void             gemini_eval_suite_free(GeminiTestSuite *suite);

/**
 * Load test cases from a CSV file.
 * Returns number of cases loaded, or -1 on error.
 */
int gemini_eval_suite_load_csv(GeminiTestSuite *suite, const char *csv_path);

/* ── Per-category stats ─────────────────────────────────────────────────────── */

#define GEMINI_EVAL_MAX_CATEGORIES  32

typedef struct {
    char  name[GEMINI_EVAL_MAX_CAT_LEN];
    int   total;
    int   top1;
    int   top3;
    int   top5;
    float mrr_sum;
} CategoryStats;

/* ── Evaluation report ──────────────────────────────────────────────────────── */

typedef struct {
    /* Aggregate */
    int    total_cases;
    int    top1_hits;
    int    top3_hits;
    int    top5_hits;
    float  mrr;
    float  avg_score;

    /* Per-category */
    CategoryStats categories[GEMINI_EVAL_MAX_CATEGORIES];
    int           category_count;

    /* Timing */
    double elapsed_sec;

    /* Beam parameters used */
    uint32_t beam_width;
    uint32_t max_depth;
} GeminiEvalReport;

/* ── Runner ─────────────────────────────────────────────────────────────────── */

/**
 * Run the evaluation suite against the engine.
 *
 * For each test case:
 *   1. Convert context words to TokenIDs
 *   2. Run le_beam_search from the last context token with given prefix filter
 *   3. Check if expected_word appears in top-K results
 *   4. Record hit/miss, rank, and score
 *
 * Prefix filtering: candidates from beam search are filtered to those
 * whose word starts with `prefix` (empty prefix = accept all).
 *
 * @param ctx        EngineContext
 * @param tok        Tokenizer
 * @param suite      Test suite
 * @param beam_width Beam width for search
 * @param max_depth  Max beam depth
 * @param report     Output report (must not be NULL)
 * @return           0 on success
 */
int gemini_eval_run(EngineContext    *ctx,
                    Tokenizer        *tok,
                    GeminiTestSuite  *suite,
                    uint32_t          beam_width,
                    uint32_t          max_depth,
                    GeminiEvalReport *report);

/**
 * Print a human-readable evaluation report to stdout.
 */
void gemini_eval_report_print(const GeminiEvalReport *report);

/**
 * Save report as JSON to a file.
 * Returns 0 on success.
 */
int gemini_eval_report_save_json(const GeminiEvalReport *report,
                                 const char             *path);

/**
 * Save report as a baseline binary blob for regression testing.
 * Returns 0 on success.
 */
int gemini_eval_save_baseline(const GeminiEvalReport *report, const char *path);

/**
 * Load a previously saved baseline.
 * Returns allocated report on success; caller must free().
 */
GeminiEvalReport *gemini_eval_load_baseline(const char *path);

/**
 * Detect regressions: returns the number of categories where
 * top1 accuracy dropped by more than threshold (e.g. 0.05 = 5%).
 */
int gemini_eval_detect_regression(const GeminiEvalReport *current,
                                  const GeminiEvalReport *baseline,
                                  float                   threshold);

#ifdef __cplusplus
}
#endif

#endif /* GEMINI_EVAL_H */
