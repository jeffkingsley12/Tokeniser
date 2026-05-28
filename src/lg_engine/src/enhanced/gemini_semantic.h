/*
 * gemini_semantic.h — Embedding-based semantic SCC merging for Gemini
 *
 * Uses FastText quantized embeddings (from fasttext_embeddings.c) to
 * identify pairs of Gemini nodes whose words are semantically close
 * (high cosine similarity) AND share similar turbulence profiles.
 * Such pairs are merged into the same SCC even without a graph cycle,
 * producing a semantically organised DAWG where synonyms cluster into
 * the same symbol.
 *
 * Example merges:
 *   ssomero ↔ sukuulu  (both = "school")
 *   agenda  ↔ genda    (verb variants "goes / go")
 *   webale  ↔ weeba    (gratitude variants)
 */

#ifndef GEMINI_SEMANTIC_H
#define GEMINI_SEMANTIC_H

#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct EngineContext  EngineContext;
typedef struct Tokenizer      Tokenizer;
typedef struct EmbeddingModel EmbeddingModel;

/* ── Configuration ──────────────────────────────────────────────────────────── */

/*
 * Only consider pairs where:
 *   cosine_similarity(a, b) >= SEMANTIC_COSINE_THRESHOLD
 *   |turbulence(a) - turbulence(b)| <= SEMANTIC_TURB_THRESHOLD
 */
#define SEMANTIC_DEFAULT_COSINE_THRESHOLD  0.82f
#define SEMANTIC_DEFAULT_TURB_THRESHOLD    0.30f

/*
 * Maximum number of merge candidates evaluated per epoch.
 * Keeps the O(N²) pair scan bounded. 2000 → ~4M pair evals max.
 */
#define SEMANTIC_MAX_NODES_SCANNED  2000

/* ── Result ─────────────────────────────────────────────────────────────────── */

typedef struct {
    int nodes_scanned;       /* Nodes included in pair scan */
    int pairs_evaluated;     /* Total (i,j) pairs checked */
    int pairs_above_cosine;  /* Pairs passing cosine threshold */
    int merges_attempted;    /* Pairs also passing turbulence threshold */
    int merges_completed;    /* Successfully merged by le_merge_sccs */
    int merges_skipped;      /* Already in same SCC or one promoted */
} SemanticMergeResult;

/* ── API ────────────────────────────────────────────────────────────────────── */

/**
 * Run one semantic merge pass over the current graph.
 *
 * Scans at most SEMANTIC_MAX_NODES_SCANNED active nodes, evaluates all
 * O(N²) pairs, and calls le_merge_sccs() for pairs that pass both gates.
 *
 * Intended to be called once per epoch, after le_begin_epoch() and
 * before le_promote_eligible() / gemini_attest_promote().
 *
 * @param ctx            EngineContext
 * @param tok            Tokenizer (for word lookup)
 * @param embeddings     Loaded EmbeddingModel (from fasttext_embeddings.c)
 * @param cosine_thresh  Cosine similarity gate [0,1]
 * @param turb_thresh    Max |turbulence delta| for merge eligibility
 * @param result         Output stats; may be NULL
 * @return               Number of merges completed, or -1 on error
 */
int gemini_semantic_merge_pass(EngineContext         *ctx,
                               Tokenizer             *tok,
                               EmbeddingModel        *embeddings,
                               float                  cosine_thresh,
                               float                  turb_thresh,
                               SemanticMergeResult   *result);

/**
 * Compute cosine similarity between two words using the embedding model.
 * Returns value in [-1, 1]; 0 if either word is not in the model.
 * This is a utility exposed for testing/debugging.
 */
float gemini_semantic_cosine(EmbeddingModel *embeddings,
                             const char     *word_a,
                             const char     *word_b);

void gemini_semantic_result_print(const SemanticMergeResult *result);

#ifdef __cplusplus
}
#endif

#endif /* GEMINI_SEMANTIC_H */
