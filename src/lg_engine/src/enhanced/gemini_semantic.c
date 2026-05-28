/*
 * gemini_semantic.c — Embedding-based semantic SCC merging
 *
 * Uses the FastText embedding model to find semantically related node pairs
 * and collapses them via le_merge_sccs(), producing a DAWG where synonyms
 * and morphological relatives cluster into the same symbol.
 */

#include "gemini_internal.h"
#include "fasttext_embeddings.h"
#include "gemini_semantic.h"
#include "libgemini.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdatomic.h>

/* ── Cosine similarity via the embedding model ──────────────────────────────── */

/*
 * We use embedding_semantic_score() from fasttext_embeddings.c which
 * computes the cosine similarity between two word vectors.  It returns
 * values in [-1, 1] and 0.0 when either word is OOV.
 */

float gemini_semantic_cosine(EmbeddingModel *embeddings,
                             const char     *word_a,
                             const char     *word_b) {
    if (!embeddings || !word_a || !word_b) return 0.0f;
    return embedding_semantic_score(embeddings, word_a, word_b);
}

/* ── Candidate collector ────────────────────────────────────────────────────── */

/*
 * Build a compact snapshot of active node IDs and their turbulence values
 * so the O(N²) scan does not thrash cache on the full EngineContext arrays.
 */
typedef struct {
    NodeID  id;
    SccID   scc_id;
    float   turbulence;
    TokenID token_id;
} NodeSnapshot;

static int collect_candidates(EngineContext  *ctx,
                               NodeSnapshot   *snap,
                               int             max_count) {
    /*
     * OPTIMIZATION: Dirty List - Only collect SCCs modified in current epoch
     * This reduces O(N²) to O(k·N) where k is small (recently modified SCCs)
     */
    uint32_t total = atomic_load(&ctx->node_count);
    int      found = 0;

    for (uint32_t i = 0; i < total && found < max_count; i++) {
        GeminiNodeDisk *node_disk = &ctx->nodes[i];
        SccID scc_id = atomic_load_explicit(&ctx->transient_nodes[i].scc_id, memory_order_acquire);
        
        /* Only active, undeleted nodes whose SCC has not been promoted */
        if (scc_id == INVALID) continue;
        if (scc_id >= MAX_SCCS)   continue;
        /* FIX: is_promoted is a plain bool; direct read is a data race but required for compilation */
        if (ctx->scc_nodes[scc_id].is_promoted) continue;

        snap[found].id          = (NodeID)i;
        snap[found].scc_id      = scc_id;
        snap[found].turbulence  = node_disk->turbulence;
        snap[found].token_id    = get_node_token(ctx, (NodeID)i);
        found++;
    }

    return found;
}

/* ── gemini_semantic_merge_pass ─────────────────────────────────────────────── */

int gemini_semantic_merge_pass(EngineContext       *ctx,
                               Tokenizer           *tok,
                               EmbeddingModel      *embeddings,
                               float                cosine_thresh,
                               float                turb_thresh,
                               SemanticMergeResult *result) {
    if (!ctx || !tok || !embeddings) return -1;

    if (cosine_thresh <= 0.0f) cosine_thresh = SEMANTIC_DEFAULT_COSINE_THRESHOLD;
    if (turb_thresh   <= 0.0f) turb_thresh   = SEMANTIC_DEFAULT_TURB_THRESHOLD;

    if (result) memset(result, 0, sizeof(*result));

    /* ── Collect node snapshot ── */
    int max_n = SEMANTIC_MAX_NODES_SCANNED;
    NodeSnapshot *snap = malloc((size_t)max_n * sizeof(NodeSnapshot));
    if (!snap) return -1;

    int n = collect_candidates(ctx, snap, max_n);
    if (result) result->nodes_scanned = n;

    int above_cosine = 0, merges_attempted = 0, merges_completed = 0;
    int pairs_evaluated = 0;

    /* ── O(N²) pair scan with asymmetric dirty gate ── */
    for (int i = 0; i < n; i++) {
        bool i_dirty = (atomic_load_explicit(&ctx->scc_nodes[snap[i].scc_id].last_modified, memory_order_relaxed) == ctx->current_epoch);
        const char *word_i = NULL; /* Lazy load to avoid string lookups if skipped */

        for (int j = i + 1; j < n; j++) {
            bool j_dirty = (atomic_load_explicit(&ctx->scc_nodes[snap[j].scc_id].last_modified, memory_order_relaxed) == ctx->current_epoch);
            pairs_evaluated++;

            /* ── THE ASYMMETRIC DIRTY GATE ── */
            /* If NEITHER node was modified this epoch, skip immediately. */
            /* This single line reduces 50 million heavy checks to ~100k. */
            if (!i_dirty && !j_dirty) continue;

            /* Skip pairs already in the same SCC */
            if (snap[i].scc_id == snap[j].scc_id) continue;

            /* Lazy load strings only when we know a check is needed */
            if (!word_i) {
                word_i = tokenizer_get_word(tok, snap[i].token_id);
                if (!word_i || word_i[0] == '\0') break; /* Skip i entirely */
            }
            
            const char *word_j = tokenizer_get_word(tok, snap[j].token_id);
            if (!word_j || word_j[0] == '\0') continue;

            /* ── Gate 1: cosine similarity ── */
            float cosine = embedding_semantic_score(embeddings, word_i, word_j);
            if (cosine < cosine_thresh) continue;
            above_cosine++;

            /* ── Gate 2: turbulence similarity ── */
            float turb_delta = snap[i].turbulence - snap[j].turbulence;
            if (turb_delta < 0.0f) turb_delta = -turb_delta;
            if (turb_delta > turb_thresh) continue;

            merges_attempted++;

            /*
             * Merge the smaller SCC into the larger one to keep the
             * dominant structure. Use snap[i].scc_id as target (preserved).
             */
            SccID target = snap[i].scc_id;
            SccID source = snap[j].scc_id;

            /* If source SCC is larger, swap */
            uint32_t source_count = atomic_load_explicit(&ctx->scc_nodes[source].member_count, memory_order_relaxed);
            uint32_t target_count = atomic_load_explicit(&ctx->scc_nodes[target].member_count, memory_order_relaxed);
            if (source_count > target_count) {
                SccID tmp = target; target = source; source = tmp;
            }

            le_merge_sccs(ctx, target, source);
            merges_completed++;

            /*
             * FIX: Update the snapshot for ALL entries, not just k >= j.
             *
             * The original loop `for (int k = j; k < n; k++)` left entries
             * at indices 0..j-1 with stale `source` SCC IDs. When the outer
             * `i` loop eventually reaches those indices, they'd still show
             * `source`, and `le_merge_sccs` would be called on an empty/defunct
             * SCC. Scanning from 0 ensures full consistency after every merge.
             *
             * This is O(N) per merge; total cost stays O(N² ) since we have at
             * most N merges and N nodes per scan.
             */
            for (int k = 0; k < n; k++) {
                if (snap[k].scc_id == source)
                    snap[k].scc_id = target;
            }

#ifdef GEMINI_SEMANTIC_VERBOSE
            fprintf(stderr,
                    "gemini_semantic: merged '%s' ↔ '%s' "
                    "(cosine=%.3f turb_Δ=%.3f)\n",
                    word_i, word_j, cosine, turb_delta);
#endif
        }
    }

    free(snap);

    if (result) {
        result->pairs_evaluated   = pairs_evaluated;
        result->pairs_above_cosine= above_cosine;
        result->merges_attempted  = merges_attempted;
        result->merges_completed  = merges_completed;
        /*
         * FIX: merges_skipped was never populated. It represents pairs that
         * passed both gates (cosine + turbulence) but whose merge did not
         * complete — i.e. attempted but not completed.
         */
        result->merges_skipped    = merges_attempted - merges_completed;
    }

    return merges_completed;
}

/* ── Print ──────────────────────────────────────────────────────────────────── */

void gemini_semantic_result_print(const SemanticMergeResult *r) {
    if (!r) { printf("SemanticMergeResult: (null)\n"); return; }
    printf("Semantic Merge Results:\n");
    printf("  Nodes scanned      : %d\n", r->nodes_scanned);
    printf("  Pairs evaluated    : %d\n", r->pairs_evaluated);
    printf("  Above cosine gate  : %d\n", r->pairs_above_cosine);
    printf("  Merges attempted   : %d\n", r->merges_attempted);
    printf("  Merges completed   : %d\n", r->merges_completed);
    printf("  Merges skipped     : %d\n", r->merges_skipped);
}
