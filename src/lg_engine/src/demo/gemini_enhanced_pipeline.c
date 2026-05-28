/*
 * gemini_enhanced_pipeline.c — Full end-to-end pipeline demo
 *
 * Demonstrates all six enhancement subsystems working together:
 *
 *   Stage 1: Build ngrams.bin from corpus         (luganda_train)
 *   Stage 2: Create enhanced engine               (gemini_enhanced_create)
 *   Stage 3: Ingest text through pipeline         (gemini_enhanced_process_text)
 *   Stage 4: Run enhanced epoch                   (gemini_enhanced_epoch)
 *   Stage 5: Query predictions                    (le_beam_search)
 *   Stage 6: Evaluate accuracy                    (gemini_enhanced_eval)
 *   Stage 7: Save model                           (gemini_enhanced_save)
 *
 * Build:
 *   gcc -std=c11 -O2 -Iinclude -Ienhanced -Iluganda \
 *       gemini_enhanced_pipeline.c \
 *       -Lbin -lgemini_enhanced -lm -lpthread \
 *       -o bin/gemini_pipeline
 *
 * Run:
 *   ./bin/gemini_pipeline corpus.txt [test_cases.csv]
 */

#include "gemini_enhanced.h"
#include "../validation/gemini_eval.h"
#include "libgemini.h"
#include "gemini_internal.h"
#include "gemini_ngram_prior.h"
#ifdef GEMINI_FEAT_SEMANTIC
#include "gemini_semantic.h"
#endif

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>

/* ── Helper: print prediction results ──────────────────────────────────────── */

static void print_beam_results(EngineContext *ctx, Tokenizer *tok,
                                BeamState *beam, uint32_t count,
                                const char *context_desc) {
    printf("  Predictions after '%s':\n", context_desc);

    if (!beam || count == 0) {
        printf("    (no predictions — more epochs needed)\n");
        return;
    }

    int shown = 0;
    for (uint32_t i = 0; i < count && shown < 5; i++) {
        BeamState *bs = &beam[i];
        if (!bs->sequence || bs->length == 0) continue;

        /* Resolve first symbol to a word */
        SymbolID sym = bs->sequence[0];
        NodeID   member_buf[32];
        uint32_t mc = le_get_symbol_nodes(ctx, sym, member_buf, 32);

        for (uint32_t m = 0; m < mc && shown < 5; m++) {
            TokenID    tid = get_node_token(ctx, member_buf[m]);
            const char *w  = tokenizer_get_word(tok, tid);
            if (!w) continue;
            printf("    %d. %-20s  (prob=%.4f)\n",
                   shown + 1, w, expf(bs->log_prob));
            shown++;
            break;   /* One word per beam candidate */
        }
    }

    if (shown == 0) {
        printf("    (candidates exist but symbols not yet resolved)\n");
    }
}

/* ── Stage 1: Corpus ingestion ──────────────────────────────────────────────── */

static int ingest_corpus(GeminiEnhanced *ge, const char *corpus_path,
                          int epoch_interval) {
    FILE *fp = fopen(corpus_path, "r");
    if (!fp) {
        fprintf(stderr, "Cannot open corpus: %s\n", corpus_path);
        return -1;
    }

    char      line[4096];
    int       lines    = 0;
    int       epochs   = 0;
    NodeID    prev     = LE_INVALID;
    clock_t   t_start  = clock();

    printf("Ingesting corpus: %s\n", corpus_path);

    while (fgets(line, sizeof(line), fp)) {
        /* Strip newline */
        char *nl = strchr(line, '\n');
        if (nl) *nl = '\0';

        if (strlen(line) < 2) { prev = LE_INVALID; continue; }

        prev = gemini_enhanced_process_text(ge, line, prev);
        lines++;

        /* Periodic epoch */
        if (epoch_interval > 0 && lines % epoch_interval == 0) {
            uint32_t promoted = gemini_enhanced_epoch(ge);
            epochs++;
            printf("  [Epoch %d | line %d] promotions: %u\n",
                   epochs, lines, promoted);
        }
    }

    /* Final epoch */
    uint32_t promoted = gemini_enhanced_epoch(ge);
    epochs++;
    printf("  [Epoch %d | FINAL | line %d] promotions: %u\n",
           epochs, lines, promoted);

    fclose(fp);

    double elapsed = (double)(clock() - t_start) / CLOCKS_PER_SEC;
    printf("Corpus ingested: %d lines, %d epochs, %.2fs\n\n",
           lines, epochs, elapsed);
    return 0;
}

/* ── Stage 5: Sample predictions ────────────────────────────────────────────── */

static void run_sample_predictions(GeminiEnhanced *ge) {
    EngineContext *ctx = gemini_enhanced_ctx(ge);
    Tokenizer     *tok = gemini_enhanced_tokenizer(ge);

    if (!ctx || !tok) {
        printf("(tokenizer not available for prediction display)\n");
        return;
    }

    /* Test pairs: context word → beam search */
    const char *test_words[] = {
        "oli",       /* "oli otya" (greeting) */
        "omwana",    /* "omwana omulungi" (beautiful child) */
        "webale",    /* "webale nnyo" (thank you very much) */
        "agenda",    /* verb continuation */
        "buli",      /* "buli lunaku" (every day) */
        NULL
    };

    printf("═══════════════════════════════════════════════════════════\n");
    printf("  SAMPLE PREDICTIONS\n");
    printf("═══════════════════════════════════════════════════════════\n\n");

    for (int i = 0; test_words[i]; i++) {
        TokenID    tid   = tokenizer_get_id(tok, test_words[i], false);
        if (tid == LE_INVALID) {
            printf("  '%s' — not in vocabulary\n\n", test_words[i]);
            continue;
        }

        uint32_t   count = 0;
        BeamState *beam  = le_beam_search(ctx, tid, 5, 3, &count);
        print_beam_results(ctx, tok, beam, count, test_words[i]);
        le_free_beam_results(beam, count);
        printf("\n");
    }
}

/* ── Main ────────────────────────────────────────────────────────────────────── */

int main(int argc, char *argv[]) {
    if (argc < 2) {
        printf("Usage: %s <corpus.txt> [test_cases.csv] [words.dawg] "
               "[ngrams.bin] [embeddings.bin]\n", argv[0]);
        printf("\nAll arguments after corpus.txt are optional.\n");
        return 1;
    }

    const char *corpus_path = argv[1];
    const char *test_csv    = argc >= 3 ? argv[2] : NULL;
    const char *dict_path   = argc >= 4 ? argv[3] : "words.dawg";
    const char *ngram_path  = argc >= 5 ? argv[4] : "ngrams.bin";
#ifdef GEMINI_FEAT_SEMANTIC
    const char *embed_path  = argc >= 6 ? argv[5] : "embeddings.bin";
#endif

    printf("\n");
    printf("╔═══════════════════════════════════════════════════════════╗\n");
    printf("║       GEMINI ENHANCED ENGINE — FULL PIPELINE              ║\n");
    printf("╚═══════════════════════════════════════════════════════════╝\n\n");

    /* ── Build configuration ── */
    EnhancedConfig cfg = gemini_enhanced_default_config();

    /* Phonology: always on */
    cfg.phon_enabled = true;

    /* Morphology: always on */
    cfg.morph_enabled             = true;
    cfg.morph_stem_confidence_min = 0.70f;

    /* Phrase seeding: always on */
    cfg.phrase_seed_enabled      = true;
    cfg.phrase_seed_base_weight  = 5.0f;

    /* N-gram prior: on if ngrams.bin exists */
    {
        FILE *f = fopen(ngram_path, "rb");
        if (f) {
            fclose(f);
            cfg.ngram_prior_enabled = true;
            cfg.ngram_prior_path    = ngram_path;
            cfg.ngram_prior_scale   = NGRAM_PRIOR_DEFAULT_SCALE;
            printf("N-gram prior: %s\n", ngram_path);
        } else {
            printf("N-gram prior: NOT FOUND at '%s' (skipping cold-start init)\n",
                   ngram_path);
        }
    }

    /* Attestation: on if words.dawg exists */
    {
        FILE *f = fopen(dict_path, "r");
        if (f) {
            fclose(f);
            cfg.attest_enabled        = true;
            cfg.attest_wordlist_path  = dict_path;
            cfg.attest_min_ratio      = 0.60f;
            printf("Attestation gate: %s\n", dict_path);
        } else {
            printf("Attestation gate: NOT FOUND at '%s' (all SCCs can promote)\n",
                   dict_path);
        }
    }

    /* Semantic merging: on if embeddings.bin exists */
#ifdef GEMINI_FEAT_SEMANTIC
    {
        FILE *f = fopen(embed_path, "rb");
        if (f) {
            fclose(f);
            cfg.semantic_enabled          = true;
            cfg.embedding_path            = embed_path;
            cfg.semantic_cosine_threshold = SEMANTIC_DEFAULT_COSINE_THRESHOLD;
            cfg.semantic_turb_threshold   = SEMANTIC_DEFAULT_TURB_THRESHOLD;
            printf("Semantic merging: %s\n", embed_path);
        } else {
            printf("Semantic merging: NOT FOUND at '%s' (cosine-merge disabled)\n",
                   embed_path);
        }
    }
#else
    printf("Semantic merging: FEATURE DISABLED at compile time\n");
#endif

    printf("\n");

    /* ── Create engine ── */
    printf("─── Stage 2: Creating enhanced engine ───────────────────────\n");
    GeminiEnhanced *ge = gemini_enhanced_create(&cfg);
    if (!ge) {
        fprintf(stderr, "FATAL: Failed to create enhanced engine\n");
        return 1;
    }
    printf("\n");

    /* ── Ingest corpus ── */
    printf("─── Stage 3: Ingesting corpus ───────────────────────────────\n");
    if (ingest_corpus(ge, corpus_path, 5000) != 0) {
        fprintf(stderr, "Corpus ingestion failed\n");
        gemini_enhanced_destroy(ge);
        return 1;
    }

    /* ── Final stats ── */
    EngineContext *ctx = gemini_enhanced_ctx(ge);
    if (ctx) {
        printf("Engine state after ingestion:\n");
        printf("  Nodes     : %u\n", get_node_count(ctx));
        printf("  Edges     : %u\n", get_edge_count(ctx));
        printf("  SCCs      : %u\n", get_scc_count(ctx));
        printf("  Symbols   : %u\n", get_symbol_count(ctx));
        printf("  Entropy   : %.4f\n", le_get_global_entropy(ctx));
        printf("\n");
    }

    /* ── Sample predictions ── */
    printf("─── Stage 5: Sample predictions ─────────────────────────────\n");
    run_sample_predictions(ge);

    /* ── Evaluation ── */
    if (test_csv) {
        printf("─── Stage 6: Accuracy evaluation ────────────────────────────\n");
        GeminiEvalReport report;
        if (gemini_enhanced_eval(ge, test_csv, &report) == 0) {
            gemini_eval_report_print(&report);
            if (gemini_eval_report_save_json(&report, "eval_results.json") == 0)
                printf("Results saved to eval_results.json\n");
            if (gemini_eval_save_baseline(&report, "baseline.bin") == 0)
                printf("Baseline saved to baseline.bin\n");
        }
        printf("\n");
    }

    /* ── Save model ── */
    printf("─── Stage 7: Saving model ───────────────────────────────────\n");
    if (gemini_enhanced_save(ge, "model.gemini"))
        printf("Model saved to model.gemini\n\n");

    /* ── Cleanup ── */
    gemini_enhanced_destroy(ge);

    printf("╔═══════════════════════════════════════════════════════════╗\n");
    printf("║  Pipeline complete.                                        ║\n");
    printf("╚═══════════════════════════════════════════════════════════╝\n\n");

    return 0;
}
