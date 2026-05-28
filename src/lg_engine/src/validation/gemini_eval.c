/*
 * gemini_eval.c — Evaluation harness for the enhanced Gemini engine
 *
 * Loads a CSV test suite, runs le_beam_search for each case, measures
 * Top-1/3/5 accuracy and MRR, and generates per-category breakdowns.
 */

/* Must precede all system includes for clock_gettime / CLOCK_MONOTONIC */
#ifndef _GNU_SOURCE
#  define _GNU_SOURCE
#endif

#include "gemini_eval.h"
#include "libgemini.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>
#include <time.h>

/* ── Baseline file magic ────────────────────────────────────────────────────── */
#define EVAL_BASELINE_MAGIC   0x47455641u   /* "GEVA" */
#define EVAL_BASELINE_VERSION 1u

/* ── Suite lifecycle ────────────────────────────────────────────────────────── */

GeminiTestSuite *gemini_eval_suite_create(int initial_capacity) {
    if (initial_capacity <= 0) initial_capacity = 256;
    GeminiTestSuite *s = calloc(1, sizeof(GeminiTestSuite));
    if (!s) return NULL;
    s->cases    = calloc((size_t)initial_capacity, sizeof(GeminiTestCase));
    s->capacity = s->cases ? initial_capacity : 0;
    if (!s->cases) { free(s); return NULL; }
    return s;
}

void gemini_eval_suite_free(GeminiTestSuite *s) {
    if (!s) return;
    free(s->cases);
    free(s);
}

/* ── CSV parser ─────────────────────────────────────────────────────────────── */

/*
 * CSV format (comma-separated, no quoting needed for Luganda):
 *   context_words, prefix, expected_word, category, difficulty
 *
 * context_words uses a space-separated list within the first field.
 */

static char *trim(char *s) {
    while (*s == ' ' || *s == '\t') s++;
    char *end = s + strlen(s) - 1;
    while (end > s && (*end == ' ' || *end == '\t' ||
                       *end == '\r' || *end == '\n'))
        *end-- = '\0';
    return s;
}

static int split_csv_line(char *line, char *fields[], int max_fields) {
    int n = 0;
    char *p = line;
    while (n < max_fields) {
        fields[n++] = p;
        char *comma  = strchr(p, ',');
        if (!comma) break;
        *comma = '\0';
        p = comma + 1;
    }
    return n;
}

int gemini_eval_suite_load_csv(GeminiTestSuite *suite, const char *path) {
    if (!suite || !path) return -1;

    FILE *fp = fopen(path, "r");
    if (!fp) {
        fprintf(stderr, "gemini_eval: cannot open '%s'\n", path);
        return -1;
    }

    char line[1024];
    int  loaded = 0;

    /* Skip header */
    if (!fgets(line, sizeof(line), fp)) { fclose(fp); return 0; }

    while (fgets(line, sizeof(line), fp)) {
        /* H-3 FIX: Use size_t arithmetic to prevent signed overflow when
         * suite->capacity >= INT_MAX/2+1. A negative newcap cast to size_t
         * passes a huge size to realloc, returns NULL, and loses all loaded data. */
        if (suite->count >= suite->capacity) {
            size_t newcap = (suite->capacity > 0) ? (size_t)suite->capacity * 2 : 256;
            if (newcap > (size_t)INT_MAX / sizeof(GeminiTestCase))
                newcap = (size_t)INT_MAX / sizeof(GeminiTestCase);
            GeminiTestCase *nb = realloc(suite->cases, newcap * sizeof(GeminiTestCase));
            if (!nb) break;
            suite->cases    = nb;
            suite->capacity = (int)newcap;
        }

        char *fields[5] = {0};
        char  buf[1024];
        snprintf(buf, sizeof(buf), "%s", line);
        int nf = split_csv_line(buf, fields, 5);
        if (nf < 3) continue;

        GeminiTestCase *tc = &suite->cases[suite->count];
        memset(tc, 0, sizeof(*tc));

        /* Field 0: context words (space-separated) */
        char *ctx_str = trim(fields[0]);
        char *sp = ctx_str;
        tc->context_size = 0;
        while (*sp && tc->context_size < GEMINI_EVAL_MAX_CONTEXT) {
            char *end = strchr(sp, ' ');
            if (end) *end = '\0';
            snprintf(tc->context[tc->context_size++], GEMINI_EVAL_MAX_WORD_LEN, "%s", sp);
            if (!end) break;
            sp = end + 1;
            while (*sp == ' ') sp++;
        }

        /* Field 1: prefix */
        snprintf(tc->prefix, GEMINI_EVAL_MAX_WORD_LEN, "%s", trim(fields[1]));

        /* Field 2: expected word */
        snprintf(tc->expected, GEMINI_EVAL_MAX_WORD_LEN, "%s", trim(fields[2]));

        /* Field 3: category (optional) */
        if (nf >= 4)
            snprintf(tc->category, GEMINI_EVAL_MAX_CAT_LEN, "%s", trim(fields[3]));
        else
            snprintf(tc->category, GEMINI_EVAL_MAX_CAT_LEN, "%s", "general");

        /* Field 4: difficulty (optional) */
        if (nf >= 5) {
            char *endp = NULL;
            double d = strtod(trim(fields[4]), &endp);
            tc->difficulty = (endp != fields[4] && *endp == '\0') ? (float)d : 0.5f;
        } else {
            tc->difficulty = 0.5f;
        }

        suite->count++;
        loaded++;
    }

    fclose(fp);
    return loaded;
}

/* ── Category helpers ───────────────────────────────────────────────────────── */

static int find_or_add_category(GeminiEvalReport *r, const char *name) {
    for (int i = 0; i < r->category_count; i++) {
        if (strncmp(r->categories[i].name, name, GEMINI_EVAL_MAX_CAT_LEN) == 0)
            return i;
    }
    if (r->category_count >= GEMINI_EVAL_MAX_CATEGORIES) return -1;
    int idx = r->category_count++;
    snprintf(r->categories[idx].name, GEMINI_EVAL_MAX_CAT_LEN, "%s", name);
    return idx;
}

/* ── Beam result prefix filter ──────────────────────────────────────────────── */

static int beam_word_starts_with(const char *word, const char *prefix) {
    if (!prefix || prefix[0] == '\0') return 1;   /* Empty prefix: accept all */
    size_t plen = strlen(prefix);
    return strncmp(word, prefix, plen) == 0;
}

/* ── gemini_eval_run ────────────────────────────────────────────────────────── */

int gemini_eval_run(EngineContext    *ctx,
                    Tokenizer        *tok,
                    GeminiTestSuite  *suite,
                    uint32_t          beam_width,
                    uint32_t          max_depth,
                    GeminiEvalReport *report) {
    if (!ctx || !tok || !suite || !report) return -1;

    memset(report, 0, sizeof(*report));
    report->beam_width = beam_width;
    report->max_depth  = max_depth;

    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);

    float total_rr    = 0.0f;
    float total_score = 0.0f;

    for (int ci = 0; ci < suite->count; ci++) {
        GeminiTestCase *tc = &suite->cases[ci];
        report->total_cases++;

        /* Resolve start token: last word in context */
        TokenID start_token = LE_INVALID;
        if (tc->context_size > 0) {
            const char *last_ctx = tc->context[tc->context_size - 1];
            start_token = tokenizer_get_id(tok, last_ctx, false);
        }
        /* M-3 FIX: The tokenizer returns 0 as "not found" and UINT32_MAX (== LE_INVALID)
         * on allocation failure. Token ID 0 is the TST sentinel and is never assigned to
         * any real word, so checking == 0 is correct today — but also guard UINT32_MAX
         * so both error paths are handled uniformly, and the code survives if the
         * tokenizer's not-found sentinel changes in a future revision. */
        if (start_token == 0 || start_token == LE_INVALID) continue;

        /* Run beam search */
        uint32_t   result_count = 0;
        BeamState *beam = le_beam_search(ctx, start_token,
                                         beam_width, max_depth, &result_count);
        if (!beam || result_count == 0) {
            le_free_beam_results(beam, result_count);
            continue;
        }

        /*
         * Each BeamState has a sequence of SymbolIDs.
         * To compare against expected_word, we resolve the FIRST symbol in each
         * candidate to its member token strings, then filter by prefix.
         *
         * filtered_rank counts how many beam results (not member nodes) passed
         * the prefix filter before the expected word was found. This gives a
         * rank in the filtered candidate list, suitable for Top-1/3/5 and MRR.
         */

        int rank_found    = 0;  /* 1-based rank in filtered candidate list; 0 = not found */
        int filtered_rank = 0;  /* How many prefix-matching beam results seen so far */

        for (uint32_t r = 0; r < result_count && rank_found == 0; r++) {
            BeamState *bs = &beam[r];
            if (!bs->sequence || bs->length == 0) continue;

            /*
             * Resolve first symbol → member nodes → token strings.
             * Check whether ANY member of this beam result matches the prefix.
             * If so, this result occupies one rank slot in the filtered list.
             * If the expected word is among those members, record the rank.
             */
            SymbolID   sym = bs->sequence[0];
            NodeID     member_nodes[64];
            uint32_t   member_count = le_get_symbol_nodes(ctx, sym,
                                                           member_nodes, 64);

            bool this_result_matches_prefix = false;
            bool expected_found_here        = false;

            for (uint32_t m = 0; m < member_count; m++) {
                TokenID    tid = get_node_token(ctx, member_nodes[m]);
                const char *w  = tokenizer_get_word(tok, tid);
                if (!w) continue;
                if (!beam_word_starts_with(w, tc->prefix)) continue;

                /* This beam result contributes at least one prefix-matching candidate */
                this_result_matches_prefix = true;

                if (strcmp(w, tc->expected) == 0) {
                    expected_found_here = true;
                    total_score += expf(bs->log_prob);
                    break;
                }
            }

            if (this_result_matches_prefix) {
                filtered_rank++;
                if (expected_found_here) {
                    rank_found = filtered_rank;
                }
            }
        }

        le_free_beam_results(beam, result_count);

        /* Accumulate metrics */
        if (rank_found == 1)                    { report->top1_hits++; }  /* FIX: was >= 1, counted every rank as top-1 */
        if (rank_found >= 1 && rank_found <= 3) { report->top3_hits++; }
        if (rank_found >= 1 && rank_found <= 5) { report->top5_hits++; }

        float rr = (rank_found > 0) ? 1.0f / (float)rank_found : 0.0f;
        total_rr += rr;

        /* Per-category */
        int cat_idx = find_or_add_category(report, tc->category);
        if (cat_idx >= 0) {
            CategoryStats *cs = &report->categories[cat_idx];
            cs->total++;
            cs->mrr_sum += rr;
            if (rank_found == 1)                    cs->top1++;
            if (rank_found >= 1 && rank_found <= 3) cs->top3++;
            if (rank_found >= 1 && rank_found <= 5) cs->top5++;
        }
    }

    /* Aggregate */
    int n = report->total_cases;
    report->mrr       = (n > 0) ? total_rr    / (float)n : 0.0f;
    report->avg_score = (n > 0) ? total_score / (float)n : 0.0f;

    clock_gettime(CLOCK_MONOTONIC, &t1);
    report->elapsed_sec = (double)(t1.tv_sec  - t0.tv_sec) +
                          (double)(t1.tv_nsec - t0.tv_nsec) * 1e-9;

    return 0;
}

/* ── Print ──────────────────────────────────────────────────────────────────── */

void gemini_eval_report_print(const GeminiEvalReport *r) {
    if (!r) return;
    int n = r->total_cases;

    printf("\n══════════════════════════════════════════════════════\n");
    printf("  GEMINI ENHANCED — EVALUATION REPORT\n");
    printf("══════════════════════════════════════════════════════\n");
    printf("  Test cases  : %d\n", n);
    printf("  Beam width  : %u    Max depth : %u\n",
           r->beam_width, r->max_depth);
    printf("  Elapsed     : %.3f s\n\n", r->elapsed_sec);

    if (n == 0) { printf("  (no results)\n"); return; }

    printf("  Top-1 accuracy : %5.1f%%  (%d/%d)\n",
           100.0f * r->top1_hits / n, r->top1_hits, n);
    printf("  Top-3 accuracy : %5.1f%%  (%d/%d)\n",
           100.0f * r->top3_hits / n, r->top3_hits, n);
    printf("  Top-5 accuracy : %5.1f%%  (%d/%d)\n",
           100.0f * r->top5_hits / n, r->top5_hits, n);
    printf("  MRR            : %.4f\n", r->mrr);
    printf("  Avg score      : %.4f\n\n", r->avg_score);

    if (r->category_count > 0) {
        printf("  Per-category breakdown:\n");
        printf("  %-20s  %5s  %6s  %6s  %6s  %6s\n",
               "Category", "Total", "Top-1", "Top-3", "Top-5", "MRR");
        printf("  %s\n", "──────────────────────────────────────────────────");
        for (int i = 0; i < r->category_count; i++) {
            const CategoryStats *cs = &r->categories[i];
            if (cs->total == 0) continue;
            printf("  %-20s  %5d  %5.1f%%  %5.1f%%  %5.1f%%  %.3f\n",
                   cs->name,
                   cs->total,
                   100.0f * cs->top1 / cs->total,
                   100.0f * cs->top3 / cs->total,
                   100.0f * cs->top5 / cs->total,
                   cs->mrr_sum / cs->total);
        }
    }
    printf("══════════════════════════════════════════════════════\n\n");
}

/* ── JSON export ────────────────────────────────────────────────────────────── */

int gemini_eval_report_save_json(const GeminiEvalReport *r, const char *path) {
    if (!r || !path) return -1;
    FILE *fp = fopen(path, "w");
    if (!fp) return -1;

    int n = r->total_cases;
    fprintf(fp, "{\n");
    fprintf(fp, "  \"total_cases\": %d,\n", n);
    fprintf(fp, "  \"top1\": %.4f,\n", n ? (float)r->top1_hits/n : 0.0f);
    fprintf(fp, "  \"top3\": %.4f,\n", n ? (float)r->top3_hits/n : 0.0f);
    fprintf(fp, "  \"top5\": %.4f,\n", n ? (float)r->top5_hits/n : 0.0f);
    fprintf(fp, "  \"mrr\": %.4f,\n", r->mrr);
    fprintf(fp, "  \"avg_score\": %.4f,\n", r->avg_score);
    fprintf(fp, "  \"elapsed_sec\": %.3f,\n", r->elapsed_sec);
    fprintf(fp, "  \"categories\": [\n");
    for (int i = 0; i < r->category_count; i++) {
        const CategoryStats *cs = &r->categories[i];
        float ct = cs->total > 0 ? (float)cs->total : 1.0f;
        fprintf(fp, "    {\"name\": \"%s\", \"total\": %d, "
                "\"top1\": %.4f, \"top3\": %.4f, \"top5\": %.4f, \"mrr\": %.4f}%s\n",
                cs->name, cs->total,
                cs->top1/ct, cs->top3/ct, cs->top5/ct,
                cs->mrr_sum/ct,
                i < r->category_count - 1 ? "," : "");
    }
    fprintf(fp, "  ]\n}\n");
    fclose(fp);
    return 0;
}

/* ── Baseline save / load ───────────────────────────────────────────────────── */

int gemini_eval_save_baseline(const GeminiEvalReport *r, const char *path) {
    if (!r || !path) return -1;
    FILE *fp = fopen(path, "wb");
    if (!fp) return -1;
    uint32_t magic   = EVAL_BASELINE_MAGIC;
    uint32_t version = EVAL_BASELINE_VERSION;
    fwrite(&magic,   4, 1, fp);
    fwrite(&version, 4, 1, fp);
    /* L-3 NOTE: fwrite(r, sizeof(*r)) is a raw struct dump that will produce
     * unreadable files if GeminiEvalReport ever gains a pointer member, if the
     * struct layout changes (padding, field reordering), or if the binary is
     * loaded on a machine with different endianness.  For production use, replace
     * with explicit field-by-field serialization before the first major release. */
    fwrite(r, sizeof(GeminiEvalReport), 1, fp);
    fclose(fp);
    return 0;
}

GeminiEvalReport *gemini_eval_load_baseline(const char *path) {
    if (!path) return NULL;
    FILE *fp = fopen(path, "rb");
    if (!fp) return NULL;
    uint32_t magic, version;
    if (fread(&magic,   4, 1, fp) != 1 || magic   != EVAL_BASELINE_MAGIC  ||
        fread(&version, 4, 1, fp) != 1 || version != EVAL_BASELINE_VERSION) {
        fclose(fp); return NULL;
    }
    GeminiEvalReport *r = malloc(sizeof(GeminiEvalReport));
    if (!r) { fclose(fp); return NULL; }
    if (fread(r, sizeof(GeminiEvalReport), 1, fp) != 1) {
        free(r); fclose(fp); return NULL;
    }
    fclose(fp);
    return r;
}

/* ── Regression detection ───────────────────────────────────────────────────── */

int gemini_eval_detect_regression(const GeminiEvalReport *cur,
                                  const GeminiEvalReport *base,
                                  float                   threshold) {
    if (!cur || !base) return -1;

    int regressions = 0;

    /* Aggregate top-1 */
    float cur_t1  = cur->total_cases  > 0 ? (float)cur->top1_hits  / cur->total_cases  : 0.0f;
    float base_t1 = base->total_cases > 0 ? (float)base->top1_hits / base->total_cases : 0.0f;
    if (base_t1 - cur_t1 > threshold) {
        fprintf(stderr, "REGRESSION: overall top-1 %.2f%% → %.2f%%\n",
                base_t1*100, cur_t1*100);
        regressions++;
    }

    /* Per-category */
    for (int i = 0; i < cur->category_count; i++) {
        const CategoryStats *cc = &cur->categories[i];
        for (int j = 0; j < base->category_count; j++) {
            const CategoryStats *bc = &base->categories[j];
            if (strncmp(cc->name, bc->name, GEMINI_EVAL_MAX_CAT_LEN) != 0)
                continue;
            float cc_t1 = cc->total > 0 ? (float)cc->top1 / cc->total : 0.0f;
            float bc_t1 = bc->total > 0 ? (float)bc->top1 / bc->total : 0.0f;
            if (bc_t1 - cc_t1 > threshold) {
                fprintf(stderr,
                        "REGRESSION [%s]: top-1 %.2f%% → %.2f%%\n",
                        cc->name, bc_t1*100, cc_t1*100);
                regressions++;
            }
        }
    }

    return regressions;
}
