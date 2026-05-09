#define _POSIX_C_SOURCE 200809L
#include "tokenizer.h"
#include "corpus_utils.h"
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <stdint.h>
#include <stdatomic.h>

/* Use CLOCK_MONOTONIC for wall-clock timing on all platforms. */
static inline uint64_t get_time_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;
}

#define MAX_TOKENS_PER_DOC 16384

/* run_benchmark: tok is non-const because we reset atomic stats counters. */
void run_benchmark(const char *label, Tokenizer *tok, const Corpus *corpus,
                   int iterations, bool fused) {
    if (!label || !tok || !corpus || iterations <= 0) return;
    if (corpus->n_docs == 0) {
        fprintf(stderr, "[benchmark] empty corpus — nothing to benchmark\n");
        return;
    }

    uint32_t *out = malloc(MAX_TOKENS_PER_DOC * sizeof(uint32_t));
    if (!out) {
        fprintf(stderr, "[benchmark] OOM allocating token output buffer\n");
        return;
    }

    /* Warm up — one full pass over the corpus outside the timed region */
    for (uint32_t i = 0; i < corpus->n_docs; i++) {
        if (!corpus->docs[i]) continue;
#if USE_FUSED
        if (fused) {
            tokenizer_encode_fused(tok, corpus->docs[i], out, MAX_TOKENS_PER_DOC);
        } else {
#endif
            tokenizer_encode(tok, corpus->docs[i], out, MAX_TOKENS_PER_DOC);
#if USE_FUSED
        }
#endif
    }

    /* Reset stats before timed run */
    atomic_store(&tok->stats.louds_single_syl_fallbacks, 0);
    atomic_store(&tok->stats.louds_multi_syl_matches, 0);
    atomic_store(&tok->stats.louds_single_syl_matches, 0);
    atomic_store(&tok->stats.fast_path_hits, 0);
    atomic_store(&tok->stats.trie_traversals, 0);

    uint64_t total_tokens = 0;
    uint64_t syll_ns = 0;
    uint64_t loud_ns = 0;

    uint64_t start_ns = get_time_ns();

    for (int it = 0; it < iterations; it++) {
        uint64_t iter_syll_start = get_time_ns();
        for (uint32_t i = 0; i < corpus->n_docs; i++) {
            if (!corpus->docs[i]) continue;
#if USE_FUSED
            if (fused) {
                int n = tokenizer_encode_fused(tok, corpus->docs[i], out,
                                               MAX_TOKENS_PER_DOC);
                if (n > 0) total_tokens += (uint64_t)(unsigned)n;
            } else {
#endif
                uint16_t syls[MAX_TOKENS_PER_DOC];
                int n_syls = syllabify(tok->syl, corpus->docs[i], syls,
                                       MAX_TOKENS_PER_DOC);
                if (n_syls > 0) {
                    uint64_t iter_loud_start = get_time_ns();
                    syll_ns += iter_loud_start - iter_syll_start;
                    int n = louds_tokenize(tok->louds, syls, (uint32_t)n_syls,
                                           out, MAX_TOKENS_PER_DOC);
                    loud_ns += get_time_ns() - iter_loud_start;
                    iter_syll_start = get_time_ns(); /* reset for next doc's syllabify */
                    if (n > 0) total_tokens += (uint64_t)(unsigned)n;
                } else {
                    /* No syllables: reset start so next doc's syllabify is timed cleanly */
                    iter_syll_start = get_time_ns();
                }
#if USE_FUSED
            }
#endif
        }
    }

    uint64_t end_ns = get_time_ns();
    uint64_t duration_ns = end_ns - start_ns;

    /* Avoid division by zero when duration is 0 (very fast runs / mock clocks) */
    if (duration_ns == 0) {
        fprintf(stderr, "[benchmark][%s] duration is 0 ns — clock resolution too low\n",
                label);
        free(out);
        return;
    }

    if (!fused && total_tokens > 0) {
        printf("Syllabify ns: %" PRIu64 " (%.2f%%)\n",
               syll_ns, (double)syll_ns / (double)duration_ns * 100.0);
        printf("Louds ns:     %" PRIu64 " (%.2f%%)\n",
               loud_ns, (double)loud_ns / (double)duration_ns * 100.0);
    }

    double elapsed_sec = (double)duration_ns / 1e9;
    double mb = (double)corpus->total_bytes * (double)iterations / (1024.0 * 1024.0);
    printf("[%s] %" PRIu64 " tokens, %.2f MB/s\n",
           label, total_tokens, mb / elapsed_sec);

    free(out);
}

int main(int argc, char **argv) {
    if (argc < 3) {
        fprintf(stderr, "Usage: %s <model_path> <corpus_path> [iterations]\n",
                argv[0]);
        return 1;
    }
    const char *model_path  = argv[1];
    const char *corpus_path = argv[2];

    /* atoi returns 0 for non-numeric input; treat 0 as invalid */
    int iterations = (argc >= 4) ? atoi(argv[3]) : 10;
    if (iterations <= 0) {
        fprintf(stderr, "[benchmark] invalid iteration count '%s', using 10\n",
                argc >= 4 ? argv[3] : "default");
        iterations = 10;
    }

    printf("Loading model: %s\n", model_path);
    Tokenizer *tok = tokenizer_load(model_path);
    if (!tok) {
        fprintf(stderr, "Failed to load model: %s\n", model_path);
        return 1;
    }

    printf("Loading corpus: %s\n", corpus_path);
    Corpus *corpus = corpus_load(corpus_path, 0, 0, 1024);
    if (!corpus) {
        fprintf(stderr, "Failed to load corpus: %s\n", corpus_path);
        tokenizer_destroy(tok);
        return 1;
    }
    printf("Loaded %u documents, %zu bytes\n", corpus->n_docs, corpus->total_bytes);

    printf("Running benchmarks (%d iterations)...\n", iterations);

    run_benchmark("Regular Tokenizer", tok, corpus, iterations, false);
#if USE_FUSED
    run_benchmark("Fused Tokenizer",   tok, corpus, iterations, true);
#endif

    corpus_free(corpus);
    tokenizer_destroy(tok);
    return 0;
}
