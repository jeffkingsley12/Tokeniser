#define _POSIX_C_SOURCE 200809L
#include "tokenizer.h"
#include "corpus_utils.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include <sys/resource.h>
#include <sys/time.h>

static double get_time_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec / 1e9;
}

static void print_peak_rss(void) {
    struct rusage usage;
    if (getrusage(RUSAGE_SELF, &usage) == 0) {
#ifdef __APPLE__
        double rss_mb = (double)usage.ru_maxrss / (1024.0 * 1024.0);
#else
        double rss_mb = (double)usage.ru_maxrss / 1024.0;
#endif
        printf("Peak RSS: %.2f MB\n", rss_mb);
    }
}

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <corpus.txt>\n", argv[0]);
        return 1;
    }

    const char *corpus_path = argv[1];

    size_t sizes[] = {
        10 * 1024 * 1024,
        50 * 1024 * 1024,
        160 * 1024 * 1024
    };

    for (int i = 0; i < 3; i++) {
        size_t target_size = sizes[i];
        printf("\n=== Training Benchmark: %.0f MB ===\n", (double)target_size / (1024*1024));
        Corpus *c = corpus_load(corpus_path, target_size, 0, 16384);
        if (!c || c->n_docs == 0) {
            corpus_free(c);
            break;
        }
        printf("Loaded %u docs, %.2f MB from %s\n", c->n_docs, (double)c->total_bytes / (1024*1024), corpus_path);

        double start = get_time_sec();
        Tokenizer *tok = tokenizer_build((const char **)c->docs, c->n_docs);
        double end = get_time_sec();

        if (tok) {
            printf("Build time: %.3f seconds\n", end - start);
            printf("Tokens built: %u\n", tok->vocab_size);
            tokenizer_destroy(tok);
        } else {
            printf("Build failed!\n");
        }
        print_peak_rss();
        corpus_free(c);
    }
    return 0;
}
