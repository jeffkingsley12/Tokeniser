#define _POSIX_C_SOURCE 200809L
#include "tokenizer.h"
#include "corpus_utils.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include <math.h>
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
    if (argc < 3) {
        fprintf(stderr, "Usage: %s <corpus.txt> <output_model.bin> [max_training_size_mb]\n", argv[0]);
        return 1;
    }

    const char *corpus_path = argv[1];
    const char *output_model_path = argv[2];
    
    size_t max_bytes = 0; /* 0 means unlimited / full corpus */
    if (argc >= 4) {
        char *endp = NULL;
        double mb = strtod(argv[3], &endp);
        if (endp == argv[3] || *endp != '\0') {
            fprintf(stderr, "Error: invalid number for max_training_size_mb: '%s'\n", argv[3]);
            return 1;
        }
        if (!isfinite(mb)) {
            fprintf(stderr, "Error: max_training_size_mb value '%s' is too large or invalid\n", argv[3]);
            return 1;
        }
        if (mb > 0.0) {
            max_bytes = (size_t)(mb * 1024.0 * 1024.0);
            printf("Limiting training size to %.2f MB (%zu bytes)\n", mb, max_bytes);
        }
    }

    printf("=== Starting Tokenizer Model Training ===\n");
    printf("Corpus:      %s\n", corpus_path);
    printf("Destination: %s\n", output_model_path);

    double load_start = get_time_sec();
    Corpus *c = corpus_load(corpus_path, max_bytes, 0, 16384);
    double load_end = get_time_sec();

    if (!c || c->n_docs == 0) {
        fprintf(stderr, "Error: Failed to load corpus from %s\n", corpus_path);
        if (c) corpus_free(c);
        return 1;
    }

    printf("Loaded %u documents, %.2f MB in %.3f seconds\n",
           c->n_docs, (double)c->total_bytes / (1024.0 * 1024.0), load_end - load_start);

    printf("\nTraining tokenizer (running Re-Pair and LOUDS trie build)...\n");
    double train_start = get_time_sec();
    Tokenizer *tok = tokenizer_build((const char **)c->docs, c->n_docs);
    double train_end = get_time_sec();

    if (!tok) {
        fprintf(stderr, "Error: tokenizer_build() failed!\n");
        corpus_free(c);
        return 1;
    }

    printf("Training completed in %.3f seconds\n", train_end - train_start);
    printf("Vocabulary size: %u entries\n", tok->vocab_size);

    printf("\nSaving trained model to %s...\n", output_model_path);
    double save_start = get_time_sec();
    int save_status = tokenizer_save(tok, output_model_path);
    double save_end = get_time_sec();

    if (save_status != 0) {
        fprintf(stderr, "Error: Failed to save model to %s\n", output_model_path);
        tokenizer_destroy(tok);
        corpus_free(c);
        return 1;
    }

    printf("Model saved successfully in %.3f seconds.\n", save_end - save_start);
    
    print_peak_rss();

    tokenizer_destroy(tok);
    corpus_free(c);
    
    printf("\n=== Model Training Successful ===\n");
    return 0;
}
