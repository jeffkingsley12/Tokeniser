/*
 * benchmark.c
 *
 * Cycles/token benchmark harness for tokenizer optimization.
 * Compile: gcc -O3 -march=native -o bench benchmark.c
 * Run: ./bench corpus.txt
 */

#define _GNU_SOURCE
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <x86intrin.h>

#include "tokenizer.h"

static inline uint64_t rdtsc() {
    return __rdtsc();
}

static char *read_file(const char *path, size_t *len) {
    FILE *f = fopen(path, "rb");
    if (!f) {
        perror("fopen");
        return NULL;
    }
    
    fseek(f, 0, SEEK_END);
    *len = ftell(f);
    fseek(f, 0, SEEK_SET);

    char *buf = malloc(*len + 1);
    if (!buf) {
        fclose(f);
        return NULL;
    }
    
    size_t n = fread(buf, 1, *len, f);
    fclose(f);
    buf[n] = '\0';
    *len = n;
    return buf;
}

int main(int argc, char **argv) {
    if (argc < 2) {
        printf("Usage: %s <corpus.txt> [model.bin]\n", argv[0]);
        printf("  corpus.txt - text file to tokenize\n");
        printf("  model.bin  - trained tokenizer model (optional)\n");
        return 1;
    }

    const char *corpus_path = argv[1];
    const char *model_path = argc > 2 ? argv[2] : "tokenizer_model.bin";

    /* Load corpus */
    size_t len;
    uint8_t *data = (uint8_t*)read_file(corpus_path, &len);
    if (!data) {
        fprintf(stderr, "Failed to load corpus: %s\n", corpus_path);
        return 1;
    }

    printf("=== Tokenizer Benchmark ===\n");
    printf("Corpus: %s (%zu bytes)\n", corpus_path, len);
    
    /* Check ASCII percentage */
    size_t ascii_bytes = 0;
    for (size_t i = 0; i < len; i++) {
        if (data[i] < 128) ascii_bytes++;
    }
    printf("ASCII: %.1f%%\n", 100.0 * ascii_bytes / len);

    /* Output buffer for tokens (allocate before tokenizer building) */
    uint32_t token_cap = (uint32_t)(len / 2 + 1000);
    uint32_t *tokens = malloc(sizeof(uint32_t) * token_cap);
    if (!tokens) {
        fprintf(stderr, "Failed to allocate token buffer\n");
        free(data);
        return 1;
    }

    /* Build or load tokenizer */
    Tokenizer *t = NULL;
    int needs_save = 0;
    
    if (access(model_path, R_OK) == 0) {
        printf("Loading model: %s\n", model_path);
        t = tokenizer_load(model_path);
    }
    
    if (!t) {
        printf("Training new tokenizer on corpus...\n");
        const char *docs[] = {(const char*)data};
        t = tokenizer_build(docs, 1);
        if (!t) {
            fprintf(stderr, "Failed to build tokenizer\n");
            free(data);
            free(tokens);
            return 1;
        }
        needs_save = 1;
    }

    /* Warmup */
    printf("Warming up...\n");
#if USE_FUSED
    for (int i = 0; i < 5; i++) {
        int n = tokenizer_encode_fused(t, (const char*)data, tokens, token_cap);
        if (n < 0) {
            fprintf(stderr, "Encode failed on warmup\n");
            free(data);
            free(tokens);
            tokenizer_destroy(t);
            return 1;
        }
    }
#else
    printf("Fused tokenizer not compiled (USE_FUSED=0).\n");
    free(data);
    free(tokens);
    tokenizer_destroy(t);
    return 1;
#endif

    /* Benchmark */
    printf("Running benchmark...\n");
    
    int iterations = 10;
    uint64_t total_cycles = 0;
    size_t total_tokens = 0;
    
#if USE_FUSED
    for (int i = 0; i < iterations; i++) {
        uint64_t start = rdtsc();
        int n = tokenizer_encode_fused(t, (const char*)data, tokens, token_cap);
        uint64_t end = rdtsc();
        
        if (n < 0) {
            fprintf(stderr, "Encode failed during benchmark\n");
            free(data);
            free(tokens);
            tokenizer_destroy(t);
            return 1;
        }
        total_cycles += (end - start);
        total_tokens += (uint32_t)n;
    }
#else
    printf("Fused tokenizer not compiled (USE_FUSED=0).\n");
    free(data);
    free(tokens);
    tokenizer_destroy(t);
    return 1;
#endif
    
    /* Calculate metrics */
    double avg_cycles = (double)total_cycles / iterations;
    double avg_tokens = (double)total_tokens / iterations;
    double seconds = avg_cycles / 3.0e9;
    double mb_per_sec = (seconds > 0.0 && len > 0)
        ? (len / (1024.0 * 1024.0)) / seconds
        : 0.0;

    printf("\n=== Results ===\n");
    printf("Tokens:      %.0f per run\n", avg_tokens);
    printf("Cycles:      %.0f per run\n", avg_cycles);

    if (total_tokens == 0 || len == 0) {
        printf("Cycles/token: n/a (no tokens produced)\n");
        printf("Tokens/MB:   n/a\n");
        printf("Throughput:  %.1f MB/s (est @ 3GHz)\n", mb_per_sec);
    } else {
        double cycles_per_token = avg_cycles / avg_tokens;
        double tokens_per_mb = avg_tokens / (len / (1024.0 * 1024.0));
        printf("Cycles/token: %.2f\n", cycles_per_token);
        printf("Tokens/MB:   %.0f\n", tokens_per_mb);
        printf("Throughput:  %.1f MB/s (est @ 3GHz)\n", mb_per_sec);
        printf("Target < 25 cycles/token: %s\n", cycles_per_token < 25 ? "✓ PASS" : "✗ FAIL");
        printf("Target > 500 MB/s:        %s\n", mb_per_sec > 500 ? "✓ PASS" : "✗ FAIL");
    }
    
    /* Fast path stats from LOUDS */
    uint64_t hits = t->stats.fast_path_hits;
    uint64_t traversals = t->stats.trie_traversals;
    if (hits + traversals > 0) {
        double fast_pct = 100.0 * (double)hits / (double)(hits + traversals);
        printf("Fast path:   %.1f%% (%lu hits, %lu LOUDS)\n", 
               fast_pct, (unsigned long)hits, (unsigned long)traversals);
    }

    if (total_tokens > 0 && len > 0) {
        printf("\n=== Targets ===\n");
        double cycles_per_token = avg_cycles / avg_tokens;
        printf("Target < 25 cycles/token: %s\n", cycles_per_token < 25 ? "✓ PASS" : "✗ FAIL");
        printf("Target > 500 MB/s:        %s\n", mb_per_sec > 500 ? "✓ PASS" : "✗ FAIL");
    }

    /* Save model if newly trained */
    if (needs_save) {
        printf("\nSaving model to: %s\n", model_path);
        if (tokenizer_save(t, model_path) < 0) {
            fprintf(stderr, "Failed to save model\n");
        }
    }

    free(data);
    free(tokens);
    tokenizer_destroy(t);
    
    return 0;
}
