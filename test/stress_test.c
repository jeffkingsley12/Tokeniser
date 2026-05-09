#include "tokenizer.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <sys/time.h>

static double get_time() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (double)tv.tv_sec + (double)tv.tv_usec / 1000000.0;
}

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <corpus.txt> [iterations]\n", argv[0]);
        return 1;
    }

    const char *corpus_path = argv[1];
    int iterations = (argc > 2) ? atoi(argv[2]) : 5;

    printf("=== Luganda Tokenizer Stress Test ===\n");
    printf("Corpus: %s\n", corpus_path);
    printf("Iterations: %d\n\n", iterations);

    /* 1. Load corpus */
    FILE *f = fopen(corpus_path, "rb");
    if (!f) {
        perror("fopen");
        return 1;
    }
    fseek(f, 0, SEEK_END);
    long fsize_long = ftell(f);
    if (fsize_long < 0) {
        perror("ftell");
        fclose(f);
        return 1;
    }
    size_t fsize = (size_t)fsize_long;
    fseek(f, 0, SEEK_SET);

    char *raw_data = malloc(fsize + 1);
    if (!raw_data) {
        fprintf(stderr, "OOM: failed to allocate %zu bytes for corpus\n", fsize);
        fclose(f);
        return 1;
    }
    if (fread(raw_data, 1, fsize, f) != fsize) {
        perror("fread");
        free(raw_data);
        fclose(f);
        return 1;
    }
    raw_data[fsize] = '\0';
    fclose(f);

    /* 2. Run stress iterations */
    for (int it = 0; it < iterations; it++) {
        printf("Iteration %d/%d...\n", it + 1, iterations);
        double start = get_time();

        /* Build tokenizer */
        const char *docs[1] = { raw_data };
        Tokenizer *tok = tokenizer_build(docs, 1);
        if (!tok) {
            fprintf(stderr, "tokenizer_build failed\n");
            continue;
        }

        double build_time = get_time() - start;
        printf("  Build time: %.2fs (Vocab size: %u)\n", build_time, tok->vocab_size);

        /* Encode (fused) */
        uint32_t *out = malloc(fsize * sizeof(uint32_t));
        if (!out) {
            fprintf(stderr, "OOM: failed to allocate output buffer\n");
            tokenizer_destroy(tok);
            continue;
        }

        double enc_start = get_time();
        int n_tokens = tokenizer_encode_fused(tok, raw_data, out, (uint32_t)fsize);
        double enc_time = get_time() - enc_start;

        if (n_tokens < 0) {
            fprintf(stderr, "tokenizer_encode_fused failed\n");
        } else {
            double throughput = (double)fsize / (1024.0 * 1024.0) / enc_time;
            printf("  Encode time: %.2fs (%.2f MB/s, %d tokens)\n", 
                   enc_time, throughput, n_tokens);
        }

        /* 3. Fuzzing: inject noise into a copy of raw_data and re-encode */
        char *noisy_data = malloc(fsize + 1);
        memcpy(noisy_data, raw_data, fsize + 1);
        
        srand((unsigned int)time(NULL));
        for (int i = 0; i < 1000; i++) {
            size_t pos = (size_t)rand() % fsize;
            noisy_data[pos] = (char)(rand() & 0xFF); /* random byte, potentially invalid UTF-8 */
        }

        int n_noisy = tokenizer_encode_fused(tok, noisy_data, out, (uint32_t)fsize);
        if (n_noisy < 0) {
            fprintf(stderr, "  FUZZ: FAILED (crashed or returned error)\n");
        } else {
            printf("  FUZZ: PASSED (handled 1000 noise injections)\n");
        }

        free(noisy_data);
        free(out);
        tokenizer_destroy(tok);
        printf("\n");
    }

    free(raw_data);
    printf("Stress test completed successfully.\n");
    return 0;
}
