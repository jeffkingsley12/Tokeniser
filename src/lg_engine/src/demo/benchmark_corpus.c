/*
 * benchmark_corpus.c
 *
 * Benchmark the lg_engine with benchmark_corpus.txt
 * Compile: gcc -O2 -I../../include -o benchmark_corpus benchmark_corpus.c \
 *           ../gemini_engine.c ../gemini_pool.c ../gemini_tokenizer.c \
 *           ../gemini_accessors.c -lm -lpthread
 * Run: ./benchmark_corpus ../../../../benchmark_corpus.txt
 */

#include "libgemini.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <sys/time.h>

static double get_time_sec() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec * 1e-6;
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
        printf("Usage: %s <corpus.txt>\n", argv[0]);
        return 1;
    }

    const char *corpus_path = argv[1];

    /* Load corpus */
    size_t len;
    char *data = read_file(corpus_path, &len);
    if (!data) {
        fprintf(stderr, "Failed to load corpus: %s\n", corpus_path);
        return 1;
    }

    printf("=== lg_engine Benchmark ===\n");
    printf("Corpus: %s (%zu bytes)\n", corpus_path, len);

    /* Initialize engine and tokenizer */
    printf("Initializing engine...\n");
    EngineContext *ctx = le_init();
    if (!ctx) {
        fprintf(stderr, "Failed to initialize engine\n");
        free(data);
        return 1;
    }

    Tokenizer *tok = tokenizer_init();
    if (!tok) {
        fprintf(stderr, "Failed to initialize tokenizer\n");
        le_destroy(ctx);
        free(data);
        return 1;
    }

    /* Process corpus */
    printf("Processing corpus...\n");
    double start_time = get_time_sec();
    
    NodeID prev_node = LE_INVALID;
    uint32_t line_count = 0;
    char *line = strtok(data, "\n");
    
    while (line != NULL) {
        /* Skip empty lines */
        if (strlen(line) > 0) {
            prev_node = tokenizer_process_text(tok, ctx, line, prev_node);
            line_count++;
        }
        line = strtok(NULL, "\n");
    }
    
    double end_time = get_time_sec();
    double elapsed = end_time - start_time;

    /* Trigger epoch to finalize metrics */
    printf("Triggering epoch...\n");
    le_begin_epoch(ctx);
    uint32_t promoted = le_promote_eligible(ctx);
    printf("Promotions: %u\n", promoted);

    /* Print statistics */
    printf("\n=== Results ===\n");
    printf("Lines processed: %u\n", line_count);
    printf("Time: %.3f seconds\n", elapsed);
    printf("Lines/sec: %.1f\n", line_count / elapsed);
    
    uint32_t nodes = get_node_count(ctx);
    uint32_t edges = get_edge_count(ctx);
    uint32_t sccs = get_scc_count(ctx);
    uint32_t symbols = get_symbol_count(ctx);
    uint32_t tokens = get_token_count(ctx);
    
    printf("Nodes: %u\n", nodes);
    printf("Edges: %u\n", edges);
    printf("SCCs: %u\n", sccs);
    printf("Symbols: %u\n", symbols);
    printf("Tokens: %u\n", tokens);
    
    float global_entropy = le_get_global_entropy(ctx);
    printf("Global entropy: %.4f\n", global_entropy);
    
    uint64_t total_merges = get_total_merges(ctx);
    uint64_t total_promotions = get_total_promotions(ctx);
    printf("Total merges: %lu\n", (unsigned long)total_merges);
    printf("Total promotions: %lu\n", (unsigned long)total_promotions);

    /* Test beam search on a common word */
    printf("\n=== Beam Search Test ===\n");
    uint32_t start_id = tokenizer_get_id(tok, "omwana", false);
    if (start_id != 0) {
        uint32_t result_count = 0;
        BeamState *beam = le_beam_search(ctx, start_id, 3, 5, &result_count);
        printf("Beam search results: %u\n", result_count);
        if (beam && result_count > 0) {
            for (uint32_t i = 0; i < result_count && i < 3; i++) {
                printf("  [%u] log_prob=%.4f length=%u\n", 
                       i, beam[i].log_prob, beam[i].length);
            }
        }
        le_free_beam_results(beam, result_count);
    } else {
        printf("Word 'omwana' not found in vocabulary\n");
    }

    /* Save engine state */
    const char *save_path = "benchmark_engine_state.bin";
    printf("\nSaving engine state to: %s\n", save_path);
    bool saved = le_save_mmap(ctx, save_path);
    if (saved) {
        printf("Save successful\n");
    } else {
        printf("Save failed\n");
    }

    /* Cleanup */
    tokenizer_destroy(tok);
    le_destroy(ctx);
    free(data);

    printf("\nBenchmark completed successfully\n");
    return 0;
}
