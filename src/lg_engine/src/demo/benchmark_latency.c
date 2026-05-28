/*
 * benchmark_latency.c
 *
 * Micro-benchmark to measure search latency and cache-line lookup efficiency
 * under descending DAWG transition sorting.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <stdatomic.h>
#include "libgemini.h"
#include "tokenizer_api.h"
#include "gemini_internal.h"

// High-precision clock utility
static double get_time_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1e9 + (double)ts.tv_nsec;
}

int main(int argc, char **argv) {
    if (argc < 3) {
        printf("Usage: %s <engine_model.bin> <tokenizer_model.bin> [iterations]\n", argv[0]);
        return 1;
    }

    const char *model_path = argv[1];
    const char *tok_path = argv[2];
    uint32_t iterations = (argc >= 4) ? (uint32_t)atoi(argv[3]) : 10000;

    printf("=== Gemini Engine Latency Benchmark ===\n");
    printf("Engine Model    : %s\n", model_path);
    printf("Tokenizer Model : %s\n", tok_path);
    printf("Target Iterations: %u\n", iterations);

    // 1. Load context and tokenizer
    EngineContext *ctx = le_load_mmap(model_path, false);
    if (!ctx) {
        fprintf(stderr, "Error: Failed to load engine model from %s\n", model_path);
        return 1;
    }

    int tok_handle = tok_load(tok_path);
    if (tok_handle < 0) {
        fprintf(stderr, "Error: Failed to load tokenizer model from %s\n", tok_path);
        le_unload_mmap(ctx);
        return 1;
    }

    // 2. Identify candidate starting tokens with active transitions
    uint32_t sym_cnt = get_symbol_count(ctx);
    uint32_t candidate_tokens[256];
    uint32_t candidate_count = 0;

    for (uint32_t i = 0; i < sym_cnt && candidate_count < 256; i++) {
        Symbol *sym = &ctx->dawg_nodes[i];
        if (atomic_load(&sym->first_transition) == LE_INVALID) continue;

        /* FIX: sym->original_scc is a SccID, not a NodeID. Using it directly
         * as a NodeID indexes into ctx->nodes[] at an unrelated position and
         * returns the wrong token. Use le_get_symbol_nodes() to obtain the
         * SCC's actual representative node, then look up its token from there. */
        NodeID member_buf[1];
        uint32_t mc = le_get_symbol_nodes(ctx, i, member_buf, 1);
        if (mc == 0) continue;

        TokenID token_id = get_node_token(ctx, member_buf[0]);
        if (token_id == LE_INVALID) continue;

        candidate_tokens[candidate_count] = token_id;
        candidate_count++;
    }

    printf("Found %u candidate symbols with active transitions.\n", candidate_count);
    if (candidate_count == 0) {
        fprintf(stderr, "Error: No active transitions found in the DAWG. Model might be untrained.\n");
        tok_free(tok_handle);
        le_unload_mmap(ctx);
        return 1;
    }

    // 3. Define configurations for latency testing
    struct {
        uint32_t beam_width;
        uint32_t max_depth;
    } configs[] = {
        {3, 2},
        {5, 3},
        {10, 5}
    };
    uint32_t config_count = sizeof(configs) / sizeof(configs[0]);

    // 4. Run benchmark loops
    for (uint32_t c = 0; c < config_count; c++) {
        uint32_t w = configs[c].beam_width;
        uint32_t d = configs[c].max_depth;
        printf("\nProfiling: Beam Width = %u, Depth = %u\n", w, d);
        printf("--------------------------------------------------\n");

        double total_ns = 0.0;
        double min_ns = 1e18;
        double max_ns = 0.0;
        uint64_t total_results_returned = 0;

        // Warm up cache
        for (uint32_t i = 0; i < 100; i++) {
            TokenID token_id = candidate_tokens[i % candidate_count];
            uint32_t res_count = 0;
            BeamState *beam = le_beam_search(ctx, token_id, w, d, &res_count);
            if (beam) {
                le_free_beam_results(beam, res_count);
            }
        }

        // Timing loop
        for (uint32_t i = 0; i < iterations; i++) {
            TokenID token_id = candidate_tokens[i % candidate_count];
            uint32_t res_count = 0;

            double start = get_time_ns();
            BeamState *beam = le_beam_search(ctx, token_id, w, d, &res_count);
            double diff = get_time_ns() - start;

            if (beam) {
                total_results_returned += res_count;
                le_free_beam_results(beam, res_count);
            }

            total_ns += diff;
            if (diff < min_ns) min_ns = diff;
            if (diff > max_ns) max_ns = diff;
        }

        double avg_us = (total_ns / iterations) / 1000.0;
        double min_us = min_ns / 1000.0;
        double max_us = max_ns / 1000.0;
        double throughput = (double)iterations / (total_ns / 1e9);

        printf("  Average Latency : %.3f us\n", avg_us);
        printf("  Min Latency     : %.3f us\n", min_us);
        printf("  Max Latency     : %.3f us\n", max_us);
        printf("  Throughput      : %.1f queries/sec\n", throughput);
        printf("  Total Results   : %lu\n", (unsigned long)total_results_returned);
    }

    // 5. Cleanup
    tok_free(tok_handle);
    le_unload_mmap(ctx);
    printf("\nBenchmark successfully completed.\n");
    return 0;
}
