#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <stdint.h>
#include <inttypes.h>
#include <errno.h>
#include "libgemini.h"

/**
 * Standard performance measurement helper
 */
static double get_wall_time(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <num_tokens>\n", argv[0]);
        return 1;
    }
    
    char* endptr;
    unsigned long num_tokens = strtoul(argv[1], &endptr, 10);
    if (*endptr != '\0' || errno == ERANGE) {
        fprintf(stderr, "ERROR: Invalid number of tokens: %s\n", argv[1]);
        return 1;
    }
    
    printf("========================================\n");
    printf("GEMINI ENGINE STRESS TEST\n");
    printf("Target: %lu tokens\n", num_tokens);
    printf("========================================\n\n");
    
    EngineContext* ctx = le_init();
    if (!ctx) {
        fprintf(stderr, "Failed to initialize\n");
        return 1;
    }
    
    // Simulate token stream with repeated patterns
    uint32_t vocab_size = 100;
    double start = get_wall_time();
    NodeID prev = LE_INVALID;
    
    for (uint64_t i = 0; i < (uint64_t)num_tokens; i++) {
        // Occasionally reset for sentence breaks (avg 15 tokens)
        if (i % 15 == 0) {
            prev = LE_INVALID;
        }

        // Generate token ID (biased toward common words) - Start at 1
        TokenID token_id;
        if (i % 10 < 7) {
            token_id = 1 + (i % 20);  // 70% from top 20 tokens (1..20)
        } else {
            token_id = 21 + (i % (vocab_size - 20));
        }
        
        /* CRITICAL FIX: le_process_token already returns the target NodeID.
         * Use the return value directly instead of a second O(HASH) lookup. */
        NodeID next = le_process_token(ctx, token_id, prev);

        if (next == INVALID) {
            /* Handle OOM or epoch-reset conditions gracefully */
            prev = LE_INVALID;
        } else {
            prev = next;
        }
        
        if (i > 0 && i % 10000 == 0) {
            printf("Progress: %" PRIu64 " tokens (%.1f%%)\n", 
                   i, 100.0 * i / num_tokens);
        }
    }
    
    double end = get_wall_time();
    double elapsed = end - start;
    
    printf("\n");
    printf("========================================\n");
    printf("PERFORMANCE RESULTS\n");
    printf("========================================\n");
    printf("Total time:      %.3f seconds\n", elapsed);
    if (elapsed > 0) {
        printf("Throughput:      %.0f tokens/sec\n", num_tokens / elapsed);
    }
    printf("\n");
    
    le_print_stats(ctx);
    
    le_destroy(ctx);
    
    return 0;
}
