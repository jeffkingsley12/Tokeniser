#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "tokenizer.h"

int main() {
    printf("Testing simple tokenizer functionality...\n");
    
    // Check if model file exists, if not build a simple one
    Tokenizer* tok = tokenizer_load("tokenizer_model.bin");
    if (!tok) {
        printf("Model file not found, building from mini corpus...\n");
        static const char *mini_corpus[] = {
            "okusoma",
            "okuwandiika",
            "abantu",
            "bali",
            "wano",
            "nnyo",
            "kya",
            "kiki",
            "kino",
        };
        tok = tokenizer_build(mini_corpus, sizeof(mini_corpus)/sizeof(mini_corpus[0]));
        if (!tok) {
            printf("Failed to build tokenizer\n");
            return 1;
        }
        printf("Built tokenizer from %zu documents\n", sizeof(mini_corpus)/sizeof(mini_corpus[0]));
    }
    
    printf("Tokenizer loaded successfully\n");
    
    // Test basic tokenization
    const char* test_text = "okusoma nnyo";
    uint32_t tokens[100];
    int token_count = tokenizer_encode(tok, test_text, tokens, 100);
    
    printf("Test text: \"%s\"\n", test_text);
    printf("Token count: %d\n", token_count);
    printf("Tokens: ");
    for (int i = 0; i < token_count; i++) {
        const char* decoded = tokenizer_decode(tok, tokens[i]);
        if (decoded) {
            printf("[%s] ", decoded);
        } else {
            printf("[UNK] ");
        }
    }
    printf("\n");
    
    // Test performance with simple timing
    printf("\nTesting performance...\n");
    clock_t start = clock();
    
    int iterations = 10000;
    for (int i = 0; i < iterations; i++) {
        tokenizer_encode(tok, test_text, tokens, 100);
    }
    
    clock_t end = clock();
    double elapsed = ((double)(end - start)) / CLOCKS_PER_SEC;
    
    printf("Completed %d iterations in %.3f seconds\n", iterations, elapsed);
    printf("Throughput: %.0f ops/sec\n", iterations / elapsed);
    
    // Cleanup
    tokenizer_destroy(tok);
    printf("Tokenizer destroyed successfully\n");
    
    return 0;
}
