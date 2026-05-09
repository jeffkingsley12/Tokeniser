#include <stdio.h>
#include <stdlib.h>
#include "tokenizer.h"

int main() {
    printf("Testing tokenizer load and destroy...\n");
    
    Tokenizer* tok = tokenizer_load("tokenizer_model.bin");
    if (!tok) {
        printf("Model file not found, building from mini corpus...\n");
        static const char *mini_corpus[] = {
            "okusoma",
            "okuwandiika",
            "abantu",
            "bali",
            "wano",
        };
        tok = tokenizer_build(mini_corpus, sizeof(mini_corpus)/sizeof(mini_corpus[0]));
        if (!tok) {
            printf("Failed to build tokenizer\n");
            return 1;
        }
        printf("Built tokenizer from %zu documents\n", sizeof(mini_corpus)/sizeof(mini_corpus[0]));
    }
    
    printf("Tokenizer loaded successfully\n");
    
    printf("Destroying tokenizer...\n");
    tokenizer_destroy(tok);
    printf("Tokenizer destroyed successfully\n");
    
    return 0;
}
