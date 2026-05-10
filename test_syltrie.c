#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "tokenizer.h"

int main() {
    const char *docs[] = {"kanywa-musenke"};
    Tokenizer *t = tokenizer_build(docs, 1);
    if (!t) {
        printf("Failed to build tokenizer\n");
        return 1;
    }
    
    const char *text = "kanywa-musenke";
    const uint8_t *wp = (const uint8_t *)text;
    while (*wp) {
        uint16_t id = UINT16_MAX;
        int consumed = consume_syllable_id(t->stbl->trie, &wp, &id);
        if (consumed == 0) {
            printf("consume_syllable_id returned 0 at '%s'\n", wp);
            break;
        }
        printf("Consumed %d bytes, ID = %u, next wp = '%s'\n", consumed, id, wp);
    }
    
    return 0;
}
