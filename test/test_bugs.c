
#include <stdio.h>
#include <stdint.h>
#include <stdbool.h>
#include <string.h>
#include "tokenizer.h"

// We need to link with syllabifier.o
// Or just include the source for a quick test (not ideal but works for scratch)

int main() {
    SyllableTable stbl = {0};
    Syllabifier *s = syllabifier_create(&stbl);
    
    const char *test_cases[] = {
        "ŋa",   // Should work in both modes (if a is ASCII)
        "ŋá",   // Should work in Predicate mode, fail in Byte-table mode
        "ŋŋa",  // Likely fails in both modes (geminate ŋŋ)
        "áa",   // Likely fails (long toned vowel)
        "bá",   // Should work in Predicate mode
        "bba",  // Should work in both
        NULL
    };
    
    for (int i = 0; test_cases[i]; i++) {
        uint16_t out[10];
        int n = syllabify(s, test_cases[i], out, 10);
        printf("Test: '%s' -> %d syllables\n", test_cases[i], n);
        for (int j = 0; j < n; j++) {
            printf("  [%d]: %s (ID %d)\n", j, stbl.entries[out[j]].text, out[j]);
        }
        if (n == 0) printf("  (No syllables produced)\n");
        printf("\n");
    }
    
    syllabifier_destroy(s);
    return 0;
}
