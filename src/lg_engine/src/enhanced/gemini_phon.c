/*
 * gemini_phon.c — Phonological normalization for Gemini tokenizer
 */

#include "gemini_phon.h"
#include <ctype.h>
#include <string.h>

static inline int is_vowel(char c) {
    c = (char)tolower((unsigned char)c);
    return (c == 'a' || c == 'e' || c == 'i' || c == 'o' || c == 'u');
}

int apply_phonology_rules(char *word, size_t buflen) {
    if (!word || buflen == 0) return 0;

    int  len   = (int)strlen(word);
    if (len < 2) return len;

    int w1 = 0;
    for (int r = 0; r < len; r++) {
        if (word[r] == '-') continue;   
        word[w1++] = word[r];
    }
    word[w1] = '\0';
    len = w1;

    int write_idx = 0;
    for (int read_idx = 0; read_idx < len; read_idx++) {
        char cur  = word[read_idx];
        char next = (read_idx + 1 < len) ? word[read_idx + 1] : '\0';

        if (is_vowel(cur) && is_vowel(next)) {
            char c = (char)tolower((unsigned char)cur);
            char n = (char)tolower((unsigned char)next);

            if (c == 'u' && n != 'u') {
                word[write_idx++] = 'w';
                continue;
            }

            if (c == 'i' && n != 'i') {
                word[write_idx++] = 'y';
                continue;
            }

            if (c == 'o' && n == 'a') {
                word[write_idx++] = 'w';
                word[write_idx++] = 'a';
                read_idx++;   
                continue;
            }
        }

        word[write_idx++] = cur;
    }

    word[write_idx] = '\0';
    (void)buflen;
    return write_idx;
}

void apply_phonology_rules_sentence(char *text, size_t buflen) {
    if (!text || buflen == 0) return;

    char *p = text;
    while (*p) {
        while (*p && isspace((unsigned char)*p)) p++;
        if (!*p) break;

        char *start = p;
        while (*p && !isspace((unsigned char)*p)) p++;

        char saved = *p;
        *p = '\0';

        int old_token_len = (int)(p - start);
        apply_phonology_rules(start, (size_t)old_token_len + 1);
        int new_token_len = (int)strlen(start);

        if (new_token_len < old_token_len) {
            int shift = old_token_len - new_token_len;
            *p = saved;                                     
            
            /* O(1) length check ensures memmove executes in O(N) total time */
            size_t tail_len = strlen(p);
            memmove(start + new_token_len, p, tail_len + 1); 
            p -= shift;                                      
        } else {
            *p = saved;
        }
    }
}

int phon_equivalent(const char *a, const char *b) {
    if (!a || !b) return 0;

    char bufa[256], bufb[256];
    strncpy(bufa, a, sizeof(bufa) - 1); bufa[sizeof(bufa) - 1] = '\0';
    strncpy(bufb, b, sizeof(bufb) - 1); bufb[sizeof(bufb) - 1] = '\0';

    for (int i = 0; bufa[i]; i++) bufa[i] = (char)tolower((unsigned char)bufa[i]);
    for (int i = 0; bufb[i]; i++) bufb[i] = (char)tolower((unsigned char)bufb[i]);

    apply_phonology_rules(bufa, sizeof(bufa));
    apply_phonology_rules(bufb, sizeof(bufb));

    return strcmp(bufa, bufb) == 0;
}