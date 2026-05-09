#include <stdio.h>
#include <stdint.h>
#include <stdbool.h>
#include <string.h>
#include "utf8_luganda.h"

/* Helper: encode a codepoint to UTF-8 and return byte count */
static int encode_utf8(uint8_t *out, uint32_t cp)
{
    if (cp < 0x80u) {
        out[0] = (uint8_t)cp;
        return 1;
    }
    if (cp < 0x800u) {
        out[0] = 0xC0 | (cp >> 6);
        out[1] = 0x80 | (cp & 0x3F);
        return 2;
    }
    if (cp < 0x10000u) {
        out[0] = 0xE0 | (cp >> 12);
        out[1] = 0x80 | ((cp >> 6) & 0x3F);
        out[2] = 0x80 | (cp & 0x3F);
        return 3;
    }
    if (cp < 0x110000u) {
        out[0] = 0xF0 | (cp >> 18);
        out[1] = 0x80 | ((cp >> 12) & 0x3F);
        out[2] = 0x80 | ((cp >> 6) & 0x3F);
        out[3] = 0x80 | (cp & 0x3F);
        return 4;
    }
    return 0;
}

int main(void)
{
    uint8_t buf[4];
    int errors = 0;
    int tests = 0;

    printf("=== UTF-8 Encoding/Decoding Tests (Codepoints 1-256) ===\n\n");
    printf("CP      Char  Bytes  Vowel  Consonant  Nasal  Luganda  Confusable  Toned  Valid\n");
    printf("----    ----  -----  -----  ---------  -----  -------  ----------  -----  -----\n");

    for (uint32_t cp = 1; cp <= 256; cp++) {
        int enc_len = encode_utf8(buf, cp);
        int char_len = utf8_char_len(buf);
        uint32_t decoded = utf8_decode(buf, char_len);

        tests++;

        if (char_len != enc_len) {
            printf("❌ U+%04X: char_len mismatch: got %d, expected %d\n",
                   cp, char_len, enc_len);
            errors++;
        }

        if (decoded != cp) {
            printf("❌ U+%04X: decode mismatch: got U+%04X\n", cp, decoded);
            errors++;
        }

        /* Print character info */
        printf("U+%04X  ", cp);

        /* Print character itself if printable ASCII, else show hex */
        if (cp >= 32 && cp < 127) {
            printf("'%c'     ", (char)cp);
        } else if (cp < 32 || cp == 127) {
            printf("[%02x]   ", (unsigned int)cp);
        } else {
            printf("[ ]     ");
        }

        /* Byte count */
        printf("%d      ", char_len);

        /* Classifications */
        printf("%s      ", lug_is_vowel(cp) ? "Yes" : "No");
        printf("%s         ", lug_is_consonant(cp) ? "Yes" : "No");
        printf("%s     ", lug_is_nasal(cp) ? "Yes" : "No");
        printf("%s       ", lug_is_luganda_consonant(cp) ? "Yes" : "No");
        printf("%s          ", lug_is_confusable(cp) ? "Yes" : "No");
        printf("%s     ", lug_is_toned(cp) ? "Yes" : "No");
        printf("%s\n", lug_is_valid_luganda_codepoint(cp) ? "Yes" : "No");
    }

    printf("\n=== Classification Tests ===\n\n");

    /* Test vowels */
    printf("Vowels: ");
    bool vowel_ok = (lug_is_vowel('a') && lug_is_vowel('e') && lug_is_vowel('i')
                    && lug_is_vowel('o') && lug_is_vowel('u'));
    printf("%s\n", vowel_ok ? "✅" : "❌");
    tests++;
    if (!vowel_ok) errors++;

    /* Test consonants */
    printf("Consonants: ");
    bool cons_ok = (lug_is_consonant('b') && lug_is_consonant('t') && lug_is_consonant('z'));
    printf("%s\n", cons_ok ? "✅" : "❌");
    tests++;
    if (!cons_ok) errors++;

    /* Test nasals */
    printf("Nasals (m,n,ŋ): ");
    bool nasal_ok = (lug_is_nasal('m') && lug_is_nasal('n') && lug_is_nasal(0x014Bu));
    printf("%s\n", nasal_ok ? "✅" : "❌");
    tests++;
    if (!nasal_ok) errors++;

    /* Test Luganda consonant (excludes q, x) */
    printf("Luganda consonant (excludes q,x): ");
    bool lug_cons_ok = (lug_is_luganda_consonant('b') && !lug_is_luganda_consonant('q') 
                       && !lug_is_luganda_consonant('x') && lug_is_luganda_consonant('z'));
    printf("%s\n", lug_cons_ok ? "✅" : "❌");
    tests++;
    if (!lug_cons_ok) errors++;

    /* Test confusables */
    printf("Confusable detection: ");
    bool conf_ok = (lug_is_confusable(0x043Eu) && !lug_is_confusable('o'));
    printf("%s\n", conf_ok ? "✅" : "❌");
    tests++;
    if (!conf_ok) errors++;

    /* Test lowercase */
    printf("Lowercase: ");
    bool lower_ok = (lug_lowercase('A') == 'a' && lug_lowercase(0x014Au) == 0x014Bu);
    printf("%s\n", lower_ok ? "✅" : "❌");
    tests++;
    if (!lower_ok) errors++;

    /* Test toned vowels */
    printf("Toned vowels: ");
    bool toned_ok = (lug_is_toned(0x00E1u) && !lug_is_toned('a'));
    printf("%s\n", toned_ok ? "✅" : "❌");
    tests++;
    if (!toned_ok) errors++;

    /* Test vowel base */
    printf("Vowel base (á→a): ");
    bool vbase_ok = (lug_vowel_base(0x00E1u) == 'a');
    printf("%s\n", vbase_ok ? "✅" : "❌");
    tests++;
    if (!vbase_ok) errors++;

    /* Test encode length */
    printf("UTF-8 encode length: ");
    bool elen_ok = (utf8_encode_len('a') == 1 && utf8_encode_len(0x014Bu) == 2);
    printf("%s\n", elen_ok ? "✅" : "❌");
    tests++;
    if (!elen_ok) errors++;

    /* Test long vowel */
    printf("Long vowel detection: ");
    bool lvowel_ok = (lug_is_long_vowel('a', 'a') && lug_is_long_vowel('a', 0x00E1u));
    printf("%s\n", lvowel_ok ? "✅" : "❌");
    tests++;
    if (!lvowel_ok) errors++;

    /* Test ny lead */
    printf("Palatal nasal (ny) lead: ");
    bool ny_ok = (lug_is_ny_lead('n') && !lug_is_ny_lead('y'));
    printf("%s\n", ny_ok ? "✅" : "❌");
    tests++;
    if (!ny_ok) errors++;

    printf("\n=== UTF-8 String Length Test ===\n\n");
    size_t slen = utf8_strlen("Ŋŋooyo");
    printf("utf8_strlen(\"Ŋŋooyo\") = %zu (expected 6): %s\n",
           slen, slen == 6 ? "✅" : "❌");
    tests++;
    if (slen != 6) errors++;

    printf("\n=== Summary ===\n");
    printf("Tests: %d | Passed: %d | Failed: %d\n",
           tests, tests - errors, errors);

    return errors > 0 ? 1 : 0;
}
