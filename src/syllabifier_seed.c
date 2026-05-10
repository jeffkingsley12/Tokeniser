/*
 * syllabifier_seed.c
 *
 * Bridges syllables.yaml → generated syllable_seeds.h → SyllableTable.
 *
 * This file is the ONLY place in the C codebase that reads the generated
 * seed arrays.  All other C files only see the forward declarations in
 * syllable_seeds.h.  If the seed format ever needs to change (e.g. adding
 * category metadata), only this file and the generator need updating.
 *
 * Lifecycle:
 *   Call stbl_seed_phonotactic(stbl) ONCE inside syllabifier_create(),
 *   before any corpus is processed.  This guarantees that every
 *   phonotactic syllable gets a stable low ID (0 .. PHONO_SEED_COUNT-1),
 *   regardless of corpus content.
 *
 *   Then call vocab_validate_morphemes(tok) inside tokenizer_build(), after
 *   the SyllableTable is frozen but before Re-Pair training, to inject
 *   the MORPHEME_SEEDS[] into the BPE vocabulary at known token IDs.
 */

#include "syllable_seeds.h"   /* generated — do not edit */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <stdbool.h>
#include <ctype.h>

#include "tokenizer.h"

/* =========================================================
 *  PHONOTACTIC SEEDS — Complete Luganda Syllable Alphabet
 *  Includes V, VV, CV, CCV (Geminates), NCV (Prenasalized),
 *  and Glide clusters (CwV, CyV).
 * ========================================================= */
static const char *PHONO_SEEDS[] = {
    /* Pure Vowels & Long Vowels (V, VV) */
    "a", "e", "i", "o", "u",
    "aa", "ee", "ii", "oo", "uu",

    /* Standard CV Combinations */
    "ba", "be", "bi", "bo", "bu",
    "ca", "ce", "ci", "co", "cu", /* Rare, mostly loanwords */
    "da", "de", "di", "do", "du",
    "fa", "fe", "fi", "fo", "fu",
    "ga", "ge", "gi", "go", "gu",
    "ja", "je", "ji", "jo", "ju",
    "ka", "ke", "ki", "ko", "ku",
    "la", "le", "li", "lo", "lu",
    "ma", "me", "mi", "mo", "mu",
    "na", "ne", "ni", "no", "nu",
    "nya", "nye", "nyi", "nyo", "nyu",
    "ŋa", "ŋe", "ŋi", "ŋo", "ŋu", /* Ng'oma sound */
    "pa", "pe", "pi", "po", "pu",
    "ra", "re", "ri", "ro", "ru",
    "sa", "se", "si", "so", "su",
    "ta", "te", "ti", "to", "tu",
    "va", "ve", "vi", "vo", "vu",
    "wa", "we", "wi", "wo", "wu",
    "ya", "ye", "yi", "yo", "yu",
    "za", "ze", "zi", "zo", "zu",

    /* Geminate Consonants (CCV) - Essential for Luganda orthography */
    "bba", "bbe", "bbi", "bbo", "bbu",
    "dda", "dde", "ddi", "ddo", "ddu",
    "ffa", "ffe", "ffi", "ffo", "ffu",
    "gga", "gge", "ggi", "ggo", "ggu",
    "jja", "jje", "jji", "jjo", "jju",
    "kka", "kke", "kki", "kko", "kku",
    "lla", "lle", "lli", "llo", "llu",
    "mma", "mme", "mmi", "mmo", "mmu",
    "nna", "nne", "nni", "nno", "nnu",
    "ppa", "ppe", "ppi", "ppo", "ppu",
    "ssa", "sse", "ssi", "sso", "ssu",
    "tta", "tte", "tti", "tto", "ttu",
    "vva", "vve", "vvi", "vvo", "vvu",
    "zza", "zze", "zzi", "zzo", "zzu",
    "nnya", "nnye", "nnyi", "nnyo", "nnyu",
    /*Geminate cvv sounds*/
    "baa", "bee", "bii", "boo", "buu",
    "daa", "dee", "dii", "doo", "duu",
    "faa", "fee", "fii", "foo", "fuu",
    "gaa", "gee", "gii", "goo", "guu",
    "jaa", "jee", "jii", "joo", "juu",
    "kaa", "kee", "kii", "koo", "kuu",
    "laa", "lee", "lii", "loo", "luu",
    "maa", "mee", "mii", "moo", "muu",
    "naa", "nee", "nii", "noo", "nuu",
    "nyaa", "nyee", "nyii", "nyoo", "nyuu",
    "ŋaa", "ŋee", "ŋii", "ŋoo", "ŋuu",
    "paa", "pee", "pii", "poo", "puu",
    "raa", "ree", "rii", "roo", "ruu",
    "saa", "see", "sii", "soo", "suu",
    "taa", "tee", "tii", "too", "tuu",
    "vaa", "vee", "vii", "voo", "vuu",
    "waa", "wee", "wii", "woo", "wuu",
    "yaa", "yee", "yii", "yoo", "yuu",
    "zaa", "zee", "zii", "zoo", "zuu",

    /* Prenasalized Clusters (NCV) */
    "mba", "mbe", "mbi", "mbo", "mbu",
    "mfa", "mfe", "mfi", "mfo", "mfu",
    "mpa", "mpe", "mpi", "mpo", "mpu",
    "mva", "mve", "mvi", "mvo", "mvu",
    "nda", "nde", "ndi", "ndo", "ndu",
    "nga", "nge", "ngi", "ngo", "ngu",
    "nja", "nje", "nji", "njo", "nju",
    "nka", "nke", "nki", "nko", "nku",
    "nsa", "nse", "nsi", "nso", "nsu",
    "nta", "nte", "nti", "nto", "ntu",
    "nza", "nze", "nzi", "nzo", "nzu",

    /* Labialized Clusters (CwV) */
    "bwa", "bwe", "bwi", "bwo", "bwu",
    "dwa", "dwe", "dwi", "dwo", "dwu",
    "fwa", "fwe", "fwi", "fwo", "fwu",
    "gwa", "gwe", "gwi", "gwo", "gwu",
    "jwa", "jwe", "jwi", "jwo", "jwu",
    "kwa", "kwe", "kwi", "kwo", "kwu",
    "lwa", "lwe", "lwi", "lwo", "lwu",
    "mwa", "mwe", "mwi", "mwo", "mwu",
    "nwa", "nwe", "nwi", "nwo", "nwu",
    "nywa", "nywe", "nywi", "nywo", "nywu",
    "pwa", "pwe", "pwi", "pwo", "pwu",
    "rwa", "rwe", "rwi", "rwo", "rwu",
    "swa", "swe", "swi", "swo", "swu", 
    "twa", "twe", "twi", "two", "twu",
    "vwa", "vwe", "vwi", "vwo", "vwu",
    "zwa", "zwe", "zwi", "zwo", "zwu",

    /* Palatalized Clusters (CyV)
     * NOTE: nya/nye/nyi/nyo/nyu are already listed in the Standard CV
     * section above. Only non-duplicate CyV patterns are listed here. */
    "bya", "bye", "byo", "byu",
    "gya", "gye", "gyo", "gyu",
    "kya", "kye", "kyo", "kyu",
    "mya", "mye", "myo", "myu",
    "pya", "pye", "pyo", "pyu",

    /* Prenasalized with Glides (NCwV, NCyV) */
    "mbya", "mbye", "mbyo", "mbyu",
    /* "mppya" cluster removed — "mpp-" is not attested in standard
     * Luganda orthography.  Likely a copy-paste error from mpya/mpye.
     * Keeping these would waste scarce vocabulary slots on patterns
     * that never match real text. */
    "mpya", "mpye", "mpyi", "mpyo", "mpyu", /* Correct prenasalized palatalized */
    "ndwa", "ndwe", "ndwi", "ndwo", "ndwu",
    "ndya", "ndye", "ndyi", "ndyo", "ndyu",
    "ngwa", "ngwe", "ngwi", "ngwo", "ngwu",
    "ngya", "ngye", "ngyi", "ngyo", "ngyu",
    "nkwa", "nkwe", "nkwi", "nkwo", "nkwu",
    "nkya", "nkye", "nkyi", "nkyo", "nkyu",
    "nswa", "nswe", "nswi", "nswo", "nswu",
    "nsya", "nsye", "nsyi", "nsyo", "nsyu",
    "ntwa", "ntwe", "ntwi", "ntwo", "ntwu",
    "ntya", "ntye", "ntyi", "ntyo", "ntyu",
    "nzwa", "nzwe", "nzwi", "nzwo", "nzwu",
    "nzya", "nzye", "nzyi", "nzyo", "nzyu",


    /* Syllabic Nasals & Semivowel Orphans */
    "m", "n", "y", "w", "ŋ",

    /* Control & Punctuation */
    " ", "\t", "\n", "\r",
    ".", ",", "!", "?", ";", ":", "-", "(", ")", "[", "]", "{", "}", "\"", "'", "/", "_", "\\"
};

#define PHONO_SEED_COUNT (sizeof(PHONO_SEEDS) / sizeof(PHONO_SEEDS[0]))

/* =========================================================
 *  Internal helpers
 *  (stbl_intern, stbl_lookup, and consume_syllable are defined in syllabifier.c;
 *   they are declared here as file-scope externs to avoid
 *   polluting the public header with implementation details.)
 * ========================================================= */

/* Intern a surface form; returns syllable ID or UINT16_MAX on error. */
extern uint16_t stbl_intern(SyllableTable *stbl, const char *text);

/* Look up without inserting; returns UINT16_MAX if absent. */
extern uint16_t stbl_lookup(const SyllableTable *stbl, const char *text);

/* Consume one syllable from UTF-8 text. */
extern int consume_syllable(const uint8_t **pp, char *syl_buf, int buf_cap);

/* =========================================================
 *  stbl_seed_phonotactic
 *
 *  Inserts every phonotactic syllable from PHONO_SEEDS[] into `stbl`.
 *
 *  Must be called before any corpus text is processed so that corpus
 *  syllables cannot steal low IDs from the phonotactic alphabet.
 *
 *  Returns the number of syllables successfully interned.
 *  On partial failure (table full) a message is printed and the count
 *  at the point of failure is returned — the build should then abort,
 *  because BASE_SYMBOL_OFFSET needs to be increased.
 * ========================================================= */

/* =========================================================
 *  stbl_seed_bytes
 *
 *  Seeds raw bytes 0x00-0xFF as IDs 1-256 for UTF-8 fallback.
 *  This ensures any byte sequence can be represented even if
 *  it doesn't match Luganda phonotactics.
 *
 *  Must be called BEFORE stbl_seed_phonotactic so phonotactic
 *  syllables get higher IDs (257+).
 * ========================================================= */

int stbl_seed_special(SyllableTable *stbl)
{
    if (!stbl) return -1;

    const char *specials[SPECIAL_TOKENS_COUNT] = {
        "[UNK]",
        "[PAD]",
        "[BOS]",
        "[EOS]",
        "[MASK]"
    };

    for (int i = 0; i < SPECIAL_TOKENS_COUNT; i++) {
        uint16_t id = stbl_intern(stbl, specials[i]);
        if (id != (uint16_t)i) {
            fprintf(stderr,
                "[seed] ERROR: Special token alignment failed at index %d: "
                "expected ID %d, got %u\n",
                i, i, id);
            return -1;
        }
    }
    return 0;
}

int stbl_seed_bytes(SyllableTable *stbl)
{
    /* No-op: byte-level seeding is removed.
     *
     * The tokenizer now operates at the syllable/subword level.
     * Individual bytes were polluting the SyllableTable with 255
     * character-level entries (IDs 5-259), which prevented the trie
     * from matching multi-character syllables like "ku", "so", "ma".
     *
     * Unknown bytes now produce TOK_UNK via consume_syllable_id's
     * fallback path.  Legitimate single-character tokens (vowels,
     * nasals, whitespace, punctuation) are seeded via the V,
     * SYLLABIC_NASAL, WHITESPACE, and PUNCTUATION YAML categories.
     */
    (void)stbl;
    return 0;
}

/*
 * stbl_seed_orphan_ascii
 *
 * Seeds every printable ASCII character (0x20-0x7E) that is NOT already
 * in the SyllableTable.  This guarantees that consume_syllable_ascii()'s
 * single-byte orphan fallback path always produces a valid stbl ID,
 * never TOK_UNK=0.
 *
 * This is the "unified admission" function: EXACTLY the same set of
 * characters that the scanner can emit MUST be present in the stbl
 * before the table is frozen and before louds_build() is called.
 *
 * Call AFTER stbl_seed_special() and stbl_seed_phonotactic() but
 * BEFORE stbl_freeze() / any corpus processing that relies on the IDs.
 *
 * Returns the number of NEW entries inserted (existing entries are
 * counted as 0 because stbl_intern deduplicates automatically).
 */
int stbl_seed_orphan_ascii(SyllableTable *stbl)
{
    if (!stbl) return -1;

    int inserted = 0;
    for (int c = 0x20; c <= 0x7E; c++) {
        char s[2] = { (char)c, '\0' };
        uint16_t existing = stbl_lookup(stbl, s);
        if (existing == UINT16_MAX) {
            uint16_t id = stbl_intern(stbl, s);
            if (id == UINT16_MAX) {
                fprintf(stderr,
                    "[seed] ERROR: stbl full while seeding ASCII 0x%02X ('%c')\n",
                    c, (char)c);
                return -1;
            }
            inserted++;
        }
    }

    fprintf(stderr, "[seed] ASCII orphan sweep: %d new entries seeded "
                    "(stbl now has %u entries)\n",
            inserted, (unsigned)stbl->count);
    return inserted;
}

uint32_t stbl_seed_phonotactic(SyllableTable *stbl)
{
    uint32_t seeded = 0;

    for (uint32_t i = 0; i < PHONO_SEED_COUNT; i++) {
        const char *s = PHONO_SEEDS[i];
        if (!s) break;

        if (stbl->count >= BASE_SYMBOL_OFFSET) {
            fprintf(stderr,
                "[seed] ERROR: SyllableTable is full after %u entries — "
                "increase BASE_SYMBOL_OFFSET in tokenizer.h "
                "(current = %d, PHONO_SEED_COUNT = %zu)\n",
                (unsigned int)seeded, BASE_SYMBOL_OFFSET, PHONO_SEED_COUNT);
            return seeded;   /* caller must treat this as fatal */
        }

        uint16_t id = stbl_intern(stbl, s);
        if (id == UINT16_MAX) {
            fprintf(stderr,
                "[seed] ERROR: stbl_intern failed for phono seed #%u: \"%s\"\n",
                i, s);
            return seeded;
        }
        seeded++;
    }

    return seeded;
}

/* =========================================================
 *  vocab_validate_morphemes
 *
 *  Pre-seeds the tokenizer vocabulary with every entry in MORPHEME_SEEDS[].
 *  Each entry is syllabified (read-only, using the frozen SyllableTable)
 *  and registered as an explicit vocabulary token so that BPE training
 *  treats it as indivisible rather than re-discovering it from frequency.
 *
 *  Call AFTER syl->frozen = true and BEFORE repair_train().
 *
 *  Returns the number of morpheme tokens successfully seeded.
 *  Unknown syllables within a morpheme produce a warning and that
 *  morpheme is skipped (not partially inserted).
 * ========================================================= */

/* =========================================================
 *  clean_and_normalize_seed — Strip comments and fix Cyrillic homoglyphs
 *
 *  MORPHEME_SEEDS[] may contain metadata (comments, glosses) after '#'.
 *  This helper extracts just the Luganda word portion and fixes common
 *  Cyrillic homoglyphs (OCR errors) to Latin equivalents.
 *  ========================================================= */
void clean_and_normalize_seed(char *dest, const char *src, size_t max_len) {
    if (!src || !dest || max_len == 0)
        return;
    
    size_t d = 0;
    const unsigned char *s = (const unsigned char *)src;

    while (s[0] != '\0' && s[0] != '#' && s[0] != ';' && d < max_len - 1) {
        /* Fix common Cyrillic homoglyphs (UTF-8) to Latin equivalents */
        if (s[0] == 0xD0) {
            if (s[1] == 0xBE) { dest[d++] = 'o'; s += 2; continue; } // о -> o
            if (s[1] == 0xB0) { dest[d++] = 'a'; s += 2; continue; } // а -> a
            if (s[1] == 0xB5) { dest[d++] = 'e'; s += 2; continue; } // е -> e
            if (s[1] == 0xBA) { dest[d++] = 'k'; s += 2; continue; } // к -> k
            if (s[1] == 0xBC) { dest[d++] = 'm'; s += 2; continue; } // м -> m
            if (s[1] == 0xBD) { dest[d++] = 'n'; s += 2; continue; } // н -> n
            if (s[1] == 0xBF) { dest[d++] = 'p'; s += 2; continue; } // п -> p
            if (s[1] == 0x81) { dest[d++] = 'c'; s += 2; continue; } // с -> c
            if (s[1] == 0xB3) { dest[d++] = 'x'; s += 2; continue; } // х -> x
        }
        if (s[0] == 0xD1 && s[1] == 0x83) { dest[d++] = 'u'; s += 2; continue; } // у -> u

        dest[d++] = *s++;
    }
    dest[d] = '\0';

    /* Trim trailing whitespace */
    while (d > 0 && isspace((unsigned char)dest[d - 1])) {
        dest[--d] = '\0';
    }
}

/*
 * vocab_validate_morphemes — Validation pass for morpheme seeds.
 *
 * Syllabifies every entry in MORPHEME_SEEDS[] to verify that it can
 * be decomposed into known syllables.  The syllable IDs themselves
 * are NOT retained — actual vocabulary injection happens later in
 * tokenizer_build() which re-syllabifies the same morphemes and
 * passes them to louds_build().
 *
 * The returned count tells the caller how many morphemes are valid
 * so it can pre-size the token array appropriately.
 *
 * If you are looking for the code that actually adds morphemes to
 * the vocabulary, see tokenizer.c:tokenizer_build() Pass 3.
 */
uint32_t vocab_validate_morphemes(Tokenizer *tok)
{
    uint32_t seeded = 0;
    uint16_t syls[MAX_SEQ_LEN];
    char clean_buf[MAX_TOKEN_CHARS];

    for (uint32_t i = 0; i < MORPHEME_SEED_COUNT; i++) {
        const char *raw_morpheme = MORPHEME_SEEDS[i];
        if (!raw_morpheme) break;

        /* Clean the string before syllabifying */
        clean_and_normalize_seed(clean_buf, raw_morpheme, sizeof(clean_buf));
        
        if (clean_buf[0] == '\0') continue;

        /* Now syllabify the CLEAN word (e.g., "omunyenge" instead of the whole line) */
        int n_syls = syllabify(tok->syl, clean_buf, syls, MAX_SEQ_LEN);
        
        if (n_syls <= 0) {
            // This will no longer trigger for definitions/spaces
            continue;
        }

        seeded++;
    }

    return seeded;
}

/* =========================================================
 *  seed_validate  (debug / test helper)
 *
 *  Checks that every phonotactic seed is present in `stbl` after
 *  seeding.  Call in debug builds or unit tests.
 *  Returns 0 if all seeds are found, -1 otherwise.
 * ========================================================= */

int seed_validate(const SyllableTable *stbl)
{
    int ok = 0;
    for (uint32_t i = 0; i < PHONO_SEED_COUNT; i++) {
        const char *s = PHONO_SEEDS[i];
        if (!s) break;
        if (stbl_lookup(stbl, s) == UINT16_MAX) {
            fprintf(stderr,
                "[seed] MISSING after seeding: phono entry #%u \"%s\"\n",
                i, s);
            ok = -1;
        }
    }
    if (ok == 0)
        fprintf(stderr, "[seed] All %zu phonotactic seeds verified.\n",
                PHONO_SEED_COUNT);
    return ok;
}
