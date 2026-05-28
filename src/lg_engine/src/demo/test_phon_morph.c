/*
 * test_phon_morph.c — Unit tests for gemini_phon and gemini_morph
 *
 * Build (from project root):
 *   gcc $(CFLAGS) $(INCLUDES) -o bin/test_phon_morph \
 *       src/demo/test_phon_morph.c bin/libgemini_enhanced.a -lm -lpthread
 */

#include "gemini_phon.h"
#include "gemini_morph.h"

#include <stdio.h>
#include <string.h>
#include <stdbool.h>

static int failures = 0;

#define CHECK(cond, msg) \
    do { \
        if (!(cond)) { fprintf(stderr, "FAIL: %s\n", (msg)); failures++; } \
        else         { printf("PASS: %s\n",           (msg)); } \
    } while (0)

/* ── Phonology ────────────────────────────────────────────────────────────── */

typedef struct { const char *input; const char *expected; } PhonCase;

static const PhonCase PHON_CASES[] = {
    /* i + V → y + V */
    { "ki-agenda",  "kyagenda"  },
    { "ki-ogera",   "kyogera"   },
    /* u + V → w + V */
    { "tu-asoma",   "twasoma"   },
    /* No coalescence needed */
    { "omwana",     "omwana"    },
    /* Hyphen strip only — no vowel pair at boundary */
    { "ssebo-nnyabo", "ssebonnyabo" },
    { NULL, NULL }
};

static void test_phonology(void) {
    printf("\n── Phonology ─────────────────────────────────────────────\n");
    for (int i = 0; PHON_CASES[i].input; i++) {
        char buf[128];
        snprintf(buf, sizeof(buf), "%s", PHON_CASES[i].input);
        apply_phonology_rules(buf, sizeof(buf));

        char msg[256];
        snprintf(msg, sizeof(msg), "'%s' → '%s'",
                 PHON_CASES[i].input, PHON_CASES[i].expected);
        CHECK(strcmp(buf, PHON_CASES[i].expected) == 0, msg);
    }

    /* Symmetry: phon_equivalent must be commutative */
    CHECK(phon_equivalent("ki-agenda", "kyagenda"),  "phon_equivalent a→b");
    CHECK(phon_equivalent("kyagenda",  "ki-agenda"), "phon_equivalent b→a (symmetry)");
    CHECK(!phon_equivalent("omwana", "abana"),        "phon_equivalent non-match");
}

/* ── Morphology ───────────────────────────────────────────────────────────── */

/*
 * Expected stems are derived by tracing through PREFIX_TABLE in gemini_morph.c.
 * The table is scanned longest-prefix-first, so a 4-char prefix beats a 3-char
 * one even when the 3-char reading is more linguistically natural.
 *
 * Detailed derivations:
 *
 *   omwana
 *     No 4-char prefix matches. First 3-char match: "omw" (NC_1_MU).
 *     len("omw") = 3; buf = "omwana"; stem = buf + 3 = "ana".
 *     Expected stem: "ana"
 *
 *   abaganda
 *     No 4-char prefix matches. First 3-char match: "aba" (NC_2_BA).
 *     stem = "abaganda" + 3 = "ganda".
 *     Expected stem: "ganda"
 *
 *   okukolima
 *     4-char match: "okuk" (NC_15_OKU, confidence 0.88).
 *     stem = "okukolima" + 4 = "olima".
 *     Expected stem: "olima"
 *
 *     NOTE — greedy-match limitation:
 *     The linguistically correct parse is oku- + kolima (to cultivate).
 *     The "okuk" entry was intended for words like "okukuuma" (to guard)
 *     where the stem begins with a vowel and the 'k' is a glide. It fires
 *     here because the 4-char scan happens before "oku". If this causes
 *     real graph fragmentation, consider adding a post-check that the
 *     character after the stripped prefix is a vowel (okuk + V → okuk,
 *     okuk + C → fall back to oku).
 *
 *   okulabirira
 *     4-char prefixes: none match (okul- not in table).
 *     3-char match: "oku" (NC_15_OKU, confidence 0.92).
 *     stem = "okulabirira" + 3 = "labirira".
 *     Expected stem: "labirira"
 *
 *   xyz
 *     No prefix match, no subject-marker match.
 *     Fallback: canonical = lowercase word = "xyz".
 *     Expected stem: "xyz"
 */

typedef struct {
    const char *word;
    const char *expected_stem;
    float       min_conf;       /* Passed to morph_get_canonical_stem */
    bool        expect_stem_used; /* true → function should return true */
} MorphCase;

static const MorphCase MORPH_CASES[] = {
    /* word          stem         min_conf  stem_used */
    { "omwana",      "ana",       0.70f,    true  },
    { "abaganda",    "ganda",     0.70f,    true  },
    { "okukolima",   "olima",     0.70f,    true  },  /* okuk- greedy match */
    { "okulabirira", "labirira",  0.70f,    true  },  /* oku- match */
    { "xyz",         "xyz",       0.70f,    false },  /* no match, passthrough */
    { NULL, NULL, 0.0f, false }
};

static void test_morphology(void) {
    printf("\n── Morphology ────────────────────────────────────────────\n");
    for (int i = 0; MORPH_CASES[i].word; i++) {
        char stem[MORPH_MAX_STEM_LEN];
        bool used = morph_get_canonical_stem(MORPH_CASES[i].word,
                                             MORPH_CASES[i].min_conf,
                                             stem, sizeof(stem));

        char msg_stem[256], msg_flag[256];
        snprintf(msg_stem, sizeof(msg_stem),
                 "morph '%s' → stem='%s' (want '%s')",
                 MORPH_CASES[i].word, stem, MORPH_CASES[i].expected_stem);
        snprintf(msg_flag, sizeof(msg_flag),
                 "morph '%s' stem_used=%d (want %d)",
                 MORPH_CASES[i].word, (int)used,
                 (int)MORPH_CASES[i].expect_stem_used);

        CHECK(strcmp(stem, MORPH_CASES[i].expected_stem) == 0, msg_stem);
        CHECK(used == MORPH_CASES[i].expect_stem_used,         msg_flag);
    }

    /* morph_analyze — spot-check noun-class and confidence fields */
    MorphInfo info;
    morph_analyze("abaganda", &info);
    CHECK(info.noun_class == NC_2_BA,       "abaganda → NC_2_BA");
    CHECK(info.number     == NUM_PLURAL,    "abaganda → plural");
    CHECK(info.confidence >= 0.70f,         "abaganda confidence >= 0.70");

    morph_analyze("omwana", &info);
    CHECK(info.noun_class == NC_1_MU,       "omwana → NC_1_MU");
    CHECK(info.number     == NUM_SINGULAR,  "omwana → singular");

    /* No-match word — confidence should be low */
    morph_analyze("xyz", &info);
    CHECK(info.confidence < 0.50f,          "xyz → low confidence");

    /* Variant table round-trip */
    MorphVariantTable *vt = morph_variant_table_create();
    morph_variant_table_register(vt, "omwana",  "ana", 42);
    morph_variant_table_register(vt, "abaganda","ganda", 43);

    CHECK(morph_variant_table_lookup(vt, "omwana")   == 42, "vt lookup omwana");
    CHECK(morph_variant_table_lookup(vt, "abaganda") == 43, "vt lookup abaganda");
    CHECK(morph_variant_table_lookup(vt, "zzz")      == 0,  "vt miss returns 0");

    /* Idempotent insert — returns existing ID */
    uint32_t existing = morph_variant_table_register(vt, "omwana", "ana", 99);
    CHECK(existing == 42, "vt idempotent insert returns original id");

    morph_variant_table_destroy(vt);
}

/* ── main ─────────────────────────────────────────────────────────────────── */

int main(void) {
    test_phonology();
    test_morphology();

    printf("\n%s — %d failure(s)\n",
           failures ? "FAILED" : "PASSED", failures);
    return failures ? 1 : 0;
}
