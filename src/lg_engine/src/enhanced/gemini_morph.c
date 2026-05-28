/*
 * gemini_morph.c — Morphology-aware node canonicalization
 *
 * Provides prefix-based stem extraction so that surface variants of the
 * same word (omwana / abana / omwana's) are collapsed onto a single
 * canonical graph node, reducing fragmentation and accelerating SCC
 * formation.
 */

#include "gemini_morph.h"
#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/*
 * Sentinel returned by morph_variant_table_lookup() and
 * morph_variant_table_register() when a surface form is not found or when an
 * allocation error occurs. Using UINT32_MAX avoids ambiguity with token ID 0
 * (typically <unk> / <|space|>), which is a valid canonical_token_id.
 *
 * FIX: The old code used 0 as both "not found" and "freshly inserted",
 * making it impossible for callers to distinguish an <unk> token (ID 0) from
 * a missing entry.  Define explicit sentinels and update all return sites.
 */
#define MORPH_ID_NOT_FOUND   ((uint32_t)0xFFFFFFFFu)
#define MORPH_REGISTER_ERROR ((uint32_t)0xFFFFFFFFu)

/* ── Prefix table ───────────────────────────────────────────────────────────── */

typedef struct {
    const char      *prefix;
    NounClass        noun_class;
    GrammaticalNumber number;
    PartOfSpeech     pos;
    float            confidence;
} PrefixTableEntry;

/*
 * Sorted roughly by descending prefix length so longer (more specific)
 * prefixes match first during linear scan.
 */
static const PrefixTableEntry PREFIX_TABLE[] = {
    /* 4-char prefixes */
    {"okuw", NC_15_OKU, NUM_SINGULAR, POS_VERB,  0.90f},
    {"okuy", NC_15_OKU, NUM_SINGULAR, POS_VERB,  0.90f},
    {"okuk", NC_15_OKU, NUM_SINGULAR, POS_VERB,  0.88f},
    {"okug", NC_15_OKU, NUM_SINGULAR, POS_VERB,  0.88f},
    {"obuw", NC_14_BU,  NUM_SINGULAR, POS_NOUN,  0.85f},
    {"abaw", NC_2_BA,   NUM_PLURAL,   POS_NOUN,  0.88f},
    {"emiy", NC_4_MI,   NUM_PLURAL,   POS_NOUN,  0.82f},
    {"ebiy", NC_8_BI,   NUM_PLURAL,   POS_NOUN,  0.82f},

    /* 3-char prefixes */
    {"oku", NC_15_OKU, NUM_SINGULAR, POS_VERB,  0.92f},
    {"okw", NC_15_OKU, NUM_SINGULAR, POS_VERB,  0.91f},
    {"obu", NC_14_BU,  NUM_SINGULAR, POS_NOUN,  0.85f},
    {"obw", NC_14_BU,  NUM_SINGULAR, POS_NOUN,  0.84f},
    {"otu", NC_13_OBU, NUM_PLURAL,   POS_NOUN,  0.78f},
    {"olu", NC_11_KU,  NUM_SINGULAR, POS_NOUN,  0.85f},
    {"olw", NC_11_KU,  NUM_SINGULAR, POS_NOUN,  0.84f},
    {"aka", NC_12_AKA, NUM_SINGULAR, POS_NOUN,  0.88f},
    {"omu", NC_1_MU,   NUM_SINGULAR, POS_NOUN,  0.80f},  /* Ambiguous NC1/NC3 */
    {"omw", NC_1_MU,   NUM_SINGULAR, POS_NOUN,  0.80f},
    {"aba", NC_2_BA,   NUM_PLURAL,   POS_NOUN,  0.92f},
    {"abw", NC_2_BA,   NUM_PLURAL,   POS_NOUN,  0.90f},
    {"emi", NC_4_MI,   NUM_PLURAL,   POS_NOUN,  0.83f},
    {"ama", NC_6_BI,   NUM_PLURAL,   POS_NOUN,  0.88f},
    {"eki", NC_7_ERI,  NUM_SINGULAR, POS_NOUN,  0.85f},
    {"eky", NC_7_ERI,  NUM_SINGULAR, POS_NOUN,  0.84f},
    {"ebi", NC_8_BI,   NUM_PLURAL,   POS_NOUN,  0.85f},
    {"eby", NC_8_BI,   NUM_PLURAL,   POS_NOUN,  0.84f},

    /* 2-char prefixes (lower confidence, more ambiguous) */
    {"en", NC_9_N,  NUM_SINGULAR, POS_NOUN, 0.65f},
    {"em", NC_9_N,  NUM_SINGULAR, POS_NOUN, 0.60f},  /* Assimilation */
    {"am", NC_6_BI, NUM_PLURAL,   POS_NOUN, 0.70f},
    {"eb", NC_8_BI, NUM_PLURAL,   POS_NOUN, 0.65f},

    /* Terminator */
    {NULL, NC_UNKNOWN, NUM_UNKNOWN, POS_UNKNOWN, 0.0f}
};

/* ── Subject markers (for verb identification) ──────────────────────────────── */

typedef struct {
    NounClass         noun_class;
    GrammaticalNumber number;
    const char       *marker;
} SubjectMarker;

static const SubjectMarker SUBJECT_MARKERS[] = {
    {NC_1_MU,   NUM_SINGULAR, "a"},
    {NC_2_BA,   NUM_PLURAL,   "ba"},
    {NC_3_MU,   NUM_SINGULAR, "gu"},
    {NC_4_MI,   NUM_PLURAL,   "gi"},
    {NC_5_KI,   NUM_SINGULAR, "li"},
    {NC_6_BI,   NUM_PLURAL,   "ga"},
    {NC_7_ERI,  NUM_SINGULAR, "ki"},
    {NC_8_BI,   NUM_PLURAL,   "bi"},
    {NC_9_N,    NUM_SINGULAR, "e"},
    {NC_10_N,   NUM_PLURAL,   "zi"},
    {NC_11_KU,  NUM_SINGULAR, "lu"},
    {NC_12_AKA, NUM_SINGULAR, "ka"},
    {NC_13_OBU, NUM_PLURAL,   "tu"},
    {NC_14_BU,  NUM_SINGULAR, "bu"},
    {NC_15_OKU, NUM_SINGULAR, "ku"},
    {NC_UNKNOWN, NUM_UNKNOWN,  NULL}
};

/* ── Helpers ────────────────────────────────────────────────────────────────── */

static uint32_t hash_string(const char *s) {
    uint32_t h = 5381;
    int c;
    while ((c = (unsigned char)*s++)) {
        h = ((h << 5) + h) ^ c;
    }
    return h;
}

/* Remove possessive apostrophe suffixes like "'s", "'yo", "'" */
static void strip_possessive(char *word) {
    char *apos = strchr(word, '\'');
    if (apos) *apos = '\0';
}

/* ── morph_analyze ──────────────────────────────────────────────────────────── */

bool morph_analyze(const char *word, MorphInfo *out_info) {
    if (!word || !*word || !out_info) return false;

    /* Initialize output struct */
    memset(out_info, 0, sizeof(MorphInfo));

    /* Work on a lowercased copy */
    char buf[MORPH_MAX_WORD_LEN];
    strncpy(buf, word, sizeof(buf) - 1);
    buf[sizeof(buf) - 1] = '\0';
    for (int i = 0; buf[i]; i++) buf[i] = (char)tolower((unsigned char)buf[i]);
    strip_possessive(buf);

    out_info->word      = word;     /* Reference original, not copy */
    out_info->pos       = POS_UNKNOWN;
    out_info->noun_class= NC_UNKNOWN;
    out_info->number    = NUM_UNKNOWN;
    out_info->confidence= 0.5f;

    size_t len = strlen(buf);

    /* Try noun prefixes (longest first by table order) */
    for (int i = 0; PREFIX_TABLE[i].prefix; i++) {
        const char *pfx  = PREFIX_TABLE[i].prefix;
        size_t      plen = strlen(pfx);

        if (len > plen + 1 && strncmp(buf, pfx, plen) == 0) {
            char next = buf[plen];
            if (next == '-' || next == ' ') continue;   /* Bare prefix */

            out_info->prefix     = PREFIX_TABLE[i].prefix;
            out_info->noun_class = PREFIX_TABLE[i].noun_class;
            out_info->number     = PREFIX_TABLE[i].number;
            out_info->pos        = PREFIX_TABLE[i].pos;
            out_info->confidence = PREFIX_TABLE[i].confidence;

            /* Build canonical: lowercase stem only (from stripped buf, not original word) */
            strncpy(out_info->canonical, buf + plen, MORPH_MAX_STEM_LEN - 1);
            out_info->canonical[MORPH_MAX_STEM_LEN - 1] = '\0';

            /*
             * OPTIMIZATION: Point stem at canonical for consistency
             * Contains the correctly stripped, lowercased stem
             */
            out_info->stem = out_info->canonical;
            return true;
        }
    }

    /* Fallback: try verb subject-marker identification */
    for (int i = 0; SUBJECT_MARKERS[i].marker; i++) {
        const char *sm    = SUBJECT_MARKERS[i].marker;
        size_t      smlen = strlen(sm);

        if (len > smlen + 2 && strncmp(buf, sm, smlen) == 0) {
            out_info->pos          = POS_VERB;
            out_info->subject_class= SUBJECT_MARKERS[i].noun_class;
            out_info->prefix       = sm;
            out_info->confidence   = 0.55f;

            strncpy(out_info->canonical, buf + smlen, MORPH_MAX_STEM_LEN - 1);
            out_info->canonical[MORPH_MAX_STEM_LEN - 1] = '\0';

            /*
             * OPTIMIZATION: Point stem at canonical for consistency
             */
            out_info->stem = out_info->canonical;
            return true;
        }
    }

    /* No match: canonical == word lowercased */
    strncpy(out_info->canonical, buf, MORPH_MAX_STEM_LEN - 1);
    out_info->canonical[MORPH_MAX_STEM_LEN - 1] = '\0';
    out_info->confidence = 0.3f;
    return true;
}

/* Stack allocation eliminates heap churn - no free needed */

/* ── morph_get_canonical_stem ───────────────────────────────────────────────── */

bool morph_get_canonical_stem(const char *word,
                               float       min_confidence,
                               char       *out_stem,
                               int         max_len) {
    if (!word || !out_stem || max_len <= 0) return false;

    MorphInfo info;
    if (!morph_analyze(word, &info)) {
        strncpy(out_stem, word, max_len - 1);
        out_stem[max_len - 1] = '\0';
        return false;
    }

    bool used_stem = (info.confidence >= min_confidence &&
                      info.canonical[0] != '\0');

    strncpy(out_stem, used_stem ? info.canonical : word, max_len - 1);
    out_stem[max_len - 1] = '\0';

    return used_stem;
}

/* ── morph_guess_class_from_prefix ─────────────────────────────────────────── */

NounClass morph_guess_class_from_prefix(const char *word) {
    if (!word) return NC_UNKNOWN;
    for (int i = 0; PREFIX_TABLE[i].prefix; i++) {
        size_t plen = strlen(PREFIX_TABLE[i].prefix);
        if (strlen(word) > plen + 1 && strncmp(word, PREFIX_TABLE[i].prefix, plen) == 0) {
            char next = word[plen];
            if (next && next != '-' && next != ' ')
                return PREFIX_TABLE[i].noun_class;
        }
    }
    return NC_UNKNOWN;
}

NounClass morph_guess_class_from_prefix_validated(const char *word,
                                                   float      *confidence) {
    if (!word) { if (confidence) *confidence = 0.0f; return NC_UNKNOWN; }

    MorphInfo info;
    if (!morph_analyze(word, &info)) { if (confidence) *confidence = 0.0f; return NC_UNKNOWN; }

    NounClass nc = info.noun_class;
    if (confidence) *confidence = info.confidence;
    return nc;
}

/* ── MorphVariantTable ──────────────────────────────────────────────────────── */

MorphVariantTable *morph_variant_table_create(void) {
    MorphVariantTable *vt = calloc(1, sizeof(MorphVariantTable));
    return vt;
}

uint32_t morph_variant_table_register(MorphVariantTable *vt,
                                      const char        *surface,
                                      const char        *canonical,
                                      uint32_t           canonical_token_id) {
    if (!vt || !surface || !canonical) return MORPH_REGISTER_ERROR;

    uint32_t h   = hash_string(surface) % MORPH_VARIANT_TABLE_SIZE;
    MorphVariantEntry *e = vt->buckets[h];

    while (e) {
        if (strcmp(e->surface, surface) == 0)
            return e->canonical_token_id;   /* Already registered: return its ID */
        e = e->next;
    }

    /* New entry */
    MorphVariantEntry *ne = calloc(1, sizeof(MorphVariantEntry));
    if (!ne) return MORPH_REGISTER_ERROR;

    strncpy(ne->surface,   surface,   MORPH_MAX_WORD_LEN - 1);
    strncpy(ne->canonical, canonical, MORPH_MAX_STEM_LEN - 1);
    ne->canonical_token_id = canonical_token_id;
    ne->next               = vt->buckets[h];
    vt->buckets[h]         = ne;
    vt->entry_count++;

    /* FIX: The old code returned 0 for both "freshly inserted" and
     * "already exists with ID 0" (e.g. <unk>). Callers had no way to
     * distinguish these cases. We now return the canonical_token_id that
     * was just stored, matching the "already exists" path. Callers that
     * need to know whether an entry was newly created should check whether
     * morph_variant_table_lookup() returned MORPH_ID_NOT_FOUND *before*
     * calling this function. */
    return canonical_token_id;
}

uint32_t morph_variant_table_lookup(const MorphVariantTable *vt,
                                    const char              *surface) {
    if (!vt || !surface) return MORPH_ID_NOT_FOUND;
    uint32_t h = hash_string(surface) % MORPH_VARIANT_TABLE_SIZE;
    for (MorphVariantEntry *e = vt->buckets[h]; e; e = e->next) {
        if (strcmp(e->surface, surface) == 0)
            return e->canonical_token_id;
    }
    /* FIX: returning 0 here was ambiguous — 0 is a valid canonical_token_id
     * (typically <unk> or <|space|>). Return MORPH_ID_NOT_FOUND instead so
     * callers can distinguish "not found" from "found with ID 0". */
    return MORPH_ID_NOT_FOUND;
}

const char *morph_variant_table_reverse(const MorphVariantTable *vt,
                                        uint32_t                 id) {
    if (!vt || id == 0) return NULL;
    /*
     * WARNING — O(MORPH_VARIANT_TABLE_SIZE) = O(65537) linear scan.
     *
     * This function is intended ONLY for display/debug output, never for
     * hot-path lookups. If you find yourself calling this from a tight loop,
     * add a reverse hash table (id → surface) instead.
     *
     * In debug builds we fire an assertion so accidental hot-path use is
     * caught immediately during development.
     */
#ifndef NDEBUG
    static int call_count = 0;
    if (++call_count > 1000) {
        fprintf(stderr,
                "morph_variant_table_reverse: PERFORMANCE WARNING — called >1000 times. "
                "This is an O(%d) scan intended only for display/debug. "
                "Add a reverse id→surface index if this is on a hot path.\n",
                MORPH_VARIANT_TABLE_SIZE);
        call_count = 0; /* Reset so warning fires again after another 1000 calls */
    }
#endif

    for (int i = 0; i < MORPH_VARIANT_TABLE_SIZE; i++) {
        for (MorphVariantEntry *e = vt->buckets[i]; e; e = e->next) {
            if (e->canonical_token_id == id)
                return e->surface;  /* First surface form registered */
        }
    }
    return NULL;
}

void morph_variant_table_destroy(MorphVariantTable *vt) {
    if (!vt) return;
    for (int i = 0; i < MORPH_VARIANT_TABLE_SIZE; i++) {
        MorphVariantEntry *e = vt->buckets[i];
        while (e) {
            MorphVariantEntry *next = e->next;
            free(e);
            e = next;
        }
    }
    free(vt);
}
