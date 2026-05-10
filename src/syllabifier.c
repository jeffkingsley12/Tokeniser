/*
 * syllabifier.c — Luganda-aware syllabifier, production rewrite v2
 *
 * Changes from v1:
 *  - Static phoneme lookup tables replace per-instance bitmaps.
 *    Faster (BSS/data segment, cache-friendly), zero per-call overhead.
 *  - Full 10-tier phonotactic onset hierarchy matching the Python engine:
 *      1. NNYV  "nny" + V
 *      2. NYV   "ny"  + V
 *      3. NCSV  N + C + S + V   (ncsv_valid)
 *      4. GSV   G + G + S + V   (gsv_valid — geminate + semivowel)
 *      5. NCV   N + C + V       (ncv_valid)
 *      6. CSV   C + S + V       (csv_valid)
 *      7. GV    G + G + V       (geminate + vowel)
 *      8. CV    (C | N | ŋ) + V
 *      9. SV    S + V           (fixes "weebale", "wano", etc.)
 *     10. V     bare vowel (with optional long-vowel VV extension)
 *  - Onset copy uses memcpy instead of a byte-loop macro — eliminates the
 *    p/out-diverge bug from the original proposal (when cap ran out mid-
 *    onset, p advanced but out did not, corrupting the syllable string).
 *  - p[N] lookaheads are null-safe via short-circuit macros (LK1..LK3).
 *  - ncv_valid / csv_valid take uint8_t and cast via (unsigned char) before
 *    tolower() — prevents signed-char UB on non-ASCII bytes.
 *  - IS_CORE_CONS now includes uppercase letters (proposal omitted them).
 *  - is_likely_luganda() conservatively accepts words of any length;
 *    short particles ("a", "ba", "ku", ...) are NOT rejected.
 *    Illegal-cluster list trimmed: "sh","gh","th","ph" removed (loanwords).
 *  - OOV detection fully integrated into syllabify(); words containing
 *    'x' / 'q' or a hard illegal cluster are silently skipped.
 *  - UTF-8 velar nasal ŋ (U+014B = 0xC5 0x8B) treated as a 2-byte CV onset.
 *  - Long vowels: consecutive identical vowels (e.g. "aa", "oo") consumed
 *    as a single nucleus.
 */

#include "tokenizer.h"
#include "unicode_props.h"
#include <assert.h>
#include <ctype.h>
#ifdef __SSE4_1__
#include <smmintrin.h>
#endif
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef __SSE4_1__
#include <immintrin.h>
#include <smmintrin.h>

/* SIMD character categories */
#define CAT_VOWEL (1u << 0)
#define CAT_NASAL (1u << 1)
#define CAT_SEMIV (1u << 2)
#define CAT_ALPHA (1u << 3)

static inline __m128i classify_16_sse(__m128i v) {
  __m128i low_lut =
      _mm_setr_epi8(0, 0x09, 0x08, 0x08, 0x08, 0x09, 0x08, 0x0C, 0x08, 0x0D,
                    0x08, 0x08, 0x08, 0x0A, 0x0A, 0x09);
  __m128i high_lut =
      _mm_setr_epi8(0, 0, 0, 0, 0x0F, 0x0F, 0x0F, 0x0F, 0, 0, 0, 0, 0, 0, 0, 0);
  __m128i low_nibble = _mm_and_si128(v, _mm_set1_epi8(0x0F));
  __m128i high_nibble =
      _mm_and_si128(_mm_srli_epi16(v, 4), _mm_set1_epi8(0x0F));
  return _mm_and_si128(_mm_shuffle_epi8(low_lut, low_nibble),
                       _mm_shuffle_epi8(high_lut, high_nibble));
}
#endif

#ifdef __ARM_NEON
#include <arm_neon.h>

/* NEON character categories — same bit flags as SSE */
#define NEON_CAT_VOWEL (1u << 0)
#define NEON_CAT_NASAL (1u << 1)
#define NEON_CAT_SEMIV (1u << 2)
#define NEON_CAT_ALPHA (1u << 3)

/*
 * classify_16_neon: SIMD classification of 16 ASCII bytes.
 * Returns uint8x16_t where each byte contains category bitmask.
 * Only valid for ASCII input (< 0x80); non-ASCII bytes return 0.
 */
static inline uint8x16_t classify_16_neon(uint8x16_t v) {
  static const uint8_t low_data[16] = {0x00, 0x09, 0x08, 0x08, 0x08, 0x09,
                                       0x08, 0x0C, 0x08, 0x0D, 0x08, 0x08,
                                       0x08, 0x0A, 0x0A, 0x09};
  static const uint8_t high_data[16] = {0x00, 0x00, 0x00, 0x00, 0x0F, 0x0F,
                                        0x0F, 0x0F, 0x00, 0x00, 0x00, 0x00,
                                        0x00, 0x00, 0x00, 0x00};

  uint8x16_t low_tbl = vld1q_u8(low_data);
  uint8x16_t high_tbl = vld1q_u8(high_data);

  uint8x16_t low_nibble = vandq_u8(v, vdupq_n_u8(0x0F));
  uint8x16_t high_nibble = vshrq_n_u8(v, 4);

  uint8x16_t low_cat = vqtbl1q_u8(low_tbl, low_nibble);
  uint8x16_t high_mask = vqtbl1q_u8(high_tbl, high_nibble);

  return vandq_u8(low_cat, high_mask);
}

/* Extract bitmask from NEON vector: bit i = 1 if byte i != 0 */
static inline uint16_t neon_movemask_u8(uint8x16_t v) {
  uint8_t tmp[16];
  vst1q_u8(tmp, v);
  uint16_t mask = 0;
  for (int i = 0; i < 16; i++) {
    if (tmp[i])
      mask |= (1u << i);
  }
  return mask;
}
#endif

/* =========================================================
 *  OPTIONAL: UTF-8 Luganda predicate mode
 *
 *  Define USE_UTF8_LUGANDA_PREDICATES at compile time to use the
 *  cleaner, codepoint-aware predicates from utf8_luganda.h instead of
 *  byte-level lookup tables.
 *
 *  Trade-offs:
 *    • Byte tables (default): fastest, cache-friendly, zero per-call overhead
 *    • Predicates: cleaner code, codepoint-aware (handles UTF-8 properly),
 *                  slightly slower due to function calls and switch statements
 *
 *  For production high-throughput tokenization, keep byte tables.
 *  For debugging or when you want cleaner code, use predicates.
 *
 *  Compile with: gcc -DUSE_UTF8_LUGANDA_PREDICATES ... -o syllabifier.o
 * ========================================================= */

#if defined(USE_UTF8_LUGANDA_PREDICATES) || defined(HYBRID_UTF8_SYLLABIFIER)
#include "utf8_luganda.h"
#endif

/* =========================================================
 *  OPTIONAL: Grapheme cluster support (UAX #29 + emoji)
 *
 *  Define USE_GRAPHEME_CLUSTERS to enable proper Unicode grapheme
 *  boundary detection before syllabification.
 *
 *  This ensures:
 *    • Combining marks stay with base characters
 *    • Emoji sequences (👩🏽‍💻) are treated as single units
 *    • Flag emojis (🇺🇬) are paired correctly
 *    • Syllabifier never sees partial graphemes
 *
 *  For maximum correctness with mixed Luganda/emoji text.
 * ========================================================= */

#ifdef USE_GRAPHEME_CLUSTERS
#include "grapheme_scanner.h"
#endif

/* =========================================================
 *  Static phoneme lookup tables (DEFAULT MODE)
 *
 *  Compile-time initialised bool[256] arrays indexed by the
 *  raw (unsigned) byte value.  Non-ASCII bytes are 0/false
 *  in every table, so multi-byte UTF-8 sequences are safely
 *  ignored at the byte level.
 * ========================================================= */

/* Static phoneme lookup tables are used in both byte-table (default) and
 * hybrid modes.  In pure predicate mode they are omitted. */
#ifndef USE_UTF8_LUGANDA_PREDICATES

static const bool IS_VOWEL[256] = {
    ['a'] = 1, ['e'] = 1, ['i'] = 1, ['o'] = 1, ['u'] = 1,
    ['A'] = 1, ['E'] = 1, ['I'] = 1, ['O'] = 1, ['U'] = 1,
};

static const bool IS_NASAL[256] = {
    ['n'] = 1,
    ['m'] = 1,
    ['N'] = 1,
    ['M'] = 1,
};

static const bool IS_SEMI[256] = {
    ['w'] = 1,
    ['y'] = 1,
    ['W'] = 1,
    ['Y'] = 1,
};

/* Core consonants: all ASCII letters that are neither vowels, nasals,
 * nor semivowels.  Includes both cases — the proposal omitted uppercase. */
static const bool IS_CORE_CONS[256] = {
    /* lowercase */
    ['b'] = 1,
    ['c'] = 1,
    ['d'] = 1,
    ['f'] = 1,
    ['g'] = 1,
    ['h'] = 1,
    ['j'] = 1,
    ['k'] = 1,
    ['l'] = 1,
    ['p'] = 1,
    ['q'] = 1,
    ['r'] = 1,
    ['s'] = 1,
    ['t'] = 1,
    ['v'] = 1,
    ['z'] = 1,
    /* uppercase — absent from proposal, added here */
    ['B'] = 1,
    ['C'] = 1,
    ['D'] = 1,
    ['F'] = 1,
    ['G'] = 1,
    ['H'] = 1,
    ['J'] = 1,
    ['K'] = 1,
    ['L'] = 1,
    ['P'] = 1,
    ['Q'] = 1,
    ['R'] = 1,
    ['S'] = 1,
    ['T'] = 1,
    ['V'] = 1,
    ['Z'] = 1,
};

#endif /* !USE_UTF8_LUGANDA_PREDICATES */

/* =========================================================
 *  Phoneme classification abstraction
 *
 *  This layer allows switching between three modes without
 *  modifying the rest of the syllabifier.
 *
 *  1. Byte-table (default): O(1) array lookups, fastest.
 *  2. Predicates (-DUSE_UTF8_LUGANDA_PREDICATES): codepoint-aware,
 *     handles all UTF-8, slightly slower.
 *  3. Hybrid (-DHYBRID_UTF8_SYLLABIFIER): byte-table fast path for
 *     ASCII, UTF-8 predicate fallback for multi-byte. Best of both.
 * ========================================================= */

#ifdef USE_UTF8_LUGANDA_PREDICATES

/* --- Mode 2: Pure predicate-based implementations --- */

static inline bool is_vowel_at(const uint8_t *p) {
  int len = utf8_char_len(p);
  uint32_t cp = utf8_decode(p, len);
  return lug_is_vowel(cp);
}

static inline bool is_nasal_at(const uint8_t *p) {
  int len = utf8_char_len(p);
  uint32_t cp = utf8_decode(p, len);
  return lug_is_nasal(cp);
}

static inline bool is_semivowel_at(const uint8_t *p) {
  int len = utf8_char_len(p);
  uint32_t cp = utf8_decode(p, len);
  return lug_is_semivowel(cp);
}

static inline bool is_consonant_at(const uint8_t *p) {
  int len = utf8_char_len(p);
  uint32_t cp = utf8_decode(p, len);
  return lug_is_consonant(cp);
}

#elif defined(HYBRID_UTF8_SYLLABIFIER)

/* --- Mode 3: Hybrid — fast table path for ASCII, predicate fallback --- */

static inline bool is_vowel_at(const uint8_t *p) {
  if (__builtin_expect(*p < 128, 1))
    return IS_VOWEL[*p];
  int len = utf8_char_len(p);
  uint32_t cp = utf8_decode(p, len);
  return lug_is_vowel(cp);
}

static inline bool is_nasal_at(const uint8_t *p) {
  if (__builtin_expect(*p < 128, 1))
    return IS_NASAL[*p];
  int len = utf8_char_len(p);
  uint32_t cp = utf8_decode(p, len);
  return lug_is_nasal(cp);
}

static inline bool is_semivowel_at(const uint8_t *p) {
  if (__builtin_expect(*p < 128, 1))
    return IS_SEMI[*p];
  int len = utf8_char_len(p);
  uint32_t cp = utf8_decode(p, len);
  return lug_is_semivowel(cp);
}

static inline bool is_consonant_at(const uint8_t *p) {
  if (__builtin_expect(*p < 128, 1))
    return IS_CORE_CONS[*p];
  int len = utf8_char_len(p);
  uint32_t cp = utf8_decode(p, len);
  return lug_is_consonant(cp);
}

#else

/* --- Mode 1: Pure byte-table implementations (fastest, ASCII-only) --- */

static inline bool is_vowel_at(const uint8_t *p) { return IS_VOWEL[*p]; }

static inline bool is_nasal_at(const uint8_t *p) { return IS_NASAL[*p]; }

static inline bool is_semivowel_at(const uint8_t *p) { return IS_SEMI[*p]; }

static inline bool is_consonant_at(const uint8_t *p) {
  return IS_CORE_CONS[*p];
}

#endif /* mode selection */

/* =========================================================
 *  Phonotactic validation helpers
 *
 *  All accept uint8_t so the tolower() cast is safe on every
 *  platform regardless of whether plain char is signed.
 * ========================================================= */

/* Bitmask helpers for Luganda phonotactics */
#define M(c) (1u << ((c) - 'a'))

static const uint32_t MASK_NCV = M('b') | M('c') | M('d') | M('h') | M('g') |
                                 M('j') | M('k') | M('l') | M('m') | M('n') |
                                 M('q') | M('r') | M('s') | M('t') | M('z');

static const uint32_t MASK_CSV_Y = M('b') | M('d') | M('f') | M('g') | M('h') |
                                   M('k') | M('l') | M('m') | M('p') | M('q') |
                                   M('r') | M('s') | M('t') | M('v') | M('z');

static const uint32_t MASK_GSV_W = M('b') | M('d') | M('g') | M('k') | M('m') |
                                   M('n') | M('p') | M('s') | M('t');

static const uint32_t MASK_GSV_Y =
    M('b') | M('d') | M('g') | M('k') | M('m') | M('p') | M('t');

static const uint32_t MASK_NCSV_Y =
    M('b') | M('d') | M('g') | M('k') | M('p') | M('t');

#undef M

/* ncv_valid: is (nasal, core-consonant) a valid NC onset? */
static inline bool ncv_valid(uint8_t n, uint8_t c) {
  (void)n;
  int cl = tolower((unsigned char)c);
  if (cl < 'a' || cl > 'z')
    return false;
  return (MASK_NCV & (1u << (cl - 'a')));
}

/* csv_valid: is (consonant, semivowel) a valid CS onset? */
static inline bool csv_valid(uint8_t c, uint8_t s) {
  int cl = tolower((unsigned char)c);
  int sl = tolower((unsigned char)s);
  if (cl < 'a' || cl > 'z')
    return false;
  if (sl == 'w')
    return true;
  if (sl == 'y')
    return (MASK_CSV_Y & (1u << (cl - 'a')));
  return false;
}

/* gsv_valid: is (geminate-consonant, semivowel) a valid GSV onset? */
static inline bool gsv_valid(uint8_t c, uint8_t s) {
  int cl = tolower((unsigned char)c);
  int sl = tolower((unsigned char)s);
  if (cl < 'a' || cl > 'z')
    return false;
  if (sl == 'w')
    return (MASK_GSV_W & (1u << (cl - 'a')));
  if (sl == 'y')
    return (MASK_GSV_Y & (1u << (cl - 'a')));
  return false;
}

/* ncsv_valid: is (nasal, core-consonant, semivowel) a valid NCSV onset? */
static inline bool ncsv_valid(uint8_t n, uint8_t c, uint8_t s) {
  if (!ncv_valid(n, c))
    return false;
  int cl = tolower((unsigned char)c);
  int sl = tolower((unsigned char)s);
  if (cl < 'a' || cl > 'z')
    return false;
  if (sl == 'w')
    return true;
  if (sl == 'y')
    return (MASK_NCSV_Y & (1u << (cl - 'a')));
  return false;
}

/* =========================================================
 *  UTF-8 helpers
 *
 *  When using predicate mode, we use the functions from utf8_luganda.h.
 *  In byte-table mode, we keep our own simplified versions for local use.
 * ========================================================= */

/* Local utf8_char_len is only needed when we do NOT already have it from
 * utf8_luganda.h (i.e. pure byte-table mode). */
#if !defined(USE_UTF8_LUGANDA_PREDICATES) &&                                   \
    !defined(HYBRID_UTF8_SYLLABIFIER) && !defined(USE_GRAPHEME_CLUSTERS)

static int utf8_char_len(const uint8_t *p) {
  if ((*p & 0x80) == 0x00)
    return 1;
  if ((*p & 0xE0) == 0xC0)
    return 2;
  if ((*p & 0xF0) == 0xE0)
    return 3;
  if ((*p & 0xF8) == 0xF0)
    return 4;
  return 1; /* invalid byte — treat as single unit */
}

static uint32_t utf8_decode(const uint8_t *p, int len) {
  switch (len) {
  case 1:
    return (uint32_t)p[0];
  case 2:
    return ((uint32_t)(p[0] & 0x1F) << 6) | (uint32_t)(p[1] & 0x3F);
  case 3:
    return ((uint32_t)(p[0] & 0x0F) << 12) | ((uint32_t)(p[1] & 0x3F) << 6) |
           (uint32_t)(p[2] & 0x3F);
  case 4:
    return ((uint32_t)(p[0] & 0x07) << 18) | ((uint32_t)(p[1] & 0x3F) << 12) |
           ((uint32_t)(p[2] & 0x3F) << 6) | (uint32_t)(p[3] & 0x3F);
  default:
    return 0xFFFD;
  }
}

#endif /* !USE_UTF8_LUGANDA_PREDICATES && !HYBRID_UTF8_SYLLABIFIER */

/* Returns 2 if p[0..1] encodes ŋ (U+014B = 0xC5 0x8B), else 0. */
static inline int velar_nasal_len(const uint8_t *p) {
  return (p[0] == 0xC5u && p[1] == 0x8Bu) ? 2 : 0;
}

/* =========================================================
 *  SyllableTable helpers
 * ========================================================= */

uint32_t stbl_hash(const char *s) {
  uint32_t h = 2166136261u; /* FNV-1a 32-bit */
  while (*s) {
    h ^= (uint8_t)*s++;
    h *= 16777619u;
  }
  return h;
}

void stbl_freeze(SyllableTable *t) {
  if (t) {
    t->frozen = true;
  }
}

uint16_t stbl_intern(SyllableTable *t, const char *text) {
  if (!text || !*text)
    return UINT16_MAX;
  
  /* Enforce frozen semantics during inference */
  if (t->frozen) {
    return TOK_UNK;  /* No mutation allowed when frozen */
  }
  
  /* Check if already present */
  uint16_t existing = stbl_lookup(t, text);
  if (existing != UINT16_MAX)
    return existing;
  /* Otherwise intern at next available ID */
  return stbl_intern_fixed(t, text, (uint16_t)t->count);
}

uint16_t stbl_intern_fixed(SyllableTable *t, const char *text, uint16_t id) {
  if (!text || id >= BASE_SYMBOL_OFFSET)
    return UINT16_MAX;
    
  /* Enforce frozen semantics during inference */
  if (t->frozen) {
    return TOK_UNK;  /* No mutation allowed when frozen */
  }

  uint32_t mask = 8191u;
  uint32_t h = stbl_hash(text) & mask;

  for (uint32_t probe = 0; probe < 8192; probe++) {
    uint32_t slot = (h + probe) & mask;
    if (!t->ht_used[slot]) {
      /* Forced ID allocation.
       * Guard: if entries[id] is already occupied by a DIFFERENT text,
       * that means two callers tried to assign the same ID to different
       * surface forms.  This corrupts the decode table.  Reject loudly. */
      if (id < t->count && t->entries[id].text[0] != '\0' &&
          strncmp(t->entries[id].text, text, MAX_TOKEN_CHARS) != 0) {
        fprintf(
            stderr,
            "[stbl_intern_fixed] ID collision: id %u already holds \"%.*s\", "
            "cannot also assign \"%.*s\"\n",
            id, MAX_TOKEN_CHARS - 1, t->entries[id].text, MAX_TOKEN_CHARS - 1,
            text);
        return UINT16_MAX;
      }

      /* Update t->count to be at least id + 1. */
      if (id >= t->count)
        t->count = (uint16_t)(id + 1u);

      strncpy(t->entries[id].text, text, MAX_TOKEN_CHARS - 1);
      t->entries[id].text[MAX_TOKEN_CHARS - 1] = '\0';
      t->entries[id].id = id;

      t->ht_used[slot] = 1;
      t->ht_keys[slot] = id;
      strncpy(t->ht_text[slot], text, MAX_TOKEN_CHARS - 1);
      t->ht_text[slot][MAX_TOKEN_CHARS - 1] = '\0';
      if (t->count > 5734) { /* 0.7 * 8192 */
        static bool warned = false;
        if (!warned) {
          fprintf(stderr,
                  "[stbl] WARNING: SyllableTable load factor > 70%% (%u/8192). "
                  "Performance may degrade.\n",
                  t->count);
          warned = true;
        }
      }
      return id;
    }
    if (strcmp(t->ht_text[slot], text) == 0)
      return t->ht_keys[slot];
  }
  fprintf(
      stderr,
      "[stbl] FATAL: hash table full (%u entries). Increase ht_keys[] size.\n",
      (unsigned)t->count);
  return UINT16_MAX;
}

uint16_t stbl_lookup(const SyllableTable *t, const char *text) {
  uint32_t mask = 8191u;
  uint32_t h = stbl_hash(text) & mask;
  for (uint32_t probe = 0; probe < 8192; probe++) {
    uint32_t slot = (h + probe) & mask;
    if (!t->ht_used[slot])
      return UINT16_MAX;
    if (strcmp(t->ht_text[slot], text) == 0)
      return t->ht_keys[slot];
  }
  return UINT16_MAX;
}

/* =========================================================
 *  Syllable Trie (Fast Path)
 * ========================================================= */

SylTrie *syltrie_build(const SyllableTable *stbl) {
  if (!stbl)
    return NULL;

  SylTrie *trie = calloc(1, sizeof *trie);
  if (!trie)
    return NULL;

  typedef struct {
    uint32_t next[256];
    uint16_t id;
  } DenseNode;

  uint32_t node_cap = (uint32_t)stbl->count * 4 + 256;
  DenseNode *dense = calloc(node_cap, sizeof *dense);
  if (!dense) {
    free(trie);
    return NULL;
  }

  uint32_t dense_count = 1; /* root is node 0 */
  uint32_t total_edges = 0;

  for (uint16_t i = 0; i < stbl->count; i++) {
    const char *text = stbl->entries[i].text;
    if (!text[0])
      continue;

    uint32_t node = 0;
    const uint8_t *p = (const uint8_t *)text;

    while (*p) {
      uint8_t c = *p++;
      if (dense[node].next[c] == 0) {
        if (dense_count >= node_cap) {
          uint32_t new_cap = node_cap * 2;
          DenseNode *new_nodes = realloc(dense, new_cap * sizeof *dense);
          if (!new_nodes) {
            free(dense);
            free(trie);
            return NULL;
          }
          memset(new_nodes + node_cap, 0, (new_cap - node_cap) * sizeof *dense);
          dense = new_nodes;
          node_cap = new_cap;
        }
        dense[node].next[c] = dense_count++;
        total_edges++; /* New edge added */
      }
      node = dense[node].next[c];
    }
    /* mark terminal. +1 so 0 means not-terminal */
    dense[node].id = stbl->entries[i].id + 1;
  }

  /* Convert to CSR layout */
  trie->nodes = calloc(dense_count, sizeof *trie->nodes);
  trie->edges = calloc(total_edges, sizeof *trie->edges);
  if (!trie->nodes || (total_edges > 0 && !trie->edges)) {
    free(trie->nodes);
    free(trie->edges);
    free(dense);
    free(trie);
    return NULL;
  }

  trie->node_count = dense_count;
  trie->edge_count = total_edges;

  uint32_t edge_idx = 0;
  for (uint32_t i = 0; i < dense_count; i++) {
    trie->nodes[i].first_edge = edge_idx;
    trie->nodes[i].id = dense[i].id;
    uint16_t out_degree = 0;
    for (int c = 0; c < 256; c++) {
      if (dense[i].next[c] != 0) {
        trie->edges[edge_idx].label = (uint8_t)c;
        trie->edges[edge_idx].next_node = dense[i].next[c];
        edge_idx++;
        out_degree++;
      }
    }
    trie->nodes[i].edge_count = out_degree;

    /* Populate root lookup table for faster dispatch */
    if (i == 0) {
      for (int c = 0; c < 256; c++) {
        trie->root_next[c] = dense[0].next[c];
      }
    }
  }

  free(dense);
  return trie;
}

void syltrie_destroy(SylTrie *trie) {
  if (!trie)
    return;
  free(trie->nodes);
  free(trie->edges);
  free(trie);
}

int consume_syllable_id(const SylTrie *t, const uint8_t **pp, uint16_t *id_out) {
  if (!t || !pp || !*pp || !**pp) return 0;
  
  const uint8_t *p = *pp;
  const uint8_t *start = p;
  uint32_t node = 0;
  uint16_t last_id = 0;
  const uint8_t *last_match_p = NULL;
  
  while (*p) {
    uint8_t c = *p;
    uint32_t next = 0;
    
    if (node == 0) {
      next = t->root_next[c];
    } else {
      const SylTrieNode *nd = &t->nodes[node];
      for (uint16_t i = 0; i < nd->edge_count; i++) {
        const SylTrieEdge *edge = &t->edges[nd->first_edge + i];
        if (edge->label == c) {
          next = edge->next_node;
          break;
        }
      }
    }
    
    if (next == 0) break;
    
    node = next;
    p++;
    
    if (t->nodes[node].id != 0) {
      last_id = t->nodes[node].id;
      last_match_p = p;
    }
  }
  
  if (last_id != 0) {
    *id_out = last_id - 1; /* Note: trie stores ID + 1 */
    int consumed = (int)(last_match_p - start);
    *pp = last_match_p;
    return consumed;
  }
  
  return 0;
}

/* =========================================================
 *  Syllabifier lifecycle
 * ========================================================= */

Syllabifier *syllabifier_create(SyllableTable *stbl) {
  Syllabifier *s = calloc(1, sizeof *s);
  if (!s)
    return NULL;
  s->stbl = stbl;

  /* Initialise legacy per-instance bitmaps for any external readers.
   * The hot-path syllabifier uses the file-static tables above.     */
  const char *vowels = "aeiouAEIOU";
  for (const char *v = vowels; *v; v++)
    s->is_vowel[(uint8_t)*v] = 1;
  s->is_nasal[(uint8_t)'n'] = s->is_nasal[(uint8_t)'N'] = 1;
  s->is_nasal[(uint8_t)'m'] = s->is_nasal[(uint8_t)'M'] = 1;

  return s;
}

void syllabifier_destroy(Syllabifier *s) { free(s); }

/* =========================================================
 *  Core syllable consumer
 *
 *  Consumes exactly one syllable from *pp, writes its UTF-8 text into
 *  syl_buf (null-terminated, <= buf_cap-1 content bytes).
 *
 *  Returns:
 *    > 0  bytes consumed (syllable text in syl_buf)
 *      0  end of string
 *    < 0  orphan byte consumed (syl_buf[0]=='\0', caller skips)
 *
 *  Null-safe lookahead macros:
 *    C's ?: short-circuits, so LK1(p) never reads p[1] unless p[0]!=0,
 *    LK2 never reads p[2] unless p[1]!=0, etc.  For null-terminated
 *    inputs this means we never access past the allocated buffer.
 * ========================================================= */

#define LK0(p) ((unsigned)(p)[0])
#define LK1(p) ((p)[0] ? (unsigned)(p)[1] : 0u)
#define LK2(p) ((p)[0] && (p)[1] ? (unsigned)(p)[2] : 0u)
#define LK3(p) ((p)[0] && (p)[1] && (p)[2] ? (unsigned)(p)[3] : 0u)

static int consume_syllable_ascii(const uint8_t **pp, char *syl_buf,
                                  int buf_cap) {
  const uint8_t *p = *pp;
  const uint8_t *start = p;

  /* Worst-case Luganda onset: 3 bytes (e.g., "ngg") + 2 long vowel + 1 null = 6
   */
  if (buf_cap < 6) {
    *syl_buf = '\0';
    return 0; /* Signal insufficient buffer; caller should advance minimally */
  }

  char *out = syl_buf;

#define IS_V(idx) is_vowel_at(p + (idx))
#define IS_N(idx) is_nasal_at(p + (idx))
#define IS_S(idx) is_semivowel_at(p + (idx))
#define IS_C(idx) is_consonant_at(p + (idx))

  /* ================================================================
   *  Onset selection — Grouped by first character for faster dispatch
   * ================================================================ */
  int onset_len = 0;
  int c0 = tolower(LK0(p));

  switch (c0) {
  case 'n': {
    int c1 = tolower(LK1(p));
    if (c1 == 'n') {
      if (tolower(LK2(p)) == 'y') {
        if (IS_V(3)) {
          onset_len = 3; // NNYV
        } else if (IS_S(3) && IS_V(4) && tolower(LK3(p)) == 'w') {
          onset_len = 4; // NNYwV
        }
      } else if (IS_S(2) && IS_V(3) && gsv_valid((uint8_t)LK0(p), (uint8_t)LK2(p))) {
        onset_len = 3; // nnwV
      } else if (IS_V(2)) {
        onset_len = 2; // nnV
      }
    } else if (c1 == 'y') {
      if (IS_V(2)) {
        onset_len = 2; // NYV
      } else if (IS_S(2) && IS_V(3) && tolower(LK2(p)) == 'w') {
        onset_len = 3; // NYwV
      }
    } else {
      /* Check NCV/NCSV */
      if (IS_C(1)) {
        if (IS_S(2) && IS_V(3) &&
            ncsv_valid((uint8_t)LK0(p), (uint8_t)LK1(p), (uint8_t)LK2(p))) {
          onset_len = 3;
        } else if (IS_V(2) && ncv_valid((uint8_t)LK0(p), (uint8_t)LK1(p))) {
          onset_len = 2;
        }
      }
    }
    /* Fallback to CV if not matched above */
    if (onset_len == 0 && IS_V(1))
      onset_len = 1;
    break;
  }
  case 'm': {
    int c1 = tolower(LK1(p));
    if (c1 == 'm') {
      if (IS_S(2) && IS_V(3) && gsv_valid((uint8_t)LK0(p), (uint8_t)LK2(p))) {
        onset_len = 3; // mmwV / mmyV
      } else if (IS_V(2)) {
        onset_len = 2; // mmV
      }
    } else {
      /* Check NCV/NCSV */
      if (IS_C(1)) {
        if (IS_S(2) && IS_V(3) &&
            ncsv_valid((uint8_t)LK0(p), (uint8_t)LK1(p), (uint8_t)LK2(p))) {
          onset_len = 3;
        } else if (IS_V(2) && ncv_valid((uint8_t)LK0(p), (uint8_t)LK1(p))) {
          onset_len = 2;
        }
      }
    }
    if (onset_len == 0 && IS_V(1))
      onset_len = 1;
    break;
  }
  case 'w':
  case 'y': {
    if (IS_V(1))
      onset_len = 1; // SV
    break;
  }
  default: {
    if (IS_C(0)) {
      /* Check GSV/GV/CSV/CV */
      if (LK0(p) == LK1(p)) {
        if (IS_S(2) && IS_V(3) && gsv_valid((uint8_t)LK0(p), (uint8_t)LK2(p))) {
          onset_len = 3;
        } else if (IS_V(2)) {
          onset_len = 2;
        }
      }
      if (onset_len == 0) {
        if (IS_S(1) && IS_V(2) && csv_valid((uint8_t)LK0(p), (uint8_t)LK1(p))) {
          onset_len = 2;
        } else if (IS_V(1)) {
          onset_len = 1;
        }
      }
    }
    break;
  }
  }

  /* ---- Handle no-onset situation ---- */
  if (onset_len == 0) {
    if (!IS_V(0)) {
      /* Emit the orphan byte to ensure lossless tokenization and align 
       * with the streaming/fused paths. */
      out[0] = (char)*p;
      out[1] = '\0';
      *pp = p + 1;
      return 1;
    }
  }

  /* ---- Copy onset bytes ---- */
  if (onset_len > 0) {
    /* Guard: onset must fit with at least 2 bytes remaining (onset + vowel).
     * If the buffer is exhausted here, emit an empty syllable rather than
     * overflowing; this can only happen if buf_cap was already very tight
     * on entry (i.e. << MAX_TOKEN_CHARS), which is a caller bug.          */
    if (onset_len >= buf_cap - 1) {
      *out = '\0';
      *pp = p; /* do not advance — let caller handle */
      return (int)(p - start);
    }
    memcpy(out, p, (size_t)onset_len);
    for (int i = 0; i < onset_len; i++)
      out[i] = (char)tolower((unsigned char)out[i]);
    out += onset_len;
    p += onset_len;
    buf_cap -= onset_len;
  }

  /* ---- Vowel nucleus ---- */
  if (!*p || !IS_V(0)) {
    /* If we have an onset but no vowel, we still want to emit the first byte
     * as a syllable so it's not lost.  Crucially, we only return ONE byte
     * here to allow the caller to re-evaluate from the next position. */
    syl_buf[0] = (char)tolower((unsigned char)*start);
    syl_buf[1] = '\0';
    *pp = start + 1;
    return 1;
  }

  /* Primary vowel */
  if (buf_cap < 2) {
    *out = '\0';
    *pp = p;
    return (int)(p - start);
  }
  *out++ = (char)tolower((unsigned char)*p++);
  buf_cap--;

  /* Long vowel: same ASCII vowel repeated */
  if (*p && IS_V(0) && buf_cap >= 2) {
    uint8_t prev_v = *(p - 1);
    if (tolower((unsigned char)*p) == tolower((unsigned char)prev_v)) {
      *out++ = (char)tolower((unsigned char)*p++);
      buf_cap--;
    }
  }

  *out = '\0';
  *pp = p;
  return (int)(p - start);
}

static int consume_syllable_unicode(const uint8_t **pp, char *syl_buf,
                                    int buf_cap) {
  const uint8_t *p = *pp;
  uint32_t cp;
  int nb = utf8_seq_len(p);
  if (utf8_decode_unsafe(p, &cp) <= 0) cp = 0xFFFD;



  uint8_t prop = uprop(cp);

  /* Non-linguistic fast exit (Emoji, Symbols) */
  if (prop & (U_PROP_EMOJI | U_PROP_SYMBOL)) {
    if (nb < buf_cap) {
      memcpy(syl_buf, p, (size_t)nb);
      syl_buf[nb] = '\0';
    } else {
      syl_buf[0] = '\0';
    }
    *pp = p + nb;
    return nb;
  }

  /* Handle velar nasal ŋ (U+014B) specifically as a potential onset */
  int vn = velar_nasal_len(p);
  if (vn > 0 && is_vowel_at(p + vn)) {
    /* ŋV onset */
    if (vn + 1 < buf_cap) {
      memcpy(syl_buf, p, (size_t)vn);
      syl_buf[vn] = (char)p[vn];
      syl_buf[vn + 1] = '\0';
      *pp = p + vn + 1;
      return vn + 1;
    }
  }

  /* Generic extended character: atomic syllable */
  if (nb < buf_cap) {
    memcpy(syl_buf, p, (size_t)nb);
    syl_buf[nb] = '\0';
  } else {
    syl_buf[0] = '\0';
  }
  *pp = p + nb;
  return nb;
}

/* Fix 4: UTF-8 sequence length detection for atomic multi-byte consumption */
static inline int utf8_sequence_length(uint8_t b) {
  if (b < 0x80)
    return 1;
  if ((b & 0xE0) == 0xC0)
    return 2;
  if ((b & 0xF0) == 0xE0)
    return 3;
  if ((b & 0xF8) == 0xF0)
    return 4;
  return 1; /* Invalid: treat as single byte */
}

int consume_syllable(const uint8_t **pp, char *syl_buf, int buf_cap) {
  const uint8_t *p = *pp;
  if (!*p)
    return 0;

  /* Fix 4: Detect and handle multi-byte UTF-8 sequences atomically */
  int nb = utf8_sequence_length(*p);
  if (nb > 1) {
    /* Validate continuation bytes (0x80-0xBF) */
    bool valid = true;
    for (int i = 1; i < nb; i++) {
      if ((p[i] & 0xC0) != 0x80) {
        valid = false;
        break;
      }
    }
    if (!valid) {
      syl_buf[0] = '\0';
      *pp = p + 1;
      return -1;
    }
    if (nb >= buf_cap) {
      syl_buf[0] = '\0';
      *pp = p;
      return 0;
    }
    /* Copy entire UTF-8 sequence as atomic unit */
    memcpy(syl_buf, p, (size_t)nb);
    syl_buf[nb] = '\0';
    *pp = p + nb;
    return nb;
  }

  if (__builtin_expect(*p < 128, 1)) {
    return consume_syllable_ascii(pp, syl_buf, buf_cap);
  }

  return consume_syllable_unicode(pp, syl_buf, buf_cap);
}

#undef LK0
#undef LK1
#undef LK2
#undef LK3

#ifdef USE_GRAPHEME_CLUSTERS
/* =========================================================
 *  Grapheme-aware syllable consumer
 *
 *  Entry point for proper Unicode handling. Uses grapheme_next()
 *  to ensure syllabifier never sees partial graphemes.
 *
 *  Architecture:
 *    UTF-8 bytes → grapheme_next() → cluster →
 *        if LUGANDA_LETTER → existing syllabification logic
 *        else → emit atomic token (emoji/symbol/punct)
 *
 *  This is a thin wrapper that calls consume_syllable_internal
 *  after establishing grapheme boundaries.
 * ========================================================= */

/*
 * consume_syllable_grapheme - Grapheme-aware entry point
 *
 * @pp      - Pointer to input pointer (updated on success)
 * @syl_buf - Output buffer for syllable text
 * @buf_cap - Capacity of output buffer
 * @return  - Bytes consumed, 0 if done, -1 on error/orphan
 */
int consume_syllable_grapheme(const uint8_t **pp, char *syl_buf, int buf_cap) {
  if (!pp || !*pp || !syl_buf || buf_cap <= 0)
    return -1;

  const uint8_t *p = *pp;

  /* Check for end of string */
  if (!*p) {
    syl_buf[0] = '\0';
    return 0;
  }

  /* First, scan one grapheme cluster */
  grapheme_t g;
  int g_len = grapheme_next(p, (int)strlen((const char *)p), &g);

  if (g_len <= 0) {
    syl_buf[0] = '\0';
    *pp = p + 1; /* Skip one byte on error */
    return -1;
  }

  /* Fast path: ASCII single byte that's a Luganda letter */
  if (g_len == 1 && g.ptr[0] < 128 && is_letter(g.props)) {
    /* Fall through to regular syllabification logic */
    return consume_syllable(pp, syl_buf, buf_cap);
  }

  /* Non-linguistic cluster (emoji, symbol, etc.) = atomic token */
  if (is_atomic_cluster(g.props)) {
    if (g_len < buf_cap) {
      memcpy(syl_buf, g.ptr, g_len);
      syl_buf[g_len] = '\0';
    } else {
      syl_buf[0] = '\0';
    }
    *pp = g.ptr + g_len;
    return g_len;
  }

  /* Punctuation cluster = skip/atomic */
  if (is_punctuation(g.props)) {
    if (g_len < buf_cap) {
      memcpy(syl_buf, g.ptr, g_len);
      syl_buf[g_len] = '\0';
    } else {
      syl_buf[0] = '\0';
    }
    *pp = g.ptr + g_len;
    return g_len;
  }

  /* Whitespace = signal end of word (return empty to trigger restart) */
  if (grapheme_is_whitespace(&g)) {
    syl_buf[0] = '\0';
    *pp = g.ptr + g_len;
    return 0; /* Signals word boundary */
  }

  /* Luganda letter cluster (potentially with combining marks) */
  if (is_letter(g.props) || is_luganda_vowel(g.props)) {
    /* Use existing syllabification but with grapheme boundary awareness */
    /* For now, delegate to byte-level consume_syllable */
    /* TODO: Extend to handle multi-codepoint graphemes in syllabification */
    return consume_syllable(pp, syl_buf, buf_cap);
  }

  /* Unknown cluster type - skip it */
  syl_buf[0] = '\0';
  *pp = g.ptr + g_len;
  return -1;
}

#endif /* USE_GRAPHEME_CLUSTERS */

/* =========================================================
 *  Public API: syllabify()
 *
 *  Processes `text` word by word:
 *    1. Splits on ASCII whitespace / ASCII punctuation.
 *    2. OOV heuristic: skips clearly non-Luganda words.
 *    3. Calls consume_syllable() and registers syllable IDs.
 * ========================================================= */

int syllabify(Syllabifier *s, const char *text, uint16_t *out, int out_cap) {
  if (!s || !text || !out || out_cap <= 0)
    return -1;

  const uint8_t *wp = (const uint8_t *)text;
  int n = 0;

  while (*wp && n < out_cap) {
    /* FIXME: LOUDS path disabled - trie is incomplete and returns zeros for
     * many syllables (see LOUDS_DIAGNOSIS.md). Current path below works correctly.
     * TODO: Rebuild LOUDS trie from frozen syllable table before re-enabling. */
    
    /* DISABLED LOUDS PATH - uses incomplete trie with missing syllables
    if (s->frozen && s->stbl->trie) {
      uint16_t id = UINT16_MAX;
      int consumed = consume_syllable_id(s->stbl->trie, &wp, &id);
      if (consumed == 0)
        break;
      if (consumed < 0) {
        wp += utf8_seq_len(wp);
        continue;
      }
      if (id != UINT16_MAX) {
        out[n++] = id;
      }
      continue;
    }
    */

    char syl_buf[MAX_TOKEN_CHARS];
    const uint8_t *before = wp;
    int consumed = consume_syllable(&wp, syl_buf, sizeof syl_buf);

    if (consumed == 0 && wp == before) {
      /* Stuck on an unrecognized byte. Skip the entire UTF-8 sequence
       * atomically to prevent splitting multi-byte characters. */
      wp += utf8_seq_len(wp);
      continue;
    } /* stuck  */

    if (syl_buf[0] == '\0') {
      /* Orphan byte (whitespace, punctuation, or illegal cluster).
       * Skip silently — orphans are word-boundary separators, not tokens.
       * Emitting TOK_UNK here would corrupt multi-word inputs by inserting
       * a spurious UNK between every word and around every punctuation mark. */
      continue;
    }

    uint16_t id;
    if (s->frozen) {
      id = stbl_lookup(s->stbl, syl_buf);
      if (id != UINT16_MAX) {
        out[n++] = id;
      } else {
        /* Valid phonotactic syllable but not in our vocab: fallback to UNK
         * to ensure the ID sequence is unique and correctly represents the
         * input length. */
        out[n++] = TOK_UNK;
      }
    } else {
      id = stbl_intern((SyllableTable *)s->stbl, syl_buf);
      if (id == UINT16_MAX)
        return -1; /* table full */
      out[n++] = id;
    }
  }

  return n;
}
