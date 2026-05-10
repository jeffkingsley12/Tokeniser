/*
 * tokenizer.c
 *
 * Top-level pipeline:
 *
 *   tokenizer_build():
 *     1. Create SyllableTable + Syllabifier
 *     2. Syllabify all documents → syllable-ID sequences
 *     3. Train Re-Pair on those sequences
 *     4. Run GrammarPruner (eff_freq → dead-mark → flatten)
 *     5. Extract live token strings from pruner_expand()
 *     6. Build LOUDS trie from token strings
 *
 *   tokenizer_encode():
 *     1. syllabify input
 *     2. louds_tokenize() → token IDs
 *
 *   tokenizer_decode():
 *     Reverse lookup via id_to_token[] array
 *
 *   tokenizer_save() / tokenizer_load():
 *     Flat binary format (header + sections).
 *     Safe for mmap on aligned platforms.
 */

#define _XOPEN_SOURCE 700
#define _DEFAULT_SOURCE

#include "tokenizer.h"
#include "syllable_seeds.h"
#include "truth_layer.h"
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

/* Forward declarations for internal functions */
extern uint16_t stbl_intern(SyllableTable *stbl, const char *text);
extern uint16_t stbl_intern_fixed(SyllableTable *stbl, const char *text,
                                  uint16_t id);
extern int stbl_seed_special(SyllableTable *stbl);
extern uint32_t stbl_seed_phonotactic(SyllableTable *stbl);
extern int stbl_seed_orphan_ascii(SyllableTable *stbl);
extern void stbl_freeze(SyllableTable *stbl);
extern void clean_and_normalize_seed(char *dest, const char *src, size_t max_len);

/* =========================================================
 *  UTF-8 Utility: Sequence Length
 *
 *  Returns the byte length of a UTF-8 sequence starting at `p`.
 *  - 1 byte  = 0xxxxxxx (ASCII)
 *  - 2 bytes = 110xxxxx 10xxxxxx
 *  - 3 bytes = 1110xxxx 10xxxxxx 10xxxxxx
 *  - 4 bytes = 11110xxx 10xxxxxx 10xxxxxx 10xxxxxx
 *  - Invalid continuation byte (10xxxxxx) returns 1 (skip it).
 * ========================================================= */
size_t utf8_seq_len(const uint8_t *p) {
  uint8_t c = p[0];
  if (c < 0x80)
    return 1; /* ASCII */
  if ((c & 0xC0) == 0x80)
    return 1; /* Invalid continuation byte */
  if ((c & 0xE0) == 0xC0)
    return 2; /* 2-byte sequence */
  if ((c & 0xF0) == 0xE0)
    return 3; /* 3-byte sequence */
  if ((c & 0xF8) == 0xF0)
    return 4; /* 4-byte sequence (emoji, etc.) */
  return 1;   /* Invalid start byte */
}

/* Defensive wrapper for utf8_seq_len (UNK-1) */
static inline size_t safe_utf8_len(const uint8_t *p) {
  if (!p)
    return 1;
  size_t len = utf8_seq_len(p);
  return (len >= 1 && len <= 4) ? len : 1;
}
#include <stdatomic.h>

#include <assert.h>
#include <ctype.h>
#include <fcntl.h>
#include <inttypes.h>
#include <libgen.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

/* =========================================================
 *  🚀 HOT PATH OPTIMIZATIONS
 * =========================================================
 *
 *  1) SIMD ASCII pre-scan (AVX2 when available)
 *  2) Branchless vowel/consonant classification tables
 *  3) Cache-aligned lookup tables
 */

#ifdef __AVX2__
#include <immintrin.h>

/* Check if 32 bytes are all ASCII (< 128) using AVX2 */
static inline int is_all_ascii_32(const uint8_t *p) {
  __m256i v = _mm256_loadu_si256((const __m256i *)p);
  __m256i mask = _mm256_set1_epi8(0x80);
  return _mm256_testz_si256(v, mask); /* true if all bytes < 128 */
}
#endif

/* Branchless classification tables - single memory lookup, zero branches */
static const uint8_t vowel_table[256] = {
    ['a'] = 1, ['e'] = 1, ['i'] = 1, ['o'] = 1, ['u'] = 1,
    ['A'] = 1, ['E'] = 1, ['I'] = 1, ['O'] = 1, ['U'] = 1,
};

static const uint8_t consonant_table[256] = {
    ['b'] = 1, ['c'] = 1, ['d'] = 1, ['f'] = 1, ['g'] = 1, ['h'] = 1, ['j'] = 1,
    ['k'] = 1, ['l'] = 1, ['m'] = 1, ['n'] = 1, ['p'] = 1, ['q'] = 1, ['r'] = 1,
    ['s'] = 1, ['t'] = 1, ['v'] = 1, ['w'] = 1, ['x'] = 1, ['y'] = 1, ['z'] = 1,
    ['B'] = 1, ['C'] = 1, ['D'] = 1, ['F'] = 1, ['G'] = 1, ['H'] = 1, ['J'] = 1,
    ['K'] = 1, ['L'] = 1, ['M'] = 1, ['N'] = 1, ['P'] = 1, ['Q'] = 1, ['R'] = 1,
    ['S'] = 1, ['T'] = 1, ['V'] = 1, ['W'] = 1, ['X'] = 1, ['Y'] = 1, ['Z'] = 1,
};

/* Combined letter table (vowel | consonant) for fast word-boundary checks */
const uint8_t is_letter_tbl[256] = {
    ['a'] = 1, ['b'] = 1, ['c'] = 1, ['d'] = 1, ['e'] = 1, ['f'] = 1, ['g'] = 1,
    ['h'] = 1, ['i'] = 1, ['j'] = 1, ['k'] = 1, ['l'] = 1, ['m'] = 1, ['n'] = 1,
    ['o'] = 1, ['p'] = 1, ['q'] = 1, ['r'] = 1, ['s'] = 1, ['t'] = 1, ['u'] = 1,
    ['v'] = 1, ['w'] = 1, ['x'] = 1, ['y'] = 1, ['z'] = 1, ['A'] = 1, ['B'] = 1,
    ['C'] = 1, ['D'] = 1, ['E'] = 1, ['F'] = 1, ['G'] = 1, ['H'] = 1, ['I'] = 1,
    ['J'] = 1, ['K'] = 1, ['L'] = 1, ['M'] = 1, ['N'] = 1, ['O'] = 1, ['P'] = 1,
    ['Q'] = 1, ['R'] = 1, ['S'] = 1, ['T'] = 1, ['U'] = 1, ['V'] = 1, ['W'] = 1,
    ['X'] = 1, ['Y'] = 1, ['Z'] = 1, ['0'] = 1, ['1'] = 1, ['2'] = 1, ['3'] = 1,
    ['4'] = 1, ['5'] = 1, ['6'] = 1, ['7'] = 1, ['8'] = 1, ['9'] = 1,
};

#define IS_VOWEL(c) vowel_table[(uint8_t)(c)]
#define IS_CONSONANT(c) consonant_table[(uint8_t)(c)]

/* ============================================================================
 *  DENSE HASH TABLES (per-instance fast-path tables)
 * ============================================================================
 */

/* The long-lived per-instance tables are built in
 * tokenizer_build_dense_hashes(). The former module-global
 * cv_dense/sp_dense/byte_direct arrays were shared across tokenizer instances
 * and could corrupt a second loaded model. */

/* Optimized syllabify wrapper — can be extended to use SIMD batching */
static inline int syllabify_optimized(const Tokenizer *t, const char *text,
                                      uint16_t *out, uint32_t cap) {
  STAT_INC(t->stats.syllabify_calls);
  int result = syllabify(t->syl, text, out, cap);
  STAT_ADD(t->stats.syllables_produced, result > 0 ? result : 0);
  return result;
}

/*
 * Fast ASCII syllable consumption using branchless tables.
 * Returns bytes consumed (always >= 1 for valid ASCII input).
 * This is the true hot path for Luganda ASCII text.
 *
 * Stats macros (INC_ATOMIC, ADD_ATOMIC) are defined in tokenizer_config.h
 */
static inline int consume_ascii_syllable_fast(const uint8_t *p,
                                              uint16_t *syl_id_out,
                                              const SylTrie *trie) {
  const uint8_t *start = p;
  uint8_t c = *p;

  /* Quick reject: non-letter ASCII */
  if (!IS_VOWEL(c) && !IS_CONSONANT(c)) {
    *syl_id_out = (uint16_t)(SPECIAL_TOKENS_COUNT +
                             c); /* Pass through as raw byte token */
    return 1;
  }

  /* For letters, use trie lookup - optimized for single-char and CV patterns */
  uint32_t node = trie->root_next[c];
  if (node == 0) {
    /* Not in syllable table - emit raw char */
    *syl_id_out = (uint16_t)(SPECIAL_TOKENS_COUNT + c);
    return 1;
  }
  p++;

  uint16_t last_id = trie->nodes[node].id;

  /* Try to extend: look for vowel (CV pattern) or consonant cluster (CCV) */
  while (*p) {
    c = *p;
    /* Fast path: stop at non-letter */
    if (!IS_VOWEL(c) && !IS_CONSONANT(c))
      break;

    /* Trie walk - single level for common case */
    const SylTrieNode *nd = &trie->nodes[node];
    uint32_t next = 0;

    /* Small edge count optimization: unrolled linear search for <= 4 edges */
    if (nd->edge_count <= 4) {
      const SylTrieEdge *edges = &trie->edges[nd->first_edge];
      for (uint16_t i = 0; i < nd->edge_count; i++) {
        if (edges[i].label == c) {
          next = edges[i].next_node;
          break;
        }
      }
    }

    if (next == 0)
      break;

    node = next;
    p++;

    if (trie->nodes[node].id != 0) {
      last_id = trie->nodes[node].id;
    }
  }

  if (last_id != 0) {
    *syl_id_out = (uint16_t)(last_id - 1);
    return (int)(p - start);
  }

  /* Fallback: single char using the seeded byte range (IDs starting at
   * SPECIAL_TOKENS_COUNT) */
  *syl_id_out = (uint16_t)(SPECIAL_TOKENS_COUNT + (uint8_t)*start);
  return 1;
}

int consume_syllable_fast(const Tokenizer *t, const uint8_t **pp,
                          uint16_t *syl_id_out) {
  const uint8_t *p = *pp;
  if (__builtin_expect(p[0] == 0, 0))
    return 0;

  /* Fix 3: Try 1-byte first (common case), then 2-byte UTF-8 */
  uint16_t sid = t->byte_to_syl[p[0]];
  if (__builtin_expect(sid != 0, 1)) {
    *syl_id_out = (uint16_t)(sid - 1);
    *pp = p + 1;
    return 1;
  }

  /* 2-byte UTF-8 path: Luganda characters (0xC0-0xDF prefix) */
  if (p[1] != 0) {
    uint16_t key2 = p[0] | ((uint16_t)p[1] << 8);
    sid = t->byte_to_syl[key2];
    if (__builtin_expect(sid != 0, 0)) {
      *syl_id_out = (uint16_t)(sid - 1);
      *pp = p + 2;
      return 2;
    }
  }

  return consume_syllable_id(t->stbl->trie, pp, syl_id_out);
}

/* =========================================================
 *  Internal: token vocabulary (built from grammar)
 * ========================================================= */

typedef struct {
  char text[MAX_TOKEN_CHARS];
  uint32_t id;
  uint32_t sym; /* grammar symbol this came from */
} TokenEntry;

void tokenizer_rebuild_fast_paths(Tokenizer *t) {
  if (!t || !t->louds || !t->stbl || !t->stbl->trie)
    return;
  const SylTrie *trie = t->stbl->trie;

  memset(t->byte_to_syl, 0, sizeof t->byte_to_syl);
  uint32_t populated = 0;

  for (uint16_t si = 0; si < t->stbl->count; si++) {
    const char *text = t->stbl->entries[si].text;
    size_t len = strlen(text);
    if (len < 1 || len > 2)
      continue;

    /* Walk the SylTrie to find the terminal node for this syllable */
    uint32_t node = 0;
    const uint8_t *bp = (const uint8_t *)text;
    bool valid = true;
    for (size_t j = 0; j < len && valid; j++) {
      uint32_t next = 0;
      if (node == 0) {
        next = trie->root_next[bp[j]];
      } else {
        const SylTrieNode *nd = &trie->nodes[node];
        for (uint16_t k = 0; k < nd->edge_count; k++) {
          if (trie->edges[nd->first_edge + k].label == bp[j]) {
            next = trie->edges[nd->first_edge + k].next_node;
            break;
          }
        }
      }
      if (next == 0) {
        valid = false;
        break;
      }
      node = next;
    }
    if (!valid)
      continue;

    /* Only use if this is a LEAF node — no children means this IS the
     * longest match, so the O(1) shortcut is correct. */
    const SylTrieNode *terminal = &trie->nodes[node];
    if (terminal->id == 0)
      continue; /* not actually terminal */
    if (terminal->edge_count > 0)
      continue; /* can be extended — skip */

    if (len == 1) {
      t->byte_to_syl[(uint8_t)text[0]] = (uint16_t)(si + 1);
    } else {
      uint16_t key =
          (uint16_t)(uint8_t)text[0] | (uint16_t)((uint8_t)text[1] << 8);
      t->byte_to_syl[key] = (uint16_t)(si + 1);
    }
    populated++;
  }

  memset(t->cv_to_token, 0, sizeof t->cv_to_token);
  memset(t->byte_to_token, 0, sizeof t->byte_to_token);
  uint32_t cv_populated = 0;
  if (t->louds) {
    for (uint32_t tid = 0; tid < t->vocab_size; tid++) {
      if (!t->id_to_str)
        break;
      const char *str = t->id_to_str[tid];
      if (str) {
        if (str[0] != '\0' && str[1] != '\0' && str[2] == '\0') {
          uint16_t key = (uint8_t)str[0] | ((uint16_t)(uint8_t)str[1] << 8);
          if (tid >= t->vocab_size)
            continue;
          if (tid + 1 <= 0xFFFFFFu) {
            /* Pack: (tid+1) in upper 24 bits, length (2) in lower 8 bits */
            t->cv_to_token[key] = ((tid + 1) << 8) | 2;
            cv_populated++;
          }
        } else if (str[0] != '\0' && str[1] == '\0') {
          if (tid >= t->vocab_size)
            continue;
          t->byte_to_token[(uint8_t)str[0]] = tid + 1;
        }
      }
    }
  }

  /* Build space-prefixed fast-path table for ▁CV / ▁CVC patterns */
  memset(t->space_cv_to_token, 0, sizeof t->space_cv_to_token);
  uint32_t space_populated = 0;

  if (t->louds) {
    for (uint32_t tid = 0; tid < t->vocab_size; tid++) {
      const char *str = t->id_to_str[tid];
      if (!str)
        continue;

      /* Detect ▁ prefix (U+2581 = 0xE2 0x96 0x81) */
      if ((uint8_t)str[0] == 0xE2 && (uint8_t)str[1] == 0x96 &&
          (uint8_t)str[2] == 0x81) {
        /*
         * Use strlen to get the exact byte count of the suffix.
         * This avoids the utf8_seq_len collision bug where ▁ba and ▁be
         * both looked like 'b'.
         */
        size_t rem_len = strlen(str + 3);

        /* Index strictly by raw byte length of the suffix */
        if (rem_len == 1 || rem_len == 2) {
          uint8_t c1 = (uint8_t)str[3];
          uint8_t c2 = (rem_len == 2) ? (uint8_t)str[4] : 0;
          uint16_t key = c1 | ((uint16_t)c2 << 8);

          if (key != 0) {
            /* Pack: (tid+1) in upper 24 bits, length in lower 8 bits */
            t->space_cv_to_token[key] = ((tid + 1) << 8) | (uint8_t)rem_len;
            space_populated++;
          }
        }
      }
    }
  }

  fprintf(stderr,
          "[tokenizer_rebuild_fast_paths] byte_to_syl: %u entries, "
          "cv_to_token: %u entries, space_cv_to_token: %u entries\n",
          populated, cv_populated, space_populated);
}

void tokenizer_rebuild_id_to_str(Tokenizer *t) {
  if (!t || !t->stbl)
    return;
  if (t->id_to_str)
    free(t->id_to_str);

  t->id_to_str = calloc(t->vocab_size, sizeof(char *));
  if (!t->id_to_str)
    return;

  uint32_t syl_id_base = 0;
  uint32_t gram_id_base = syl_id_base + t->stbl->count;
  uint32_t morph_id_base = gram_id_base + (t->rs ? t->rs->rule_count : 0);

  /* 1. Syllables */
  for (uint32_t si = 0; si < t->stbl->count && si < t->vocab_size; si++) {
    t->id_to_str[syl_id_base + si] = t->stbl->entries[si].text;
  }

  /* 2. Grammar Rules (Subwords) */
  if (t->gp_expansions && t->rs) {
    for (uint32_t ri = 0; ri < t->rs->rule_count; ri++) {
      uint32_t tid = gram_id_base + ri;
      if (tid < t->vocab_size) {
        if (t->gp_expansions[ri] != NULL) {
          t->id_to_str[tid] = t->gp_expansions[ri];
        }
      }
    }
  }

  /* 3. Morpheme Seeds */
  if (t->morph_strings) {
    for (uint32_t mi = 0; mi < MORPHEME_SEED_COUNT; mi++) {
      uint32_t tid = morph_id_base + mi;
      if (tid < t->vocab_size && t->morph_strings[mi]) {
        t->id_to_str[tid] = t->morph_strings[mi];
      }
    }
  }
}

/* Fix 1: Build dense hash tables from sparse arrays for cache efficiency */
/* Golden ratio constant for multiplicative hashing — Knuth's recommended
 * hash multiplier for 32-bit integers. Distributes keys uniformly. */
#define HASH_MULTIPLIER 0x9E3779B1u

static inline uint32_t dense_hash_idx(uint16_t k) {
  return ((uint32_t)k * HASH_MULTIPLIER) >> 24;
}

static void tokenizer_build_dense_hashes(Tokenizer *t) {
  if (!t)
    return;

  /* Clear dense tables */
  memset(t->cv_dense_keys, 0, sizeof(t->cv_dense_keys));
  memset(t->cv_dense_vals, 0, sizeof(t->cv_dense_vals));
  memset(t->sp_dense_keys, 0, sizeof(t->sp_dense_keys));
  memset(t->sp_dense_vals, 0, sizeof(t->sp_dense_vals));
  memset(t->byte_direct, 0, sizeof(t->byte_direct));

  /* Populate byte_direct: upper 16 bits = token+1, lower 16 = advance */
  for (int i = 0; i < 256; i++) {
    if (t->byte_to_token[i]) {
      t->byte_direct[i] = ((uint32_t)t->byte_to_token[i] << 16) | 1;
    }
  }

  /* Populate cv_dense from sparse cv_to_token */
  uint32_t cv_count = 0;
  for (uint32_t i = 0; i < 65536; i++) {
    if (t->cv_to_token[i]) {
      uint32_t h = dense_hash_idx((uint16_t)i) & (DENSE_HASH_SIZE - 1);
      /* Linear probing with wrap */
      for (int probe = 0; probe < 4 && t->cv_dense_vals[h]; probe++) {
        h = (h + 1) & (DENSE_HASH_SIZE - 1);
      }
      if (!t->cv_dense_vals[h]) {
        t->cv_dense_keys[h] = (uint16_t)i;
        t->cv_dense_vals[h] = t->cv_to_token[i];
        cv_count++;
      }
    }
  }

  /* Populate sp_dense from sparse space_cv_to_token */
  uint32_t sp_count = 0;
  for (uint32_t i = 0; i < 65536; i++) {
    if (t->space_cv_to_token[i]) {
      uint32_t h = dense_hash_idx((uint16_t)i) & (DENSE_HASH_SIZE - 1);
      for (int probe = 0; probe < 4 && t->sp_dense_vals[h]; probe++) {
        h = (h + 1) & (DENSE_HASH_SIZE - 1);
      }
      if (!t->sp_dense_vals[h]) {
        t->sp_dense_keys[h] = (uint16_t)i;
        t->sp_dense_vals[h] = t->space_cv_to_token[i];
        sp_count++;
      }
    }
  }

  fprintf(
      stderr,
      "[tokenizer_build_dense] cv_dense: %u entries, sp_dense: %u entries\n",
      cv_count, sp_count);
}

/* Lookup in cv_dense hash table. Returns 0 on miss. */
static inline uint32_t dense_cv_get(const Tokenizer *t, uint16_t key) {
  uint32_t h = dense_hash_idx(key) & (DENSE_HASH_SIZE - 1);
  for (int i = 0; i < 4; i++) {
    if (t->cv_dense_vals[h] == 0)
      return 0; /* Empty slot = miss */
    if (t->cv_dense_keys[h] == key)
      return t->cv_dense_vals[h];
    h = (h + 1) & (DENSE_HASH_SIZE - 1);
  }
  return 0; /* Not found after max probes */
}

/* Lookup in sp_dense hash table. Returns 0 on miss. */
static inline uint32_t dense_sp_get(const Tokenizer *t, uint16_t key) {
  uint32_t h = dense_hash_idx(key) & (DENSE_HASH_SIZE - 1);
  for (int i = 0; i < 4; i++) {
    if (t->sp_dense_vals[h] == 0)
      return 0; /* Empty slot = miss */
    if (t->sp_dense_keys[h] == key)
      return t->sp_dense_vals[h];
    h = (h + 1) & (DENSE_HASH_SIZE - 1);
  }
  return 0; /* Not found after max probes */
}

/* =========================================================
 *  tokenizer_build
 * ========================================================= */

typedef struct {
  char *text;
  size_t len;
} UniqueSyllable;

static int compare_syllables(const void *a, const void *b) {
  const UniqueSyllable *sa = (const UniqueSyllable *)a;
  const UniqueSyllable *sb = (const UniqueSyllable *)b;
  return strcmp(sa->text, sb->text);
}

Tokenizer *tokenizer_build(const char **docs, uint32_t n_docs) {
  if (!docs || n_docs == 0)
    return NULL;

  Tokenizer *t = calloc(1, sizeof *t);
  if (!t)
    return NULL;

  /* --- 1. Syllable table + syllabifier --- */
  t->stbl = calloc(1, sizeof(SyllableTable));
  if (!t->stbl)
    goto fail;

  /* Seed special tokens first (IDs 0..SPECIAL_TOKENS_COUNT-1) */
  if (stbl_seed_special(t->stbl) != 0) {
    fprintf(stderr, "[tokenizer_build] Failed to seed special tokens\n");
    goto fail;
  }

  /* Seed full byte range (0..255) starting after special tokens.
   * This ensures bytes are at IDs
   * SPECIAL_TOKENS_COUNT..SPECIAL_TOKENS_COUNT+255 */
  for (int i = 0; i < 256; i++) {
    char s[2] = {(char)i, 0};
    uint16_t id = SPECIAL_TOKENS_COUNT + i;
    if (stbl_intern_fixed(t->stbl, s, id) == UINT16_MAX) {
      fprintf(stderr, "[tokenizer_build] Failed to seed byte %d at ID %u\n", i,
              id);
      goto fail;
    }
  }

  /* Seed phonotactic syllables (CV patterns, etc.) starting above byte range */
  uint32_t phono_seeded = stbl_seed_phonotactic(t->stbl);
  if (phono_seeded == 0) {
    fprintf(stderr, "[tokenizer_build] Warning: No phonotactic seeds loaded\n");
  }

  /* Unified admission: seed every printable ASCII character that
   * consume_syllable_ascii() can emit as a single-byte orphan.
   * This MUST happen before any corpus processing so that the IDs
   * are stable and the stbl is a superset of the scanner's output
   * domain.  Prevents TOK_UNK holes in LOUDS edge-label sequences. */
  if (stbl_seed_orphan_ascii(t->stbl) < 0) {
    fprintf(stderr, "[tokenizer_build] Error: ASCII orphan seeding failed\n");
    goto fail;
  }

  t->syl = syllabifier_create(t->stbl);
  if (!t->syl)
    goto fail;

  /* Syllabifier stays UNFROZEN during build so it can intern atoms from
   * seeds/corpus. */
  (void)vocab_validate_morphemes(t);

  /* Morpheme seeds were interned via vocab_validate_morphemes() above. */

  /* --- 2. Two-pass deterministic syllable discovery --- */

  /* Pass 1: Collect all unique syllables from corpus */
  UniqueSyllable *unique_sylls = NULL;
  size_t unique_count = 0;
  size_t unique_capacity = 1024;

  unique_sylls = malloc(unique_capacity * sizeof(UniqueSyllable));
  if (!unique_sylls) {
    goto fail;
  }

  /* Temporary syllabifier for collection (unfrozen) */
  Syllabifier *temp_syl = syllabifier_create(t->stbl);
  if (!temp_syl) {
    free(unique_sylls);
    goto fail;
  }

  for (uint32_t i = 0; i < n_docs; i++) {
    uint16_t buf[MAX_SYLLABLES];
    int n = syllabify_optimized(t, docs[i], buf, MAX_SYLLABLES);
    if (n < 0)
      n = 0;

    for (int j = 0; j < n; j++) {
      /* Use tokenizer_decode to get the syllable text */
      const char *decoded = tokenizer_decode(t, buf[j]);
      if (decoded && strlen(decoded) > 0) {
        /* Check if already collected */
        int found = 0;
        for (size_t k = 0; k < unique_count; k++) {
          if (strcmp(unique_sylls[k].text, decoded) == 0) {
            found = 1;
            break;
          }
        }

        if (!found) {
          /* Add to unique list */
          if (unique_count >= unique_capacity) {
            unique_capacity *= 2;
            UniqueSyllable *tmp =
                realloc(unique_sylls, unique_capacity * sizeof(UniqueSyllable));
            if (!tmp) {
              syllabifier_destroy(temp_syl);
              for (size_t k = 0; k < unique_count; k++)
                free(unique_sylls[k].text);
              free(unique_sylls);
              goto fail;
            }
            unique_sylls = tmp;
          }

          unique_sylls[unique_count].text = strdup(decoded);
          unique_sylls[unique_count].len = strlen(decoded);
          unique_count++;
        }
      }
    }
  }

  syllabifier_destroy(temp_syl);

  /* Sort syllables lexicographically for deterministic ID assignment */
  qsort(unique_sylls, unique_count, sizeof(UniqueSyllable), compare_syllables);

  /* Pass 2: Assign IDs in sorted order using a fresh syllable table */
  /* Create a new syllable table for deterministic assignment */
  SyllableTable *deterministic_stbl = calloc(1, sizeof(SyllableTable));
  if (!deterministic_stbl) {
    for (size_t i = 0; i < unique_count; i++) {
      free(unique_sylls[i].text);
    }
    free(unique_sylls);
    goto fail;
  }

  /* Seed special tokens first (same as original) */
  if (stbl_seed_special(deterministic_stbl) != 0) {
    fprintf(stderr, "[tokenizer_build] Failed to seed special tokens in "
                    "deterministic table\n");
    free(deterministic_stbl);
    for (size_t i = 0; i < unique_count; i++) {
      free(unique_sylls[i].text);
    }
    free(unique_sylls);
    goto fail;
  }

  /* Seed full byte range */
  for (int i = 0; i < 256; i++) {
    char s[2] = {(char)i, 0};
    uint16_t id = SPECIAL_TOKENS_COUNT + i;
    if (stbl_intern_fixed(deterministic_stbl, s, id) == UINT16_MAX) {
      fprintf(stderr,
              "[tokenizer_build] Failed to seed byte %d at ID %u in "
              "deterministic table\n",
              i, id);
      free(deterministic_stbl);
      for (size_t i = 0; i < unique_count; i++) {
        free(unique_sylls[i].text);
      }
      free(unique_sylls);
      goto fail;
    }
  }

  /* Seed phonotactic syllables */
  uint32_t phono_seeded_det = stbl_seed_phonotactic(deterministic_stbl);
  if (phono_seeded_det == 0) {
    fprintf(stderr, "[tokenizer_build] Warning: No phonotactic seeds loaded in "
                    "deterministic table\n");
  }

  /* Assign IDs in alphabetical order */
  for (size_t i = 0; i < unique_count; i++) {
    uint16_t id = stbl_intern(deterministic_stbl, unique_sylls[i].text);
    if (id == UINT16_MAX) {
      fprintf(stderr, "[tokenizer_build] Failed to intern syllable %zu: %s\n",
              i, unique_sylls[i].text);
      free(deterministic_stbl);
      for (size_t j = 0; j < unique_count; j++) {
        free(unique_sylls[j].text);
      }
      free(unique_sylls);
      goto fail;
    }
  }

  /* Replace the original syllable table with the deterministic one */
  syllabifier_destroy(t->syl);
  t->syl = NULL;          /* prevent double-free if create fails below */
  free(t->stbl);
  t->stbl = deterministic_stbl;

  /* Enforce invariant: entries[i].id must equal i for correct trie traversal */
  for (uint16_t i = 0; i < t->stbl->count; i++) {
    t->stbl->entries[i].id = i;
  }

  t->syl = syllabifier_create(t->stbl);
  if (!t->syl) {
    fprintf(stderr, "[tokenizer_build] Failed to create syllabifier with "
                    "deterministic table\n");
    for (size_t i = 0; i < unique_count; i++) {
      free(unique_sylls[i].text);
    }
    free(unique_sylls);
    goto fail;
  }

  /* Syllable table is now deterministic with fixed IDs */

  /* Clean up temporary syllable collection */
  for (size_t i = 0; i < unique_count; i++) {
    free(unique_sylls[i].text);
  }
  free(unique_sylls);

  /* --- 3. Re-syllabify all documents with deterministic IDs --- */
  uint16_t **seqs = safe_calloc(n_docs, sizeof *seqs);
  uint32_t *lens = safe_calloc(n_docs, sizeof *lens);
  if (!seqs || !lens) {
    free(seqs);
    free(lens);
    goto fail;
  }

  for (uint32_t i = 0; i < n_docs; i++) {
    uint16_t *buf = malloc(MAX_SYLLABLES * sizeof *buf);
    if (!buf) {
      for (uint32_t j = 0; j < i; j++)
        free(seqs[j]);
      free(seqs);
      free(lens);
      goto fail;
    }
    int n = syllabify_optimized(t, docs[i], buf, MAX_SYLLABLES);
    if (n < 0)
      n = 0;
    seqs[i] = buf;
    lens[i] = (uint32_t)n;
  }

  /* --- 3. Truth Layer Annotation --- */
  TruthTrie *tt = truth_trie_create();
  MergeMask **masks = safe_calloc(n_docs, sizeof(MergeMask *));
  if (tt && masks) {
    truth_trie_build_from_seed(tt, t);
    for (uint32_t i = 0; i < n_docs; i++) {
      if (lens[i] < 2)
        continue;

      masks[i] = merge_mask_create(lens[i] - 1);
      if (!masks[i])
        continue;

      /* Default: all merges allowed */
      for (size_t j = 0; j < lens[i] - 1; j++)
        set_merge_allowed(masks[i], j, true);

      /* Annotate with truth matches (Boundary-Only Locking) */
      for (uint32_t j = 0; j < lens[i]; j++) {
        TruthMatchInfo match;
        int match_len = truth_match_ids(tt, seqs[i] + j, lens[i] - j, &match);
        if (match_len > 1) {
          /* Lock boundary before the span */
          if (j > 0)
            set_merge_allowed(masks[i], j - 1, false);

          /* Lock boundary after the span */
          if (j + match_len - 1 < lens[i] - 1) {
            set_merge_allowed(masks[i], j + match_len - 1, false);
          }

          /* Skip ahead, but internal merges remain allowed */
          j += (uint32_t)(match_len - 1);
        }
      }
    }
  }

  /* --- 4. Re-Pair training --- */
  t->rs = repair_create();
  if (!t->rs) {
    /* Cleanup masks and truth trie */
    if (masks) {
      for (uint32_t i = 0; i < n_docs; i++)
        merge_mask_destroy(masks[i]);
      free(masks);
    }
    truth_trie_destroy(tt);
    for (uint32_t i = 0; i < n_docs; i++)
      free(seqs[i]);
    free(seqs);
    free(lens);
    goto fail;
  }

  if (repair_train_with_mask(t->rs, (const uint16_t **)seqs, lens, n_docs,
                             (const MergeMask **)masks) != 0) {
    if (masks) {
      for (uint32_t i = 0; i < n_docs; i++)
        merge_mask_destroy(masks[i]);
      free(masks);
    }
    truth_trie_destroy(tt);
    for (uint32_t i = 0; i < n_docs; i++)
      free(seqs[i]);
    free(seqs);
    free(lens);
    goto fail;
  }

  /* Cleanup masks and truth trie after training */
  if (masks) {
    for (uint32_t i = 0; i < n_docs; i++)
      merge_mask_destroy(masks[i]);
    free(masks);
  }
  truth_trie_destroy(tt);

  /* Finalize the syllable table: no more interning allowed.
   * This ensures stbl->count is stable for ID base calculation. */
  t->syl->frozen = true;
  stbl_freeze(
      t->stbl); /* Also freeze the underlying table to block stbl_intern */

  for (uint32_t i = 0; i < n_docs; i++)
    free(seqs[i]);
  free(seqs);
  free(lens);

  /* --- 4. Grammar pruning --- */
  t->gp = pruner_create(t->rs, t->stbl);
  if (!t->gp)
    goto fail;

  /* Pass 1: effective frequency accounting.
   * OOM here causes every rule to be marked dead and an unusable tokenizer.
   * Treat false return as a fatal build error.                              */
  if (!pruner_compute_eff_freq(t->gp)) {
    fprintf(stderr,
            "[tokenizer_build] pruner_compute_eff_freq OOM — aborting\n");
    goto fail;
  }

  /* Pass 2: mark rules whose eff_freq < MIN_PAIR_FREQ as dead.          */
  pruner_mark_dead(t->gp);
  pruner_expand_all(t->gp, t->stbl);
  if (t->gp->fatal_oom) {
    fprintf(stderr,
            "[tokenizer_build] OOM expanding grammar rules — aborting\n");
    goto fail;
  }

  /* Rule expansion is lazy: pruner_expand() is called on demand below
   * (and at decode time after load).  Eager expansion is unnecessary
   * because tokenizer_encode/decode are read-only after construction
   * and the expansion cache (gp->expansions) is safe for concurrent
   * readers once fully written.  If you need strict thread-safety for
   * concurrent builds, call pruner_expand() on every live rule here.    */

  /* --- 5. Finalize Vocabulary IDs --- */

  /* All morpheme syllables were interned during pass 1. */

  uint32_t syl_id_base = 0;
  uint32_t gram_id_base = syl_id_base + t->stbl->count;
  uint32_t morph_id_base = gram_id_base + t->rs->rule_count;

  /* Safety check: verify token_cap won't overflow */
  if (t->rs->rule_count > UINT32_MAX - 16 - t->stbl->count - MORPHEME_SEED_COUNT) {
    fprintf(stderr, "[tokenizer_build] token capacity calculation would overflow\n");
    goto fail;
  }
  uint32_t token_cap =
      t->rs->rule_count + t->stbl->count + MORPHEME_SEED_COUNT + 16;
  char **tok_strings = calloc(token_cap, sizeof *tok_strings);
  uint32_t *tok_ids = calloc(token_cap, sizeof *tok_ids);
  if (!tok_strings || !tok_ids) {
    free(tok_strings);
    free(tok_ids);
    goto fail;
  }

  uint32_t n_tokens = 0;

  /* Pass 1: Syllables */
  for (uint16_t si = 0; si < t->stbl->count; si++) {
    const char *text = t->stbl->entries[si].text;
    uint16_t tmp_syls[MAX_SEQ_LEN];
    int n = syllabify_optimized(t, text, tmp_syls, MAX_SEQ_LEN);
    bool has_unk = false;
    for (int j = 0; j < n; j++) if (tmp_syls[j] == TOK_UNK) { has_unk = true; break; }
    
    if (n > 0 && !has_unk) {
      tok_strings[n_tokens] = (char *)text;
      tok_ids[n_tokens] = syl_id_base + si;
      n_tokens++;
    }
  }

  /* Pass 2: Grammar rules */
  for (uint32_t ri = 0; ri < t->rs->rule_count; ri++) {
    Rule *r = &t->rs->rules[ri];
    if (r->dead)
      continue;
    const char *exp = pruner_expand(t->gp, r->lhs, t->stbl);
    if (!exp || exp[0] == '\0')
      continue;

    /* Verify no TOK_UNK in expansion */
    uint16_t tmp_syls[MAX_SEQ_LEN];
    int n = syllabify_optimized(t, exp, tmp_syls, MAX_SEQ_LEN);
    bool has_unk = false;
    for (int j = 0; j < n; j++) if (tmp_syls[j] == TOK_UNK) { has_unk = true; break; }
    
    if (n > 0 && !has_unk) {
      tok_strings[n_tokens] = (char *)exp;
      tok_ids[n_tokens] = gram_id_base + ri;
      n_tokens++;
    }
  }

  /* Pass 3: Morpheme seeds */
  t->morph_strings = calloc(MORPHEME_SEED_COUNT, sizeof(char *));
  for (uint32_t mi = 0; mi < MORPHEME_SEED_COUNT; mi++) {
    const char *m = MORPHEME_SEEDS[mi];
    if (!m)
      break;
    
    char clean_buf[MAX_TOKEN_CHARS];
    clean_and_normalize_seed(clean_buf, m, sizeof(clean_buf));
    if (clean_buf[0] == '\0')
      continue;
    
    uint16_t syls[MAX_SEQ_LEN];
    int n = syllabify_optimized(t, clean_buf, syls, MAX_SEQ_LEN);
    if (n > 0) {
      /* Verify no TOK_UNK was produced. If the seed contains characters not in
       * our SyllableTable (e.g. English definitions mislabeled as Luganda),
       * we skip the token to prevent [louds_build] warnings and trie holes. */
      bool has_unk = false;
      for (int j = 0; j < n; j++) {
        if (syls[j] == TOK_UNK) {
          has_unk = true;
          break;
        }
      }
      if (!has_unk) {
        t->morph_strings[mi] = strdup(clean_buf);
        tok_strings[n_tokens] = t->morph_strings[mi];
        tok_ids[n_tokens] = morph_id_base + mi;
        n_tokens++;
      }
    }
  }

  /* Build the deterministic byte-trie for fast O(1) syllable matching */
  t->stbl->trie = syltrie_build(t->stbl);
  if (!t->stbl->trie) {
    fprintf(stderr, "[tokenizer_build] failed to build syllable trie\n");
    goto fail;
  }

  /* --- 6. Finalize Vocab and Build LOUDS Trie --- */
  t->vocab_size = morph_id_base + MORPHEME_SEED_COUNT;
  t->louds = louds_build((const char **)tok_strings, tok_ids, n_tokens, t->stbl,
                         t->syl);
  free(tok_strings);
  free(tok_ids);

  if (!t->louds)
    goto fail;

  tokenizer_rebuild_id_to_str(t);

  tokenizer_rebuild_fast_paths(t);
  tokenizer_build_dense_hashes(t); /* Build per-instance dense hash tables */

  return t;

fail:
  tokenizer_destroy(t);
  return NULL;
}

/* =========================================================
 *  tokenizer_destroy
 * ========================================================= */

void tokenizer_destroy(Tokenizer *t) {
  if (!t)
    return;

  if (t->morph_strings) {
    for (uint32_t i = 0; i < MORPHEME_SEED_COUNT; i++)
      free(t->morph_strings[i]);
    free(t->morph_strings);
  }

  /* Free id_to_str pointer array first (strings are owned by components) */
  free(t->id_to_str);

  /* Free loaded expansion strings (when gp is NULL after load) */
  if (t->gp_expansions) {
    for (uint32_t i = 0; i < t->rs->rule_count; i++)
      free(t->gp_expansions[i]);
    free(t->gp_expansions);
  }

  /* Destroy components in dependency order */
  louds_destroy(t->louds);
  pruner_destroy(t->gp);
  repair_destroy(t->rs);

  if (t->stbl) {
    if (t->stbl->trie)
      syltrie_destroy(t->stbl->trie);
    free(t->stbl);
  }

  syllabifier_destroy(t->syl);
  free(t);
}

/* =========================================================
 *  tokenizer_encode
 * ========================================================= */

int tokenizer_encode(const Tokenizer *t, const char *text, uint32_t *out,
                     uint32_t out_cap) {
  if (!t || !text || !out)
    return -1;

  uint16_t syls[MAX_SYLLABLES];
  int n_syls = syllabify_optimized(t, text, syls, MAX_SYLLABLES);
  if (n_syls <= 0)
    return 0;

  return louds_tokenize(t->louds, syls, (uint32_t)n_syls, out, out_cap);
}

/* O(1) reverse decode using the pre-built id_to_str table (H-03).
 * Input token_id is the public ID; now identical to internal ID.
 * Special tokens (0..SPECIAL_TOKENS_COUNT-1) are valid and decode to their
 * strings. */
const char *tokenizer_decode(const Tokenizer *t, uint32_t token_id) {
  if (!t || !t->id_to_str)
    return NULL;
  if (token_id >= t->vocab_size)
    return NULL;
  return t->id_to_str[token_id];
}

/* =========================================================
 *  Serialization format
 *
 *  [ MAGIC (8 bytes) ]
 *  [ stbl_count  u32 ]
 *  [ stbl entries: stbl_count × MAX_TOKEN_CHARS ]
 *  [ rule_count  u32 ]
 *  [ rules: rule_count × sizeof(Rule) ]
 *  [ expansions: rule_count × MAX_TOKEN_CHARS ]
 *  [ expanded flags: rule_count × 1 byte ]
 *  [ louds: bv_len u32, bv_words × u64, label_count u32,
 *           labels × u16, token_ids × u32,
 *           sb_count u32, rank1_sb × u32 ]
 * ========================================================= */

/* Serialisation magic + format version.
 * Byte layout: "LUGTOK1" (7 chars) + version byte.
 * v1 = initial; v2 = hash-table fixes; v3 = BASE_SYMBOL_OFFSET 1024→257.
 * v4 = CRC32 checksum for corruption detection.
 * Multi-byte integers are written in host byte order.  Models are not
 * portable across architectures with different endianness.              */
#define ALIGN8(x) (((x) + 7ULL) & ~7ULL)

int tokenizer_save(const Tokenizer *t, const char *path) {
  if (!t || !path)
    return -1;

  FILE *f = fopen(path, "wb+");
  if (!f) {
    perror("tokenizer_save: fopen");
    return -1;
  }

  /* -------------------------------------------------------------------------
   * 1. Prepare Header
   * -------------------------------------------------------------------------
   */
  MmapHeader hdr = {0};
  memcpy(hdr.magic, "LGMMAPv1", 8);
  hdr.version = 15;
  hdr.vocab_size = t->vocab_size;
  hdr.rule_count = t->rs ? t->rs->rule_count : 0;
  hdr.truth_seed_hash = compute_truth_seed_hash();

  /* Placeholder for the header — we will rewrite it at the end once all
   * offsets and the total data_size are known. */
  if (fwrite(&hdr, sizeof(hdr), 1, f) != 1)
    goto io_err;

  /* -------------------------------------------------------------------------
   * 2. SyllableTable Section
   * -------------------------------------------------------------------------
   */
  fseeko(f, ALIGN8(ftello(f)), SEEK_SET);
  hdr.stbl_offset = (uint64_t)ftello(f);

  /* Zero out pointer before writing to disk to avoid garbage on reload */
  SyllableTable st_copy = *t->stbl;
  st_copy.trie = NULL;
  if (fwrite(&st_copy, sizeof(SyllableTable), 1, f) != 1)
    goto io_err;

  /* -------------------------------------------------------------------------
   * 3. RePairState Rules Section
   * -------------------------------------------------------------------------
   */
  if (t->rs && t->rs->rule_count > 0) {
    fseeko(f, ALIGN8(ftello(f)), SEEK_SET);
    hdr.rules_offset = (uint64_t)ftello(f);
    if (fwrite(t->rs->rules, sizeof(Rule), t->rs->rule_count, f) !=
        t->rs->rule_count)
      goto io_err;
  }

  /* -------------------------------------------------------------------------
   * 4. LOUDS CSR Section
   * -------------------------------------------------------------------------
   */
  if (t->louds && t->louds->has_csr) {
    fseeko(f, ALIGN8(ftello(f)), SEEK_SET);
    hdr.louds_offset = (uint64_t)ftello(f);

    /* Zero out all pointers before writing LOUDS header */
    LOUDS l_copy = *t->louds;
    l_copy.row_ptr = NULL;
    l_copy.labels = NULL;
    l_copy.next_node = NULL;
    l_copy.terminals = NULL;
    l_copy.klens = NULL;
    l_copy.labels_legacy = NULL;
    l_copy.token_ids = NULL;
    l_copy.koffs = NULL;
    l_copy.rank1_sb = NULL;
    l_copy.louds_bv = NULL;

    if (fwrite(&l_copy, sizeof(LOUDS), 1, f) != 1)
      goto io_err;

    /* Write Split CSR arrays (each ALIGN8) */
    off_t block_start = ftello(f);
    (void)block_start; /* debugging */

    fseeko(f, ALIGN8(ftello(f)), SEEK_SET);
    if (fwrite(t->louds->row_ptr, sizeof(uint32_t), t->louds->node_count + 1,
               f) != t->louds->node_count + 1)
      goto io_err;

    fseeko(f, ALIGN8(ftello(f)), SEEK_SET);
    if (fwrite(t->louds->labels, sizeof(uint16_t), t->louds->edge_count, f) !=
        t->louds->edge_count)
      goto io_err;

    fseeko(f, ALIGN8(ftello(f)), SEEK_SET);
    if (fwrite(t->louds->next_node, sizeof(uint32_t), t->louds->edge_count,
               f) != t->louds->edge_count)
      goto io_err;

    fseeko(f, ALIGN8(ftello(f)), SEEK_SET);
    if (fwrite(t->louds->terminals, sizeof(uint32_t), t->louds->edge_count,
               f) != t->louds->edge_count)
      goto io_err;
  }

  /* -------------------------------------------------------------------------
   * 5. Expansions Section
   * -------------------------------------------------------------------------
   */
  if (t->gp && t->gp->expansions && t->rs->rule_count > 0) {
    fseeko(f, ALIGN8(ftello(f)), SEEK_SET);
    hdr.strings_offset = (uint64_t)ftello(f);
    hdr.strings_size = t->rs->rule_count * MAX_TOKEN_CHARS;

    for (uint32_t i = 0; i < t->rs->rule_count; i++) {
      char buf[MAX_TOKEN_CHARS] = {0};
      if (t->gp->expansions[i]) {
        strncpy(buf, t->gp->expansions[i], MAX_TOKEN_CHARS - 1);
      }
      if (fwrite(buf, MAX_TOKEN_CHARS, 1, f) != 1)
        goto io_err;
    }
  }

  /* -------------------------------------------------------------------------
   * 6. Finalize Header
   * -------------------------------------------------------------------------
   */
  hdr.data_size = (uint64_t)ftello(f);
  fseeko(f, 0, SEEK_SET);
  if (fwrite(&hdr, sizeof(hdr), 1, f) != 1)
    goto io_err;

  /* -------------------------------------------------------------------------
   * 7. Append CRC32 for integrity (Security Fix)
   * -------------------------------------------------------------------------
   */
  uint32_t crc = 0;
  fseeko(f, 0, SEEK_SET);

  /* Calculate CRC of the entire file content written so far */
  uint8_t crc_buf[8192];
  uint64_t remaining = hdr.data_size;
  while (remaining > 0) {
    size_t to_read =
        (remaining > sizeof(crc_buf)) ? sizeof(crc_buf) : (size_t)remaining;
    if (fread(crc_buf, 1, to_read, f) != to_read) {
      fprintf(stderr, "[save] error reading back for CRC calculation\n");
      goto io_err;
    }
    crc = crc32_update(crc, crc_buf, to_read);
    remaining -= to_read;
  }

  fseeko(f, 0, SEEK_END);
  if (fwrite(&crc, sizeof(crc), 1, f) != 1)
    goto io_err;

  fclose(f);
  return 0;

io_err:
  fprintf(stderr, "[save] I/O error writing model to %s\n", path);
  fclose(f);
  unlink(path);
  return -1;
}

Tokenizer *tokenizer_load(const char *path) {
  if (!path)
    return NULL;

  FILE *f = fopen(path, "rb");
  if (!f) {
    perror("tokenizer_load: fopen");
    return NULL;
  }

  MmapHeader hdr;
  if (fread(&hdr, sizeof(hdr), 1, f) != 1) {
    fprintf(stderr, "[load] could not read header\n");
    fclose(f);
    return NULL;
  }

  if (memcmp(hdr.magic, "LGMMAPv1", 8) != 0) {
    fprintf(stderr, "[load] bad magic\n");
    fclose(f);
    return NULL;
  }

  /* -------------------------------------------------------------------------
   * 0. Verify CRC32 Integrity
   * -------------------------------------------------------------------------
   */
  fseeko(f, 0, SEEK_END);
  off_t file_size = ftello(f);
  if (file_size < (off_t)(sizeof(MmapHeader) + 4)) {
    fprintf(stderr, "[load] file too small for CRC\n");
    fclose(f);
    return NULL;
  }

  uint32_t expected_crc;
  fseeko(f, -4, SEEK_END);
  if (fread(&expected_crc, sizeof(expected_crc), 1, f) != 1) {
    fprintf(stderr, "[load] could not read CRC\n");
    fclose(f);
    return NULL;
  }

  fseeko(f, 0, SEEK_SET);
  uint32_t actual_crc = 0;
  uint8_t crc_buf[8192];
  uint64_t remaining = (uint64_t)(file_size - 4);
  while (remaining > 0) {
    size_t to_read =
        (remaining > sizeof(crc_buf)) ? sizeof(crc_buf) : (size_t)remaining;
    if (fread(crc_buf, 1, to_read, f) != to_read) {
      fprintf(stderr, "[load] error reading for CRC verification\n");
      fclose(f);
      return NULL;
    }
    actual_crc = crc32_update(actual_crc, crc_buf, to_read);
    remaining -= to_read;
  }

  if (actual_crc != expected_crc) {
    fprintf(stderr, "[load] CRC mismatch: expected %08x, got %08x\n",
            expected_crc, actual_crc);
    fclose(f);
    return NULL;
  }

  /* Reset to start of header data for subsequent loading */
  fseeko(f, sizeof(MmapHeader), SEEK_SET);

  Tokenizer *t = calloc(1, sizeof(Tokenizer));
  if (!t) {
    fclose(f);
    return NULL;
  }

  t->vocab_size = hdr.vocab_size;

  /* 1. SyllableTable */
  if (fseeko(f, (off_t)hdr.stbl_offset, SEEK_SET) != 0)
    goto io_err;
  t->stbl = calloc(1, sizeof(SyllableTable));
  if (!t->stbl)
    goto io_err;
  if (fread(t->stbl, sizeof(SyllableTable), 1, f) != 1)
    goto io_err;

  /* Enforce invariant: entries[i].id must equal i for correct trie traversal */
  for (uint16_t i = 0; i < t->stbl->count; i++) {
    t->stbl->entries[i].id = i;
  }

  /* 2. RePairState */
  t->rs = calloc(1, sizeof(RePairState));
  if (!t->rs)
    goto io_err;
  t->rs->rule_count = hdr.rule_count;
  if (hdr.rules_offset > 0 && hdr.rule_count > 0) {
    /* Guard against a crafted file with rule_count > MAX_RULES, which
     * would overflow the fixed-size rules[] array inside RePairState. */
    if (hdr.rule_count > MAX_RULES) {
      fprintf(stderr, "[load] rule_count %" PRIu32 " exceeds MAX_RULES %d\n",
              hdr.rule_count, MAX_RULES);
      goto io_err;
    }
    if (fseeko(f, (off_t)hdr.rules_offset, SEEK_SET) != 0)
      goto io_err;
    if (fread(t->rs->rules, sizeof(Rule), hdr.rule_count, f) != hdr.rule_count)
      goto io_err;
  }

  /* 3. LOUDS CSR */
  if (hdr.louds_offset > 0) {
    if (fseeko(f, (off_t)hdr.louds_offset, SEEK_SET) != 0)
      goto io_err;
    t->louds = calloc(1, sizeof(LOUDS));
    if (!t->louds)
      goto io_err;
    if (fread(t->louds, sizeof(LOUDS), 1, f) != 1)
      goto io_err;

    LOUDS *l = t->louds;
    /* Reject unreasonably large counts from a crafted file before
     * multiplying — prevents silent integer overflow in the malloc sizes. */
    if (l->node_count > (1u << 24) || l->edge_count > (1u << 24)) {
      fprintf(stderr, "[load] LOUDS node/edge count implausibly large\n");
      goto io_err;
    }
    l->row_ptr = malloc((l->node_count + 1) * sizeof(uint32_t));
    l->labels = malloc(l->edge_count * sizeof(uint16_t));
    l->next_node = malloc(l->edge_count * sizeof(uint32_t));
    l->terminals = malloc(l->edge_count * sizeof(uint32_t));
    if (!l->row_ptr || !l->labels || !l->next_node || !l->terminals)
      goto io_err;

    uint64_t off = ALIGN8(hdr.louds_offset + sizeof(LOUDS));
    if (fseeko(f, (off_t)off, SEEK_SET) != 0)
      goto io_err;
    if (fread(l->row_ptr, sizeof(uint32_t), l->node_count + 1, f) !=
        l->node_count + 1)
      goto io_err;

    off = ALIGN8((uint64_t)ftello(f));
    if (fseeko(f, (off_t)off, SEEK_SET) != 0)
      goto io_err;
    if (fread(l->labels, sizeof(uint16_t), l->edge_count, f) != l->edge_count)
      goto io_err;

    off = ALIGN8((uint64_t)ftello(f));
    if (fseeko(f, (off_t)off, SEEK_SET) != 0)
      goto io_err;
    if (fread(l->next_node, sizeof(uint32_t), l->edge_count, f) !=
        l->edge_count)
      goto io_err;

    off = ALIGN8((uint64_t)ftello(f));
    if (fseeko(f, (off_t)off, SEEK_SET) != 0)
      goto io_err;
    if (fread(l->terminals, sizeof(uint32_t), l->edge_count, f) !=
        l->edge_count)
      goto io_err;

    l->has_csr = true;
  }

  /* 4. Expansions */
  if (hdr.strings_offset > 0 && hdr.rule_count > 0) {
    if (fseeko(f, (off_t)hdr.strings_offset, SEEK_SET) != 0)
      goto io_err;
    t->gp_expansions = calloc(hdr.rule_count, sizeof(char *));
    for (uint32_t i = 0; i < hdr.rule_count; i++) {
      t->gp_expansions[i] = malloc(MAX_TOKEN_CHARS);
      if (fread(t->gp_expansions[i], MAX_TOKEN_CHARS, 1, f) != 1)
        goto io_err;
    }
  }

  fclose(f);

  /* 5. Rebuild transient objects */
  t->stbl->trie = syltrie_build(t->stbl);
  if (!t->stbl->trie) {
    fprintf(stderr, "[load] failed to rebuild syllable trie\n");
    tokenizer_destroy(t);
    return NULL;
  }

  t->syl = syllabifier_create(t->stbl);
  if (!t->syl) {
    fprintf(stderr, "[load] failed to create syllabifier\n");
    tokenizer_destroy(t);
    return NULL;
  }
  t->syl->frozen = true;
  stbl_freeze(t->stbl); /* Ensure table is frozen to match build behavior */

  /* 6. Re-seed and clean hardcoded morphemes to rebuild id_to_str */
  t->morph_strings = calloc(MORPHEME_SEED_COUNT, sizeof(char *));
  for (uint32_t mi = 0; mi < MORPHEME_SEED_COUNT; mi++) {
    const char *m = MORPHEME_SEEDS[mi];
    if (!m) break;
    
    char clean_buf[MAX_TOKEN_CHARS];
    clean_and_normalize_seed(clean_buf, m, sizeof(clean_buf));
    if (clean_buf[0] == '\0') continue;
    
    uint16_t syls[MAX_SEQ_LEN];
    if (syllabify_optimized(t, clean_buf, syls, MAX_SEQ_LEN) > 0) {
      t->morph_strings[mi] = strdup(clean_buf);
    }
  }

  tokenizer_rebuild_id_to_str(t);

  tokenizer_rebuild_fast_paths(t);
  tokenizer_build_dense_hashes(t);

  return t;

io_err:
  fprintf(stderr, "[load] I/O error reading model\n");
  tokenizer_destroy(t);
  fclose(f);
  return NULL;
}

#if USE_FUSED
/* =========================================================
 *  FUSED STREAMING TOKENIZER (EXPERIMENTAL)
 *
 *  Combines syllabification + trie traversal into one pass.
 *
 *  ⚠️ WARNING: Currently 45% SLOWER than two-pass due to branch
 *  misprediction. Disabled by default in production builds.
 *  Enable with: -DUSE_FUSED=1
 *
 *  For production use, prefer tokenizer_encode() (two-pass with SIMD).
 * ========================================================= */

/* Action table for branchless dispatch (Fix 1 - Action Table) */
typedef enum {
  BA_ASCII_LOWER = 0,
  BA_ASCII_UPPER = 1,
  BA_DIGIT = 2,
  BA_PUNCT = 3,
  BA_SPACE = 4,
  BA_UTF8_2B = 5,
  BA_UTF8_3B = 6,
  BA_UTF8_4B = 7,
  BA_UTF8_CONT = 8,
  BA_CONTROL = 9,
} ByteAction;

static const uint8_t byte_action[256] = {
    [0x00 ... 0x1F] = BA_CONTROL,   [0x20] = BA_SPACE,
    [0x21 ... 0x2F] = BA_PUNCT,     [0x30 ... 0x39] = BA_DIGIT,
    [0x3A ... 0x40] = BA_PUNCT,     [0x41 ... 0x5A] = BA_ASCII_UPPER,
    [0x5B ... 0x60] = BA_PUNCT,     [0x61 ... 0x7A] = BA_ASCII_LOWER,
    [0x7B ... 0x7E] = BA_PUNCT,     [0x7F] = BA_CONTROL,
    [0x80 ... 0xBF] = BA_UTF8_CONT, [0xC0 ... 0xDF] = BA_UTF8_2B,
    [0xE0 ... 0xEF] = BA_UTF8_3B,   [0xF0 ... 0xF7] = BA_UTF8_4B,
    [0xF8 ... 0xFF] = BA_CONTROL,
};

int tokenizer_encode_fused_v2(const Tokenizer *t, const char *text,
                              uint32_t *out, uint32_t out_cap) {
  if (!t || !text || !out || !t->louds || !t->stbl)
    return -1;

  const LOUDS *l = t->louds;
  const uint8_t *p = (const uint8_t *)text;
  const uint8_t *end = p + strlen(text);
  uint32_t out_cnt = 0;

  while (p < end && out_cnt < out_cap) {
    uint8_t act = byte_action[*p];

    switch (act) {
    case BA_PUNCT:
    case BA_DIGIT:
    case BA_SPACE: {
      uint32_t d = t->byte_direct[*p];
      if (d) {
        out[out_cnt++] = (d >> 16) - 1;
        p += d & 0xFFFF;
        continue;
      }
      break;
    }
    case BA_ASCII_LOWER:
    case BA_ASCII_UPPER: {
      if (end - p >= 2) {
        uint16_t key = p[0] | ((uint16_t)p[1] << 8);
        uint32_t entry = dense_cv_get(t, key);
        if (entry) {
          out[out_cnt++] = (entry >> 8) - 1;
          p += entry & 0xFF;
          continue;
        }
      }
      uint32_t d = t->byte_direct[*p];
      if (d) {
        out[out_cnt++] = (d >> 16) - 1;
        p++;
        continue;
      }
      break;
    }
    case BA_UTF8_3B: {
      /* Check for ▁ prefix (0xE2 0x96 0x81) */
      if (end - p >= 5 && p[1] == 0x96 && p[2] == 0x81) {
        uint16_t key = p[3] | ((uint16_t)p[4] << 8);
        uint32_t entry = dense_sp_get(t, key);
        if (!entry)
          entry = dense_sp_get(t, (uint16_t)p[3]);
        if (entry) {
          out[out_cnt++] = (entry >> 8) - 1;
          p += 3 + (entry & 0xFF);
          continue;
        }
      }
      break;
    }
    case BA_UTF8_4B: {
      /* Skip emoji/symbols atomically */
      if (end - p >= 4) {
        out[out_cnt++] = t->unk_token_id;
        p += 4;
        continue;
      }
      break;
    }
    case BA_UTF8_2B: {
      if (end - p >= 2) {
        uint16_t key = p[0] | ((uint16_t)p[1] << 8);
        uint32_t entry = dense_cv_get(t, key);
        if (entry) {
          out[out_cnt++] = (entry >> 8) - 1;
          p += 2;
          continue;
        }
      }
      break;
    }
    default:
      break;
    }

    /* Fallback: syllable + trie traversal */
    const uint8_t *next_p = p;
    uint16_t syl_id;
    int consumed = consume_syllable_fast(t, &next_p, &syl_id);
    if (consumed <= 0) {
      p++;
      continue;
    }

    if (t->fast_token[syl_id] && !t->can_extend[syl_id]) {
      out[out_cnt++] = t->fast_token[syl_id] - 1;
      p = next_p;
      continue;
    }

    /* Full trie traversal */
    uint32_t node = 0, best_id = 0, best_len = 0;
    uint32_t i = 0;
    while (i < (uint32_t)(end - p)) {
      uint16_t sid = (i == 0) ? syl_id : 0; /* Would need syllabify here */
      uint32_t child = louds_find_child(l, node, sid);
      if (child == 0)
        break;
      node = child;
      if (l->nodes[node].token_id != 0) {
        best_id = l->nodes[node].token_id;
        best_len = i + 1;
      }
      i++;
    }
    out[out_cnt++] = best_id ? best_id - 1 : t->unk_token_id;
    p = next_p;
  }
  return (int)out_cnt;
}

/* Original tokenizer_encode_fused - kept for comparison/benchmarking */
int tokenizer_encode_fused(const Tokenizer *t, const char *text, uint32_t *out,
                           uint32_t out_cap) {
  if (!t || !text || !out || !t->louds || !t->stbl)
    return -1;

  const LOUDS *l = t->louds;
  const uint8_t *p = (const uint8_t *)text;

  uint32_t out_cnt = 0;
  while (*p && out_cnt < out_cap) {
    /*
     * 🔥 SAFE CV_TO_TOKEN FAST-PATH: Now enabled with length encoding
     * The original implementation stored only token_id+1 without length,
     * causing hash collisions. Now we store (tid+1)<<8 | length to prevent
     * collisions and ensure correct token consumption.
     */

    /* 🔥 SAFE SPACE-PREFIX FAST-PATH (O(1) for ▁CV / ▁CVC) */
    /* Check for ▁ prefix (0xE2 0x96 0x81) */
    if (__builtin_expect(p[0] == 0xE2 && p[1] == 0x96 && p[2] == 0x81, 1)) {
      /* Ensure we don't read past null terminator */
      if (__builtin_expect(p[3] != '\0', 1)) {
        /* Fix 1: Use dense hash table for space-prefixed tokens */
        uint16_t key2 = (uint16_t)p[3] | ((uint16_t)p[4] << 8);
        uint32_t entry = dense_sp_get(t, key2);

        /* Fallback to 1-byte match (▁C) */
        if (__builtin_expect(entry == 0, 0)) {
          entry = dense_sp_get(t, (uint16_t)p[3]);
        }

        if (__builtin_expect(entry != 0, 1)) {
          uint32_t tid = (entry >> 8) - 1;
          uint8_t match_len = entry & 0xFF;

          /*
           * MAXIMAL MATCH GUARD (Optional but recommended):
           * If the matched token can be extended in the trie (i.e., it's a
           * prefix of a longer token like ▁bang), we should fall back to the
           * trie to ensure we find the longest match. For high-throughput
           * syllable tokenizers, this is often skipped as syllables are atomic,
           * but strict BPE requires this check.
           *
           * If skipping the check, the fast-path assumes ▁CV/CVC tokens
           * are leaves or sufficiently distinct.
           */

          INC_ATOMIC(t->stats.fast_path_hits);
          out[out_cnt++] = tid;
          p += 3 + match_len; /* Safely skip EXACT matched bytes */
          goto next_token;
        }
      }
    }

    uint32_t node = 0;
    uint32_t best_id = 0;
    const uint8_t *best_p = p;
    const uint8_t *token_start = p;

    /* Single greedy path */
    while (*p) {
      /* 🚀 Speculative Byte-Pattern Fast-Path */
      if (__builtin_expect(node == 0, 1)) {
        uint8_t c1 = p[0];
        if (__builtin_expect(c1 != '\0', 1)) {
          uint8_t c2 = p[1];
          if (is_letter_tbl[c2]) {
            if (__builtin_expect(!is_letter_tbl[p[2]], 1)) {
              uint16_t key = c1 | ((uint16_t)c2 << 8);
              uint32_t entry = dense_cv_get(t, key); /* Fix 1: dense hash */
              if (entry != 0) {
                uint32_t tid = (entry >> 8) - 1;
                uint8_t match_len = entry & 0xFF;
                INC_ATOMIC(t->stats.fast_path_hits);
                out[out_cnt++] = tid;
                p += match_len; /* Use exact length from entry */
                goto next_token;
              }
            }
          } else {
            /* Fix 1: Use byte_direct for O(1) single-byte lookup */
            uint32_t d = t->byte_direct[c1];
            if (d != 0) {
              INC_ATOMIC(t->stats.fast_path_hits);
              out[out_cnt++] = (d >> 16) - 1;
              p += d & 0xFFFF;
              goto next_token;
            }
          }
        }
      }

      uint16_t syl_id;
      const uint8_t *next_p = p;
      int consumed = consume_syllable_fast(t, &next_p, &syl_id);

      if (__builtin_expect(consumed <= 0, 0)) {
        if (consumed == 0)
          break;

        /* orphan byte: skip safely.
         * If this is the start of a multi-byte UTF-8 sequence, skip
         * the entire sequence as one atomic unit to prevent splitting
         * characters (emoji, combining marks, etc.) into garbage tokens. */
        if (node == 0) {
          size_t skip = utf8_seq_len(p); /* skip whole UTF-8 char */
          p += skip;                     /* skip byte(s) */
          token_start = p;               /* restart token cleanly */
        } else {
          size_t skip = utf8_seq_len(p);
          p += skip; /* continue but DO NOT mutate token_start */
        }
        break;
      }

      /* 🚀 Speculative Syllable Fast-Path */
      if (__builtin_expect(node == 0, 1)) {
        /* 1-syl fast path */
        uint32_t ft = l->fast_token[syl_id];
        if (ft != 0) {
          /* If followed by space/punct/EOL, no multi-syllable token can match
           */
          if (__builtin_expect(
                  !l->can_extend[syl_id] || !is_letter_tbl[*next_p], 1)) {
            INC_ATOMIC(t->stats.fast_path_hits);
            out[out_cnt++] = ft - 1;
            p = next_p;
            goto next_token;
          }
        }
      }
      INC_ATOMIC(t->stats.trie_traversals);
      p = next_p;

      /* Traverse LOUDS trie with linear scan for small edge counts */
      const TrieNode *nd = &l->nodes[node];
      if (__builtin_expect(nd->edge_count == 0, 0))
        break;

      const TrieEdge *edges = &l->edges[nd->first_edge];
      uint32_t next = 0;

      if (nd->edge_count < 8) {
        for (uint16_t i = 0; i < nd->edge_count; i++) {
          if (edges[i].label == syl_id) {
            next = edges[i].next_node;
            break;
          }
        }
      } else {
        uint32_t lo = 0, hi = nd->edge_count;
        while (lo < hi) {
          uint32_t mid = (lo + hi) >> 1;
          if (edges[mid].label < syl_id)
            lo = mid + 1;
          else
            hi = mid;
        }
        if (lo < nd->edge_count && edges[lo].label == syl_id) {
          next = edges[lo].next_node;
        }
      }

      if (next == 0)
        break;

      node = next;
      __builtin_prefetch(&l->nodes[node], 0, 1);
      if (l->nodes[node].token_id != 0) {
        best_id = l->nodes[node].token_id;
        best_p = p;
      }
    }

    if (best_id != 0) {
      out[out_cnt++] = best_id - 1;
      p = best_p;
    } else {
      /* Fallback: no token matched. Emit raw syllable and advance. */
      uint16_t syl_id;
      const uint8_t *fallback_p = token_start;
      int consumed = consume_syllable_fast(t, &fallback_p, &syl_id);
      if (consumed > 0) {
        if (syl_id != UINT16_MAX) {
          out[out_cnt++] = (uint32_t)syl_id;
        }
        p = fallback_p;
      } else if (*p) {
        /* Already handled orphan byte above if node==0, but just in case: */
        if (p == token_start) {
          p++;
        }
      }
    }
  next_token:;
  }

  return out_cnt;
}
#endif /* USE_FUSED */

/* =========================================================
 *  STREAMING TOKENIZER API
 * ========================================================= */

void tokenizer_cursor_init(TokenizerCursor *cursor, const Tokenizer *t,
                           const char *text) {
  if (!cursor || !t || !text)
    return;
  cursor->tokenizer = t;
  cursor->cursor = text;
  cursor->eos = false;
}

/* Streaming tokenization with resumable cursor.
 * Returns number of tokens written (may be < out_cap if buffer full).
 * Updates cursor->cursor so caller can resume on next call.
 * Sets cursor->eos when end of input reached. */
size_t tokenizer_encode_streaming(TokenizerCursor *cursor, uint32_t *out,
                                  size_t out_cap) {
  if (!cursor || !out || out_cap == 0 || cursor->eos)
    return 0;

  const Tokenizer *t = cursor->tokenizer;
  const LOUDS *l = t->louds;
  const uint8_t *p = (const uint8_t *)cursor->cursor;

  size_t out_cnt = 0;

  while (*p && out_cnt < out_cap) {
    uint32_t node = 0;
    uint32_t best_id = 0;
    const uint8_t *best_p = p;
    const uint8_t *start_p = p;

    /* Single greedy path through trie */
    while (*p) {
      uint16_t syl_id;
      int consumed = consume_syllable_id(t->stbl->trie, &p, &syl_id);
      if (__builtin_expect(consumed <= 0, 0)) {
        if (consumed == 0)
          break;
        p++;
        continue;
      }

      if (__builtin_expect(syl_id == UINT16_MAX, 0))
        break;

      /* Traverse Split CSR LOUDS trie */
      uint32_t start = l->row_ptr[node];
      uint32_t end = l->row_ptr[node + 1];
      uint32_t count = end - start;

      if (__builtin_expect(count == 0, 0))
        break;

      uint32_t next = 0xFFFFFFFF;

      /* Small linear scan for fanout < 8 */
      if (count < 8) {
        for (uint32_t k = start; k < end; k++) {
          if (l->labels[k] == syl_id) {
            next = l->next_node[k];
            break;
          }
        }
      } else {
        /* Binary search */
        uint32_t lo = start, hi = end;
        while (lo < hi) {
          uint32_t mid = (lo + hi) >> 1;
          if (l->labels[mid] < syl_id)
            lo = mid + 1;
          else
            hi = mid;
        }
        if (lo < end && l->labels[lo] == syl_id) {
          next = l->next_node[lo];
        }
      }

      if (next == 0xFFFFFFFF)
        break;

      node = next;
      if (l->terminals[node] != 0) {
        best_id = l->terminals[node];
        best_p = p;
      }
    }

    if (best_id != 0) {
      /* High-priority large syllable/morpheme match */
      out[out_cnt++] = best_id - 1;
      p = best_p;
    } else {
      /* Fallback: Emit raw ASCII/UTF-8 byte ID for residuals (e.g., 'o' = 111)
       */
      out[out_cnt++] = (uint32_t)*start_p;
      p = start_p + 1;
    }
  }

  /* Update cursor state for resumption */
  cursor->cursor = (const char *)p;
  if (!*p)
    cursor->eos = true;

  return out_cnt;
}
