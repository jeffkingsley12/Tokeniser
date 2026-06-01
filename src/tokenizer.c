/*
 * tokenizer.c
 *
 * Top-level pipeline:
 *
 * tokenizer_build():
 * 1. Create SyllableTable + Syllabifier
 * 2. Syllabify all documents → syllable-ID sequences
 * 3. Train Re-Pair on those sequences
 * 4. Run GrammarPruner (eff_freq → dead-mark → flatten)
 * 5. Extract live token strings from pruner_expand()
 * 6. Build LOUDS trie from token strings
 *
 * tokenizer_encode():
 * 1. syllabify input
 * 2. louds_tokenize() → token IDs
 *
 * tokenizer_decode():
 * Reverse lookup via id_to_token[] array
 *
 * tokenizer_save() / tokenizer_load():
 * Flat binary format (header + sections).
 * Safe for mmap on aligned platforms.
 */

#define _XOPEN_SOURCE 700
#define _DEFAULT_SOURCE

#include "tokenizer.h"
#include "syllable_seeds.h"
#include "truth_layer.h"
#include <fcntl.h>
#include <limits.h>
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
 * UTF-8 Utility: Sequence Length
 *
 * Returns the byte length of a UTF-8 sequence starting at `p`.
 * - 1 byte  = 0xxxxxxx (ASCII)
 * - 2 bytes = 110xxxxx 10xxxxxx
 * - 3 bytes = 1110xxxx 10xxxxxx 10xxxxxx
 * - 4 bytes = 11110xxx 10xxxxxx 10xxxxxx 10xxxxxx
 * - Invalid continuation byte (10xxxxxx) returns 1 (skip it).
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
/* stdatomic.h is included below with the rest of the system headers */

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
 * 🚀 HOT PATH OPTIMIZATIONS
 * =========================================================
 *
 * 1) SIMD ASCII pre-scan (AVX2 when available)
 * 2) Branchless vowel/consonant classification tables
 * 3) Cache-aligned lookup tables
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
    ['X'] = 1, ['Y'] = 1, ['Z'] = 1,
};

const uint32_t crc32_table[256] = {
    0x00000000, 0x77073096, 0xee0e612c, 0x990951ba, 0x076dc419, 0x706af48f,
    0xe963a535, 0x9e6495a3, 0x0edb8832, 0x79dcb8a4, 0xe0d5e91e, 0x97d2d988,
    0x09b64c2b, 0x7eb17cbd, 0xe7b82d07, 0x90bf1d91, 0x1db71064, 0x6ab020f2,
    0xf3b97148, 0x84be41de, 0x1adad47d, 0x6ddde4eb, 0xf4d4b551, 0x83d385c7,
    0x136c9856, 0x646ba8c0, 0xfd62f97a, 0x8a65c9ec, 0x14015c4f, 0x63066cd9,
    0xfa0f3d63, 0x8d080df5, 0x3b6e20c8, 0x4c69105e, 0xd56041e4, 0xa2677172,
    0x3c03e4d1, 0x4b04d447, 0xd20d85fd, 0xa50ab56b, 0x35b5a8fa, 0x42b2986c,
    0xdbbbc9d6, 0xacbcf940, 0x32d86ce3, 0x45df5c75, 0xdcd60dcf, 0xabd13d59,
    0x26d930ac, 0x51de003a, 0xc8d75180, 0xbfd06116, 0x21b4f4b5, 0x56b3c423,
    0xcfba9599, 0xb8bda50f, 0x2802b89e, 0x5f058808, 0xc60cd9b2, 0xb10be924,
    0x2f6f7c87, 0x58684c11, 0xc1611dab, 0xb6662d3d, 0x76dc4190, 0x01db7106,
    0x98d220bc, 0xefd5102a, 0x71b18589, 0x06b6b51f, 0x9fbfe4a5, 0xe8b8d433,
    0x7807c9a2, 0x0f00f934, 0x9609a88e, 0xe10e9818, 0x7f6a0dbb, 0x086d3d2d,
    0x91646c97, 0xe6635c01, 0x6b6b51f4, 0x1c6c6162, 0x856530d8, 0xf262004e,
    0x6c0695ed, 0x1b01a57b, 0x8208f4c1, 0xf50fc457, 0x65b0d9c6, 0x12b7e950,
    0x8bbeb8ea, 0xfcb9887c, 0x62dd1ddf, 0x15da2d49, 0x8cd37cf3, 0xfbd44c65,
    0x4db26158, 0x3ab551ce, 0xa3bc0074, 0xd4bb30e2, 0x4adfa541, 0x3dd895d7,
    0xa4d1c46d, 0xd3d6f4fb, 0x4369e96a, 0x346ed9fc, 0xad678846, 0xda60b8d0,
    0x44042d73, 0x33031de5, 0xaa0a4c5f, 0xdd0d7cc9, 0x5005713c, 0x270241aa,
    0xbe0b1010, 0xc90c2086, 0x5768b525, 0x206f85b3, 0xb966d409, 0xce61e49f,
    0x5edef90e, 0x29d9c998, 0xb0d09822, 0xc7d7a8b4, 0x59b33d17, 0x2eb40d81,
    0xb7bd5c3b, 0xc0ba6cad, 0xedb88320, 0x9abfb3b6, 0x03b6e20c, 0x74b1d29a,
    0xead54739, 0x9dd277af, 0x04db2615, 0x73dc1683, 0xe3630b12, 0x94643b84,
    0x0d6d6a3e, 0x7a6a5aa8, 0xe40ecf0b, 0x9309ff9d, 0x0a00ae27, 0x7d079eb1,
    0xf00f9344, 0x8708a3d2, 0x1e01f268, 0x6906c2fe, 0xf762575d, 0x806567cb,
    0x196c3671, 0x6e6b06e7, 0xfed41b76, 0x89d32be0, 0x10da7a5a, 0x67dd4acc,
    0xf9b9df6f, 0x8ebeeff9, 0x17b7be43, 0x60b08ed5, 0xd6d6a3e8, 0xa1d1937e,
    0x38d8c2c4, 0x4fdff252, 0xd1bb67f1, 0xa6bc5767, 0x3fb506dd, 0x48b2364b,
    0xd80d2bda, 0xaf0a1b4c, 0x36034af6, 0x41047a60, 0xdf60efc3, 0xa867df55,
    0x316e8eef, 0x4669be79, 0xcb61b38c, 0xbc66831a, 0x256fd2a0, 0x5268e236,
    0xcc0c7795, 0xbb0b4703, 0x220216b9, 0x5505262f, 0xc5ba3bbe, 0xb2bd0b28,
    0x2bb45a92, 0x5cb36a04, 0xc2d7ffa7, 0xb5d0cf31, 0x2cd99e8b, 0x5bdeae1d,
    0x9b64c2b0, 0xec63f226, 0x756aa39c, 0x026d930a, 0x9c0906a9, 0xeb0e363f,
    0x72076785, 0x05005713, 0x95bf4a82, 0xe2b87a14, 0x7bb12bae, 0x0cb61b38,
    0x92d28e9b, 0xe5d5be0d, 0x7cdcefb7, 0x0bdbdf21, 0x86d3d2d4, 0xf1d4e242,
    0x68ddb3f8, 0x1fda836e, 0x81be16cd, 0xf6b9265b, 0x6fb077e1, 0x18b74777,
    0x88085ae6, 0xff0f6a70, 0x66063bca, 0x11010b5c, 0x8f659eff, 0xf862ae69,
    0x616bffd3, 0x166ccf45, 0xa00ae278, 0xd70dd2ee, 0x4e048354, 0x3903b3c2,
    0xa7672661, 0xd06016f7, 0x4969474d, 0x3e6e77db, 0xaed16a4a, 0xd9d65adc,
    0x40df0b66, 0x37d83bf0, 0xa9bcae53, 0xdebb9ec5, 0x47b2cf7f, 0x30b5ffe9,
    0xbdbdf21c, 0xcabac28a, 0x53b39330, 0x24b4a3a6, 0xbad03605, 0xcdd70693,
    0x54de5729, 0x23d967bf, 0xb3667a2e, 0xc4614ab8, 0x5d681b02, 0x2a6f2b94,
    0xb40bbe37, 0xc30c8ea1, 0x5a05df1b, 0x2d02ef8d
};

uint32_t crc32_update(uint32_t crc, const uint8_t *data, size_t len) {
    crc = ~crc;
    for (size_t i = 0; i < len; i++) {
        crc = (crc >> 8) ^ crc32_table[(crc ^ data[i]) & 0xFF];
    }
    return ~crc;
}

#define IS_VOWEL(c) vowel_table[(uint8_t)(c)]
#define IS_CONSONANT(c) consonant_table[(uint8_t)(c)]

/* ============================================================================
 * DENSE HASH TABLES (per-instance fast-path tables)
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
  uint16_t sid = t->utf8_1b_to_syl[p[0]];
  if (__builtin_expect(sid != 0, 1)) {
    *syl_id_out = (uint16_t)(sid - 1);
    *pp = p + 1;
    return 1;
  }

  /* 2-byte UTF-8 path: Luganda characters (0xC0-0xDF prefix) */
  if (p[1] != 0) {
    int key2 = pack_2byte_utf8(p[0], p[1]);
    if (key2 >= 0) {
      sid = t->utf8_2b_to_syl[key2];
      if (__builtin_expect(sid != 0, 0)) {
        *syl_id_out = (uint16_t)(sid - 1);
        *pp = p + 2;
        return 2;
      }
    }
  }

  return consume_syllable_id(t->stbl->trie, pp, syl_id_out);
}

/* =========================================================
 * Internal: token vocabulary (built from grammar)
 * ========================================================= */

typedef struct {
  char text[MAX_TOKEN_CHARS];
  uint32_t id;
  uint32_t sym; /* grammar symbol this came from */
} TokenEntry;

static void tokenizer_build_dense_hashes(Tokenizer *t);

void tokenizer_rebuild_fast_paths(Tokenizer *t) {
  if (!t || !t->louds || !t->stbl || !t->stbl->trie)
    return;
  const SylTrie *trie = t->stbl->trie;

  t->fast_paths = safe_calloc(1, sizeof(TokenizerFastPaths));
  if (!t->fast_paths) return;

  memset(t->utf8_1b_to_syl, 0, sizeof(t->utf8_1b_to_syl));
  memset(t->utf8_2b_to_syl, 0, sizeof(t->utf8_2b_to_syl));
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
      t->utf8_1b_to_syl[(uint8_t)text[0]] = (uint16_t)(si + 1);
    } else {
      int key = pack_2byte_utf8((uint8_t)text[0], (uint8_t)text[1]);
      if (key >= 0) {
        t->utf8_2b_to_syl[key] = (uint16_t)(si + 1);
      }
    }
    populated++;
  }

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
            t->fast_paths->cv_to_token[key] = ((tid + 1) << 8) | 2;
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
            t->fast_paths->space_cv_to_token[key] = ((tid + 1) << 8) | (uint8_t)rem_len;
            space_populated++;
          }
        }
      }
    }
  }

  fprintf(stderr,
          "[tokenizer_rebuild_fast_paths] utf8_1b_to_syl: %u entries, "
          "cv_to_token: %u entries, space_cv_to_token: %u entries\n",
          populated, cv_populated, space_populated);

  tokenizer_build_dense_hashes(t);
  
  free(t->fast_paths);
  t->fast_paths = NULL;
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
  /* Shift right by (32 - log2(DENSE_HASH_SIZE)) to map hash into [0, DENSE_HASH_SIZE - 1] */
  #if DENSE_HASH_SIZE == 1024
    return ((uint32_t)k * HASH_MULTIPLIER) >> 22;
  #elif DENSE_HASH_SIZE == 2048
    return ((uint32_t)k * HASH_MULTIPLIER) >> 21;
  #elif DENSE_HASH_SIZE == 4096
    return ((uint32_t)k * HASH_MULTIPLIER) >> 20;
  #else
    return (((uint32_t)k * HASH_MULTIPLIER) >> (32 - __builtin_ctz(DENSE_HASH_SIZE)));
  #endif
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
    if (t->fast_paths->cv_to_token[i]) {
      uint32_t h = dense_hash_idx((uint16_t)i) & (DENSE_HASH_SIZE - 1);
      bool inserted = false;
      /* Linear probing with wrap */
      for (int probe = 0; probe < 16 && t->cv_dense_vals[h]; probe++) {
        if (t->cv_dense_keys[h] == (uint16_t)i) { inserted = true; break; }
        h = (h + 1) & (DENSE_HASH_SIZE - 1);
      }
      if (!t->cv_dense_vals[h]) {
        t->cv_dense_keys[h] = (uint16_t)i;
        t->cv_dense_vals[h] = t->fast_paths->cv_to_token[i];
        cv_count++;
        inserted = true;
      }
      if (!inserted) {
         assert(0 && "cv_dense hash overflow — increase DENSE_HASH_SIZE or probe limit");
      }
    }
  }

  /* Populate sp_dense from sparse space_cv_to_token */
  uint32_t sp_count = 0;
  for (uint32_t i = 0; i < 65536; i++) {
    if (t->fast_paths->space_cv_to_token[i]) {
      uint32_t h = dense_hash_idx((uint16_t)i) & (DENSE_HASH_SIZE - 1);
      bool inserted = false;
      for (int probe = 0; probe < 16 && t->sp_dense_vals[h]; probe++) {
        if (t->sp_dense_keys[h] == (uint16_t)i) { inserted = true; break; }
        h = (h + 1) & (DENSE_HASH_SIZE - 1);
      }
      if (!t->sp_dense_vals[h]) {
        t->sp_dense_keys[h] = (uint16_t)i;
        t->sp_dense_vals[h] = t->fast_paths->space_cv_to_token[i];
        sp_count++;
        inserted = true;
      }
      if (!inserted) {
         assert(0 && "sp_dense hash overflow — increase DENSE_HASH_SIZE or probe limit");
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
  for (int i = 0; i < 16; i++) {
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
  for (int i = 0; i < 16; i++) {
    if (t->sp_dense_vals[h] == 0)
      return 0; /* Empty slot = miss */
    if (t->sp_dense_keys[h] == key)
      return t->sp_dense_vals[h];
    h = (h + 1) & (DENSE_HASH_SIZE - 1);
  }
  return 0; /* Not found after max probes */
}

/* =========================================================
 * Morphological Constraint Mask Synthesis
 * ========================================================= */

#include "morph_constraints.h"

/* -------------------------------------------------------------------------
 * Base syllable → mask (fast path for the 50–100 most common morphemes).
 * In production this should be a perfect hash; shown here as a switch
 * for clarity.
 * ------------------------------------------------------------------------- */
static MorphMask syllable_to_mask(const Tokenizer *t, uint16_t syl_id)
{
    if (syl_id >= t->stbl->count) return 0;
    const char *txt = t->stbl->entries[syl_id].text;

    /* Noun class prefixes */
    if (strcmp(txt, "mu") == 0 || strcmp(txt, "omu") == 0)
        return FEAT_NOUN_CLASS_1 | FEAT_NOUN_CLASS_3 | FEAT_NOUN_CLASS_18 | ROLE_PREFIX;
    if (strcmp(txt, "ba") == 0 || strcmp(txt, "aba") == 0)
        return FEAT_NOUN_CLASS_2 | ROLE_PREFIX;
    if (strcmp(txt, "ki") == 0 || strcmp(txt, "eki") == 0)
        return FEAT_NOUN_CLASS_7 | ROLE_PREFIX;
    if (strcmp(txt, "bi") == 0 || strcmp(txt, "ebi") == 0)
        return FEAT_NOUN_CLASS_8 | ROLE_PREFIX;
    if (strcmp(txt, "n")  == 0 || strcmp(txt, "en")  == 0)
        return FEAT_NOUN_CLASS_9 | ROLE_PREFIX;
    if (strcmp(txt, "lu") == 0 || strcmp(txt, "olu") == 0)
        return FEAT_NOUN_CLASS_11 | ROLE_PREFIX;
    if (strcmp(txt, "ka") == 0 || strcmp(txt, "aka") == 0)
        return FEAT_DIMINUTIVE | ROLE_PREFIX;
    if (strcmp(txt, "bu") == 0 || strcmp(txt, "obu") == 0)
        return FEAT_ABSTRACT | ROLE_PREFIX;
    if (strcmp(txt, "ku") == 0 || strcmp(txt, "oku") == 0)
        return FEAT_INFINITIVE | ROLE_PREFIX;
    if (strcmp(txt, "wa") == 0 || strcmp(txt, "awa") == 0)
        return FEAT_LOCATIVE | ROLE_PREFIX;
    if (strcmp(txt, "ma") == 0 || strcmp(txt, "ama") == 0)
        return FEAT_NOUN_CLASS_6 | ROLE_PREFIX;
    if (strcmp(txt, "li") == 0 || strcmp(txt, "eri") == 0)
        return FEAT_NOUN_CLASS_5 | ROLE_PREFIX;
    if (strcmp(txt, "mi") == 0 || strcmp(txt, "emi") == 0)
        return FEAT_NOUN_CLASS_4 | ROLE_PREFIX;
    if (strcmp(txt, "zi") == 0 || strcmp(txt, "enzi") == 0)
        return FEAT_NOUN_CLASS_10 | ROLE_PREFIX;
    if (strcmp(txt, "tu") == 0 || strcmp(txt, "otu") == 0)
        return FEAT_NOUN_CLASS_13 | ROLE_PREFIX;
    if (strcmp(txt, "ga") == 0 || strcmp(txt, "aga") == 0)
        return FEAT_NOUN_CLASS_22 | ROLE_PREFIX;
    if (strcmp(txt, "gu") == 0 || strcmp(txt, "ogu") == 0)
        return FEAT_AUGMENTATIVE | ROLE_PREFIX;

    /* Roots */
    if (strcmp(txt, "lim")  == 0 || strcmp(txt, "som") == 0 ||
        strcmp(txt, "lya")  == 0 || strcmp(txt, "nya") == 0 ||
        strcmp(txt, "gul")  == 0 || strcmp(txt, "tun") == 0 ||
        strcmp(txt, "lab")  == 0 || strcmp(txt, "kub") == 0 ||
        strcmp(txt, "zib")  == 0 || strcmp(txt, "vub") == 0 ||
        strcmp(txt, "tamb") == 0 || strcmp(txt, "sinz") == 0 ||
        strcmp(txt, "kola") == 0 || strcmp(txt, "yimb") == 0 ||
        strcmp(txt, "zany") == 0)
        return FEAT_VERB_ROOT | ROLE_ROOT;

    /* Suffixes / particles */
    if (strcmp(txt, "vu") == 0 || strcmp(txt, "fu") == 0)
        return ROLE_SUFFIX;
    if (strcmp(txt, "ne") == 0 || strcmp(txt, "na") == 0 ||
        strcmp(txt, "e")  == 0)
        return ROLE_PARTICLE;

    return 0;   /* unknown syllable: no constraints, no features */
}

/* -------------------------------------------------------------------------
 * Resolve a grammar symbol (syllable or previously built rule) to its mask.
 * ------------------------------------------------------------------------- */
static MorphMask symbol_mask(const Tokenizer *t, uint32_t sym)
{
    if (sym < BASE_SYMBOL_OFFSET) {                     /* base syllable */
        return syllable_to_mask(t, (uint16_t)sym);
    } else {                                            /* Re-Pair rule  */
        uint32_t idx = sym - BASE_SYMBOL_OFFSET;
        uint32_t tid = t->stbl->count + idx;             /* token_id space */
        return (idx < t->rs->rule_count) ? t->token_features[tid] : 0;
    }
}

static MorphMask symbol_requires(const Tokenizer *t, uint32_t sym)
{
    if (sym < BASE_SYMBOL_OFFSET) return 0;
    uint32_t idx = sym - BASE_SYMBOL_OFFSET;
    uint32_t tid = t->stbl->count + idx;
    return (idx < t->rs->rule_count) ? t->token_requires[tid] : 0;
}

static MorphMask symbol_forbids(const Tokenizer *t, uint32_t sym)
{
    if (sym < BASE_SYMBOL_OFFSET) return 0;
    uint32_t idx = sym - BASE_SYMBOL_OFFSET;
    uint32_t tid = t->stbl->count + idx;
    return (idx < t->rs->rule_count) ? t->token_forbids[tid] : 0;
}

/* -------------------------------------------------------------------------
 * Bottom-up bitwise logic synthesis for every Re-Pair rule.
 *
 *   X → L R
 *   features(X) = features(L) | features(R)
 *   forbids(X)  = forbids(L)  | forbids(R)
 *   requires(X) = requires(L) | (requires(R) & ~features(L))
 *
 * The last line is the critical one: whatever R needs that L does NOT
 * provide becomes a requirement of the combined symbol X.
 * ------------------------------------------------------------------------- */
static void build_grammar_masks(Tokenizer *t)
{
    if (!t->rs) return;
    uint32_t base = t->stbl->count;   /* token_id of rule 0 */

    for (uint32_t ri = 0; ri < t->rs->rule_count; ri++) {
        Rule *r   = &t->rs->rules[ri];
        uint32_t L = r->rhs[0];
        uint32_t R = r->rhs[1];

        MorphMask Lf = symbol_mask(t, L);
        MorphMask Lr = symbol_requires(t, L);
        MorphMask Lb = symbol_forbids(t, L);

        MorphMask Rf = symbol_mask(t, R);
        MorphMask Rr = symbol_requires(t, R);
        MorphMask Rb = symbol_forbids(t, R);

        uint32_t tid = base + ri;

        t->token_features[tid] = Lf | Rf;
        t->token_forbids[tid]  = Lb | Rb;
        t->token_requires[tid] = Lr | (Rr & ~Lf);

        /* Optional: derive log_prob from Re-Pair effective frequency */
        t->token_features[tid] |= 0; /* placeholder for freq scoring */
    }
}

/* -------------------------------------------------------------------------
 * Multi-syllable tokens (morpheme seeds, live grammar expansions) may
 * contain several truth-layer matches. We run the AC automaton over the
 * syllable sequence and OR the results together.
 * ------------------------------------------------------------------------- */
static MorphMask truth_mask_for_text(Tokenizer *t, TruthTrie *tt,
                                     const char *text)
{
    uint16_t syls[MAX_SEQ_LEN];
    int n = syllabify(t->syl, text, syls, MAX_SEQ_LEN);
    if (n <= 0) return 0;

    TruthMatchInfo info = {0};
    truth_match_ids(tt, syls, (size_t)n, &info);

    MorphMask m = 0;
    if (info.category_mask & TRUTH_PREFIX)  m |= ROLE_PREFIX;
    if (info.category_mask & TRUTH_SUFFIX) m |= ROLE_SUFFIX;
    if (info.category_mask & TRUTH_ROOT)    m |= ROLE_ROOT;
    if (info.category_mask & TRUTH_PARTICLE) m |= ROLE_PARTICLE;

    /* Also fold in any class-specific bits from individual syllables */
    for (int i = 0; i < n; i++) m |= syllable_to_mask(t, syls[i]);
    return m;
}

/* -------------------------------------------------------------------------
 * Master mask builder — call once during tokenizer_build(), after the
 * vocabulary size is known but before louds_build() or immediately after.
 * ------------------------------------------------------------------------- */
static void tokenizer_build_masks(Tokenizer *t, TruthTrie *tt)
{
    if (!t || !t->stbl) return;

    t->token_features = calloc(t->vocab_size, sizeof(MorphMask));
    t->token_requires = calloc(t->vocab_size, sizeof(MorphMask));
    t->token_forbids  = calloc(t->vocab_size, sizeof(MorphMask));
    if (!t->token_features || !t->token_requires || !t->token_forbids) {
        fprintf(stderr, "[masks] OOM allocating constraint tables\n");
        return;
    }

    /* 1. Syllables (base layer) */
    for (uint16_t si = 0; si < t->stbl->count; si++) {
        t->token_features[si] = syllable_to_mask(t, si);
    }

    /* 2. Re-Pair rules (bottom-up synthesis) */
    build_grammar_masks(t);

    /* 3. Morpheme seeds (truth-trie + syllable scan) */
    uint32_t morph_base = t->stbl->count + (t->rs ? t->rs->rule_count : 0);
    for (uint32_t mi = 0; mi < MORPHEME_SEED_COUNT; mi++) {
        if (!t->morph_strings[mi]) continue;
        uint32_t tid = morph_base + mi;
        if (tid >= t->vocab_size) continue;
        t->token_features[tid] = truth_mask_for_text(t, tt, t->morph_strings[mi]);
    }

    /* 4. Grammar expansions (pruner output) — optional */
    if (t->gp_expansions && t->rs) {
        for (uint32_t ri = 0; ri < t->rs->rule_count; ri++) {
            if (t->rs->rules[ri].dead) continue;
            uint32_t tid = t->stbl->count + ri;
            if (!t->token_features[tid] && t->gp_expansions[ri]) {
                t->token_features[tid] = truth_mask_for_text(t, tt, t->gp_expansions[ri]);
            }
        }
    }
}

/* =========================================================
 * tokenizer_build
 * ========================================================= */

/* --- O(1) Hash Set Deduplication Helpers --- */
static inline uint32_t str_hash(const char *s) {
    uint32_t h = 2166136261u;
    while (*s) h = (h ^ (uint8_t)*s++) * 16777619u;
    return h;
}

static inline bool dedup_insert(const char **set, uint32_t cap, const char *s) {
    uint32_t i = str_hash(s) & (cap - 1);
    while (set[i]) {
        if (strcmp(set[i], s) == 0) return false; /* duplicate */
        i = (i + 1) & (cap - 1);
    }
    set[i] = s;
    return true;
}

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
      if (buf[j] >= t->stbl->count) continue;
      const char *decoded = t->stbl->entries[buf[j]].text;
      if (decoded && decoded[0] != '\0') {
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
          if (!unique_sylls[unique_count].text) {
            syllabifier_destroy(temp_syl);
            for (size_t k = 0; k < unique_count; k++)
              free(unique_sylls[k].text);
            free(unique_sylls);
            goto fail;
          }
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
  SyllableTable *deterministic_stbl = calloc(1, sizeof(SyllableTable));
  if (!deterministic_stbl) {
    for (size_t i = 0; i < unique_count; i++) {
      free(unique_sylls[i].text);
    }
    free(unique_sylls);
    goto fail;
  }

  if (stbl_seed_special(deterministic_stbl) != 0) {
    fprintf(stderr, "[tokenizer_build] Failed to seed special tokens in deterministic table\n");
    free(deterministic_stbl);
    for (size_t i = 0; i < unique_count; i++) free(unique_sylls[i].text);
    free(unique_sylls);
    goto fail;
  }

  for (int i = 0; i < 256; i++) {
    char s[2] = {(char)i, 0};
    uint16_t id = SPECIAL_TOKENS_COUNT + i;
    if (stbl_intern_fixed(deterministic_stbl, s, id) == UINT16_MAX) {
      fprintf(stderr, "[tokenizer_build] Failed to seed byte %d at ID %u in deterministic table\n", i, id);
      free(deterministic_stbl);
      for (size_t j = 0; j < unique_count; j++) free(unique_sylls[j].text);
      free(unique_sylls);
      goto fail;
    }
  }

  uint32_t phono_seeded_det = stbl_seed_phonotactic(deterministic_stbl);
  if (phono_seeded_det == 0) {
    fprintf(stderr, "[tokenizer_build] Warning: No phonotactic seeds loaded in deterministic table\n");
  }

  for (size_t i = 0; i < unique_count; i++) {
    uint16_t id = stbl_intern(deterministic_stbl, unique_sylls[i].text);
    if (id == UINT16_MAX) {
      fprintf(stderr, "[tokenizer_build] Failed to intern syllable %zu: %s\n", i, unique_sylls[i].text);
      free(deterministic_stbl);
      for (size_t j = 0; j < unique_count; j++) free(unique_sylls[j].text);
      free(unique_sylls);
      goto fail;
    }
  }

  syllabifier_destroy(t->syl);
  t->syl = NULL;
  free(t->stbl);
  t->stbl = deterministic_stbl;

  for (uint16_t i = 0; i < t->stbl->count; i++) {
    t->stbl->entries[i].id = i;
  }

  t->syl = syllabifier_create(t->stbl);
  if (!t->syl) {
    fprintf(stderr, "[tokenizer_build] Failed to create syllabifier with deterministic table\n");
    for (size_t i = 0; i < unique_count; i++) free(unique_sylls[i].text);
    free(unique_sylls);
    goto fail;
  }

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
      for (uint32_t j = 0; j < i; j++) free(seqs[j]);
      free(seqs);
      free(lens);
      goto fail;
    }
    int n = syllabify_optimized(t, docs[i], buf, MAX_SYLLABLES);
    if (n < 0) n = 0;
    seqs[i] = buf;
    lens[i] = (uint32_t)n;
  }

  /* --- 3. Truth Layer Annotation --- */
  TruthTrie *tt = truth_trie_create();
  MergeMask **masks = safe_calloc(n_docs, sizeof(MergeMask *));
  if (tt && masks) {
    truth_trie_build_from_seed(tt, t);
    for (uint32_t i = 0; i < n_docs; i++) {
      if (lens[i] < 2) continue;

      masks[i] = merge_mask_create(lens[i] - 1);
      if (!masks[i]) continue;

      for (size_t j = 0; j < lens[i] - 1; j++)
        set_merge_allowed(masks[i], j, true);

      for (uint32_t j = 0; j < lens[i]; j++) {
        TruthMatchInfo match;
        int match_len = truth_match_ids(tt, seqs[i] + j, lens[i] - j, &match);
        if (match_len > 1) {
          if (j > 0) set_merge_allowed(masks[i], j - 1, false);
          if (j + match_len - 1 < lens[i] - 1) {
            set_merge_allowed(masks[i], j + match_len - 1, false);
          }
          j += (uint32_t)(match_len - 1);
        }
      }
    }
  }

  /* --- 4. Re-Pair training --- */
  t->rs = repair_create();
  if (!t->rs) {
    if (masks) {
      for (uint32_t i = 0; i < n_docs; i++) merge_mask_destroy(masks[i]);
      free(masks);
    }
    truth_trie_destroy(tt);
    for (uint32_t i = 0; i < n_docs; i++) free(seqs[i]);
    free(seqs);
    free(lens);
    goto fail;
  }

  if (repair_train_with_mask(t->rs, (const uint16_t **)seqs, lens, n_docs,
                             (const MergeMask **)masks) != 0) {
    if (masks) {
      for (uint32_t i = 0; i < n_docs; i++) merge_mask_destroy(masks[i]);
      free(masks);
    }
    truth_trie_destroy(tt);
    for (uint32_t i = 0; i < n_docs; i++) free(seqs[i]);
    free(seqs);
    free(lens);
    goto fail;
  }

  if (masks) {
    for (uint32_t i = 0; i < n_docs; i++) merge_mask_destroy(masks[i]);
    free(masks);
  }
  truth_trie_destroy(tt);

  t->syl->frozen = true;
  stbl_freeze(t->stbl); 

  for (uint32_t i = 0; i < n_docs; i++) free(seqs[i]);
  free(seqs);
  free(lens);

  /* --- 4. Grammar pruning --- */
  t->gp = pruner_create(t->rs, t->stbl);
  if (!t->gp) goto fail;

  if (!pruner_compute_eff_freq(t->gp)) {
    fprintf(stderr, "[tokenizer_build] pruner_compute_eff_freq OOM — aborting\n");
    goto fail;
  }

  pruner_mark_dead(t->gp);
  pruner_expand_all(t->gp, t->stbl);
  if (t->gp->fatal_oom) {
    fprintf(stderr, "[tokenizer_build] OOM expanding grammar rules — aborting\n");
    goto fail;
  }

  /* --- 5. Finalize Vocabulary IDs --- */
  uint32_t syl_id_base = 0;
  uint32_t gram_id_base = syl_id_base + t->stbl->count;
  uint32_t morph_id_base = gram_id_base + t->rs->rule_count;

  if (t->rs->rule_count > UINT32_MAX - 16 - t->stbl->count - MORPHEME_SEED_COUNT) {
    fprintf(stderr, "[tokenizer_build] token capacity calculation would overflow\n");
    goto fail;
  }
  uint32_t token_cap = t->rs->rule_count + t->stbl->count + MORPHEME_SEED_COUNT + 16;
  char **tok_strings = calloc(token_cap, sizeof *tok_strings);
  uint32_t *tok_ids = calloc(token_cap, sizeof *tok_ids);
  if (!tok_strings || !tok_ids) {
    free(tok_strings);
    free(tok_ids);
    goto fail;
  }

  uint32_t n_tokens = 0;

  /* --- O(1) Hash Set Deduplication --- */
  /* Dynamic sizing: 33% load factor prevents probe collisions */
  uint32_t dedup_cap = 65536;
  while (dedup_cap <= token_cap * 3) {
    dedup_cap <<= 1;
  }
  const char **dedup_set = calloc(dedup_cap, sizeof(char *));
  if (!dedup_set) {
    free(tok_strings);
    free(tok_ids);
    goto fail;
  }

  /* Pass 1: Syllables */
  for (uint16_t si = 0; si < t->stbl->count; si++) {
    const char *text = t->stbl->entries[si].text;
    uint16_t tmp_syls[MAX_SEQ_LEN];
    int n = syllabify_optimized(t, text, tmp_syls, MAX_SEQ_LEN);
    bool has_unk = false;
    for (int j = 0; j < n; j++) if (tmp_syls[j] == TOK_UNK) { has_unk = true; break; }
    
    if (n > 0 && !has_unk) {
      if (dedup_insert(dedup_set, dedup_cap, text)) {
        tok_strings[n_tokens] = (char *)text;
        tok_ids[n_tokens] = syl_id_base + si;
        n_tokens++;
      }
    }
  }

  /* Pass 2: Grammar rules */
  for (uint32_t ri = 0; ri < t->rs->rule_count; ri++) {
    Rule *r = &t->rs->rules[ri];
    if (r->dead) continue;
    const char *exp = pruner_expand(t->gp, r->lhs, t->stbl);
    if (!exp || exp[0] == '\0') continue;

    uint16_t tmp_syls[MAX_SEQ_LEN];
    int n = syllabify_optimized(t, exp, tmp_syls, MAX_SEQ_LEN);
    bool has_unk = false;
    for (int j = 0; j < n; j++) if (tmp_syls[j] == TOK_UNK) { has_unk = true; break; }

    if (n > 0 && !has_unk) {
      if (dedup_insert(dedup_set, dedup_cap, exp)) {
        tok_strings[n_tokens] = (char *)exp;
        tok_ids[n_tokens] = gram_id_base + ri;
        n_tokens++;
      }
    }
  }

  /* Pass 3: Morpheme seeds */
  t->morph_strings = calloc(MORPHEME_SEED_COUNT, sizeof(char *));
  for (uint32_t mi = 0; mi < MORPHEME_SEED_COUNT; mi++) {
    const char *m = MORPHEME_SEEDS[mi];
    if (!m) break;

    char clean_buf[MAX_TOKEN_CHARS];
    clean_and_normalize_seed(clean_buf, m, sizeof(clean_buf));
    if (clean_buf[0] == '\0') continue;

    uint16_t syls[MAX_SEQ_LEN];
    int n = syllabify_optimized(t, clean_buf, syls, MAX_SEQ_LEN);
    if (n > 0) {
      bool has_unk = false;
      for (int j = 0; j < n; j++) {
        if (syls[j] == TOK_UNK) {
          has_unk = true;
          break;
        }
      }
      if (!has_unk) {
        t->morph_strings[mi] = strdup(clean_buf);
        if (dedup_insert(dedup_set, dedup_cap, t->morph_strings[mi])) {
          tok_strings[n_tokens] = t->morph_strings[mi];
          tok_ids[n_tokens] = morph_id_base + mi;
          n_tokens++;
        }
      }
    }
  }

  t->stbl->trie = syltrie_build(t->stbl);
  if (!t->stbl->trie) {
    fprintf(stderr, "[tokenizer_build] failed to build syllable trie\n");
    free(dedup_set);
    goto fail;
  }

  /* --- 6. Finalize Vocab and Build LOUDS Trie --- */
  /* CRITICAL FIX: vocab_size must cover the full sparse ID space,
   * not just the dense n_tokens count after deduplication.
   * The maximum ID allocated is bounded by morph_id_base + MORPHEME_SEED_COUNT. */
  t->vocab_size = morph_id_base + MORPHEME_SEED_COUNT;

  /* --- NEW: compile morphological constraints --- */
  TruthTrie *mask_tt = truth_trie_create();
  if (mask_tt) {
    truth_trie_build_from_seed(mask_tt, t);
    tokenizer_build_masks(t, mask_tt);   /* builds token_features/requires/forbids */
    truth_trie_destroy(mask_tt);
  }

  t->louds = louds_build((const char **)tok_strings, tok_ids, n_tokens, t->stbl,
                         t->syl);
  free(tok_strings);
  free(tok_ids);
  free((void *)dedup_set);

  if (!t->louds) goto fail;

  tokenizer_rebuild_id_to_str(t);
  tokenizer_rebuild_fast_paths(t);

  return t;

fail:
  tokenizer_destroy(t);
  return NULL;
}

/* =========================================================
 * tokenizer_destroy
 * ========================================================= */

void tokenizer_destroy(Tokenizer *t) {
  if (!t) return;

  if (t->morph_strings) {
    for (uint32_t i = 0; i < MORPHEME_SEED_COUNT; i++)
      free(t->morph_strings[i]);
    free(t->morph_strings);
  }

  free(t->id_to_str);

  free(t->token_features);
  free(t->token_requires);
  free(t->token_forbids);

  if (t->gp_expansions) {
      if (!t->mmap_addr && t->rs) {
          for (uint32_t i = 0; i < t->rs->rule_count; i++)
              free(t->gp_expansions[i]);
      }
      free(t->gp_expansions);
  }

  if (!t->mmap_addr) {
      louds_destroy(t->louds);
  }

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
 * tokenizer_encode
 * ========================================================= */

int tokenizer_encode(const Tokenizer *t, const char *text, uint32_t *out,
                     uint32_t out_cap) {
  if (!t || !text || !out) return -1;

#if USE_LATTICE_ENCODING
  /* Use lattice-based beam search when enabled */
  if (t->token_features && t->louds && t->louds->has_csr) {
    return tokenizer_encode_lattice(t, text, out, out_cap);
  }
#endif

  /* Fast path: if no constraint tables built, fall back to greedy LOUDS */
  if (!t->token_features) {
    uint16_t syls[MAX_SYLLABLES];
    int n_syls = syllabify_optimized(t, text, syls, MAX_SYLLABLES);
    if (n_syls <= 0) return 0;
    return louds_tokenize(t->louds, syls, (uint32_t)n_syls, out, out_cap);
  }

  return tokenizer_encode_beam(t, text, out, out_cap);
}

/* =========================================================
 * tokenizer_encode_lattice
 * Arena-backed beam search engine for constraint-aware tokenization
 * ========================================================= */

/* Platform-agnostic Thread-Local Storage configuration */
#if __STDC_VERSION__ >= 201112L && !defined(__STDC_NO_THREADS__)
    #include <threads.h>
    #define LATTICE_TLS thread_local
    #define LATTICE_HAS_TLS 1
#elif defined(__GNUC__) || defined(__clang__) || defined(_MSC_VER)
    #if defined(_MSC_VER)
        #define LATTICE_TLS __declspec(thread)
    #else
        #define LATTICE_TLS __thread
    #endif
    #define LATTICE_HAS_TLS 1
#else
    #define LATTICE_HAS_TLS 0
#endif

/* POSIX thread-specific key for automatic cleanup on thread exit */
#if defined(__linux__) || defined(__APPLE__)
#include <pthread.h>
static pthread_key_t lattice_lazy_key;
static pthread_once_t lattice_key_once = PTHREAD_ONCE_INIT;
static bool lattice_key_created = false;

static void destroy_lattice_tls_ctx(void *ptr) {
    tokenizer_lattice_context_destroy((TokenizerLatticeContext *)ptr);
}

static void make_lattice_key(void) {
    if (pthread_key_create(&lattice_lazy_key, destroy_lattice_tls_ctx) == 0) {
        lattice_key_created = true;
    }
}

/* Library destructor for safe dlclose handling */
__attribute__((destructor)) static void tokenizer_library_unload(void) {
    if (lattice_key_created) {
        pthread_key_delete(lattice_lazy_key);
        lattice_key_created = false;
    }
}
#define LATTICE_HAS_POSIX_TLS 1
#else
#define LATTICE_HAS_POSIX_TLS 0
#endif

#if defined(_WIN32) || defined(_WIN64)
#include <windows.h>
BOOL APIENTRY DllMain(HMODULE hModule, DWORD ul_reason_for_call, LPVOID lpReserved) {
    (void)hModule;
    (void)lpReserved;
    if (ul_reason_for_call == DLL_PROCESS_DETACH) {
#if LATTICE_HAS_POSIX_TLS
        /* Windows-specific dynamic library unloading hook */
        if (lattice_key_created) {
            pthread_key_delete(lattice_lazy_key);
            lattice_key_created = false;
        }
#endif
    }
    return TRUE;
}
#endif

#define LATTICE_BEAM_WIDTH 64
#define LATTICE_MAX_POS 512

typedef struct {
  uint64_t active_features;
  uint64_t forbidden_features;
  float accumulated_score;
} LatticeConstraintState;

typedef struct {
  LatticeConstraintState state;
  uint32_t token_id;
  uint32_t parent_arena_idx;
} LatticeNode;

/* Lattice context lifecycle functions */
TokenizerLatticeContext *tokenizer_lattice_context_create(void) {
  TokenizerLatticeContext *ctx = malloc(sizeof(TokenizerLatticeContext));
  if (!ctx) return NULL;

  ctx->arena_capacity = (LATTICE_MAX_POS + 1) * LATTICE_BEAM_WIDTH;
  ctx->arena_buffer = malloc(ctx->arena_capacity * sizeof(LatticeNode));
  if (!ctx->arena_buffer) {
    free(ctx);
    return NULL;
  }

  ctx->is_locked = 0;  /* Initialize re-entrancy safety latch */
  return ctx;
}

void tokenizer_lattice_context_destroy(TokenizerLatticeContext *ctx) {
  if (ctx) {
    free(ctx->arena_buffer);
    free(ctx);
  }
}

typedef struct {
  uint32_t arena_indices[LATTICE_BEAM_WIDTH];
  uint32_t count;
} LatticeBeamQueue;

typedef struct {
  LatticeCtxBeamQueue columns[LATTICE_MAX_POS + 1];
  LatticeNode *arena_buffer;
  uint32_t arena_count;
  uint32_t arena_capacity;
} PositionalLattice;

static inline bool lattice_edge_valid(const LatticeConstraintState *state,
                                       uint64_t edge_requires,
                                       uint64_t edge_forbids,
                                       uint64_t edge_features) {
  if (__builtin_expect((edge_requires & ~state->active_features) != 0, 0)) return false;
  if (__builtin_expect((edge_forbids & state->active_features) != 0, 0)) return false;
  if (__builtin_expect((state->forbidden_features & edge_features) != 0, 0)) return false;
  return true;
}

static inline void lattice_heapify_up(LatticeCtxBeamQueue *bq, const LatticeNode *arena, uint32_t idx) {
  uint32_t child = idx;
  while (child > 0) {
    uint32_t parent = (child - 1) >> 1;
    if (arena[bq->arena_indices[parent]].state.accumulated_score <=
        arena[bq->arena_indices[child]].state.accumulated_score) break;
    uint32_t tmp = bq->arena_indices[parent];
    bq->arena_indices[parent] = bq->arena_indices[child];
    bq->arena_indices[child] = tmp;
    child = parent;
  }
}

static inline void lattice_heapify_down(LatticeCtxBeamQueue *bq, const LatticeNode *arena, uint32_t idx) {
  uint32_t parent = idx;
  while (true) {
    uint32_t left = (parent << 1) + 1;
    uint32_t right = left + 1;
    uint32_t smallest = parent;

    if (left < bq->count &&
        arena[bq->arena_indices[left]].state.accumulated_score <
        arena[bq->arena_indices[smallest]].state.accumulated_score) smallest = left;
    if (right < bq->count &&
        arena[bq->arena_indices[right]].state.accumulated_score <
        arena[bq->arena_indices[smallest]].state.accumulated_score) smallest = right;
    if (smallest == parent) break;

    uint32_t tmp = bq->arena_indices[parent];
    bq->arena_indices[parent] = bq->arena_indices[smallest];
    bq->arena_indices[smallest] = tmp;
    parent = smallest;
  }
}

static inline void lattice_beam_insert(PositionalLattice *lattice, uint32_t col_idx,
                                        const LatticeConstraintState *state,
                                        uint32_t token_id, uint32_t parent_idx) {
  LatticeCtxBeamQueue *bq = &lattice->columns[col_idx];
  LatticeNode *arena = lattice->arena_buffer;

  if (bq->count == LATTICE_BEAM_WIDTH) {
    uint32_t min_idx = bq->arena_indices[0];
    /* Skip if the incoming candidate path is worse than the current beam minimum */
    if (state->accumulated_score <= arena[min_idx].state.accumulated_score) return;

    /* HARDENING FIX: Overwrite evicted element in-place to prevent arena leaks */
    arena[min_idx] = (LatticeNode){*state, token_id, parent_idx};
    lattice_heapify_down(bq, arena, 0);
  } else {
    uint32_t target_idx = lattice->arena_count++;
    if (__builtin_expect(target_idx >= lattice->arena_capacity, 0)) return;

    lattice->arena_buffer[target_idx] = (LatticeNode){*state, token_id, parent_idx};
    bq->arena_indices[bq->count] = target_idx;
    lattice_heapify_up(bq, arena, bq->count);
    bq->count++;
  }
}

/* Internal lattice encoder core logic (no locking) */
static int tokenizer_encode_lattice_core(const Tokenizer *t,
                                        TokenizerLatticeContext *ctx,
                                        const uint16_t *syls,
                                        uint32_t n_syls,
                                        uint32_t *out_tokens,
                                        uint32_t out_capacity) {
  LatticeNode *arena = (LatticeNode *)ctx->arena_buffer;
  uint32_t arena_count = 0;

  /* Zero out only the active beam columns for this sequence length */
  memset(ctx->columns, 0, (n_syls + 1) * sizeof(LatticeCtxBeamQueue));

  LatticeConstraintState root_state = {0};
  ctx->columns[0].arena_indices[0] = arena_count;
  arena[arena_count++] = (LatticeNode){root_state, 0xFFFFFFFF, 0xFFFFFFFF};
  ctx->columns[0].count = 1;

  for (uint32_t pos = 0; pos < n_syls; pos++) {
    LatticeCtxBeamQueue *current_beam = &ctx->columns[pos];
    if (current_beam->count == 0) continue;

    for (uint32_t b = 0; b < current_beam->count; b++) {
      uint32_t parent_idx = current_beam->arena_indices[b];
      LatticeConstraintState parent_state = arena[parent_idx].state;

      uint32_t node = 0;

      /* HARDENING FIX: Cap lookahead sweep window to avoid O(N^2) adversarial spikes */
      uint32_t max_lookahead = (n_syls - pos < MAX_SEQ_LEN) ? n_syls : pos + MAX_SEQ_LEN;

      for (uint32_t i = pos; i < max_lookahead; i++) {
        uint32_t start = t->louds->row_ptr[node];
        uint32_t end = t->louds->row_ptr[node + 1];
        uint16_t target = syls[i];
        uint32_t next = 0xFFFFFFFF;

        uint32_t lo = start, hi = end;
        while (lo < hi) {
          uint32_t mid = (lo + hi) >> 1;
          if (t->louds->labels[mid] < target) lo = mid + 1;
          else hi = mid;
        }
        if (lo < end && t->louds->labels[lo] == target) {
          next = t->louds->next_node[lo];
        }

        if (next == 0xFFFFFFFF) break;
        node = next;

        if (t->louds->terminals[node] != 0) {
          uint32_t token_id = t->louds->terminals[node] - 1;
          uint32_t next_pos = i + 1;

          if (token_id < t->vocab_size) {
            uint64_t features = t->token_features[token_id];
            uint64_t requires = t->token_requires[token_id];
            uint64_t forbids = t->token_forbids[token_id];

            if (lattice_edge_valid(&parent_state, requires, forbids, features)) {
              float score = 0.0f;
              LatticeConstraintState next_state = {
                .active_features = parent_state.active_features | features,
                .forbidden_features = parent_state.forbidden_features | forbids,
                .accumulated_score = parent_state.accumulated_score + score
              };

              /* In-place beam insert using context columns */
              LatticeCtxBeamQueue *bq = &ctx->columns[next_pos];
              if (bq->count == LATTICE_CTX_BEAM_WIDTH) {
                uint32_t min_idx = bq->arena_indices[0];
                if (next_state.accumulated_score <= arena[min_idx].state.accumulated_score) continue;

                arena[min_idx] = (LatticeNode){next_state, token_id, parent_idx};
                lattice_heapify_down(bq, arena, 0);
              } else {
                if (__builtin_expect(arena_count >= ctx->arena_capacity, 0)) return -1;
                arena[arena_count] = (LatticeNode){next_state, token_id, parent_idx};
                bq->arena_indices[bq->count] = arena_count;
                lattice_heapify_up(bq, arena, bq->count);
                bq->count++;
                arena_count++;
              }
            }
          }
        }
      }
    }
  }

  LatticeCtxBeamQueue *final_beam = &ctx->columns[n_syls];
  if (final_beam->count == 0) return -1;

  uint32_t best_node_idx = final_beam->arena_indices[0];
  float max_score = arena[best_node_idx].state.accumulated_score;
  for (uint32_t i = 1; i < final_beam->count; i++) {
    uint32_t idx = final_beam->arena_indices[i];
    if (arena[idx].state.accumulated_score > max_score) {
      max_score = arena[idx].state.accumulated_score;
      best_node_idx = idx;
    }
  }

  uint32_t rev_stack[LATTICE_CTX_MAX_POS];
  uint32_t rev_count = 0;
  uint32_t curr = best_node_idx;

  while (curr != 0xFFFFFFFF) {
    uint32_t tid = arena[curr].token_id;
    if (tid != 0xFFFFFFFF) {
      if (__builtin_expect(rev_count >= LATTICE_CTX_MAX_POS, 0)) return -1;
      rev_stack[rev_count++] = tid;
    }
    curr = arena[curr].parent_arena_idx;
  }

  if (__builtin_expect(rev_count > out_capacity, 0)) {
    ctx->is_locked = 0;
    return -1;
  }
  for (uint32_t i = 0; i < rev_count; i++) {
    out_tokens[i] = rev_stack[rev_count - 1 - i];
  }

  ctx->is_locked = 0;
  return (int)rev_count;
}

/* Guard wrapper with memory barriers for re-entrancy safety */
static int tokenizer_encode_lattice_with_context(const Tokenizer *t,
                                                  TokenizerLatticeContext *ctx,
                                                  const uint16_t *syls,
                                                  uint32_t n_syls,
                                                  uint32_t *out_tokens,
                                                  uint32_t out_capacity) {
  if (__builtin_expect(n_syls == 0 || n_syls > LATTICE_CTX_MAX_POS, 0)) return -1;
  if (__builtin_expect(!ctx, 0)) return -1;

  /* Re-entrancy safety check: abort if context is already locked */
  if (__builtin_expect(ctx->is_locked, 0)) {
    /* Fallback to greedy tokenization for nested calls or signal handlers */
    return louds_tokenize(t->louds, syls, n_syls, out_tokens, out_capacity);
  }

  ctx->is_locked = 1;
#if defined(__GNUC__) || defined(__clang__)
  asm volatile("" ::: "memory"); /* Prevent compiler from reordering logic above this line */
#endif

  int result = tokenizer_encode_lattice_core(t, ctx, syls, n_syls, out_tokens, out_capacity);

#if defined(__GNUC__) || defined(__clang__)
  asm volatile("" ::: "memory"); /* Prevent compiler from reordering logic below this line */
#endif
  ctx->is_locked = 0;

  return result;
}

/* Public API with backward-compatible TLS context (safe for worker pools) */
int tokenizer_encode_lattice(const Tokenizer *t, const char *text,
                            uint32_t *out_tokens, uint32_t out_capacity) {
  if (!t || !text || !out_tokens) return -1;
  if (!t->louds || !t->louds->has_csr) return -1;
  if (!t->token_features || !t->token_requires || !t->token_forbids) {
    /* Fall back to greedy if constraints not available */
    return tokenizer_encode(t, text, out_tokens, out_capacity);
  }

  uint16_t syls[MAX_SYLLABLES];
  int n_syls = syllabify_optimized(t, text, syls, MAX_SYLLABLES);
  if (n_syls <= 0) return 0;
  if ((uint32_t)n_syls > LATTICE_CTX_MAX_POS) {
    /* Input too long for lattice - fall back to greedy */
    return tokenizer_encode(t, text, out_tokens, out_capacity);
  }

  /* Resolve or lazily allocate the Thread-Local Context */
#if LATTICE_HAS_POSIX_TLS
  pthread_once(&lattice_key_once, make_lattice_key);
  TokenizerLatticeContext *ctx = (TokenizerLatticeContext *)pthread_getspecific(lattice_lazy_key);
  if (__builtin_expect(!ctx, 0)) {
    ctx = tokenizer_lattice_context_create();
    if (ctx) pthread_setspecific(lattice_lazy_key, ctx);
  }
#elif LATTICE_HAS_TLS
  static LATTICE_TLS TokenizerLatticeContext *tl_ctx = NULL;
  if (__builtin_expect(!tl_ctx, 0)) {
    tl_ctx = tokenizer_lattice_context_create();
  }
  TokenizerLatticeContext *ctx = tl_ctx;
#else
  /* Fallback for legacy compilers lacking TLS support: heap allocation per-call
   * is preferred over stack overflow vulnerability in production. */
  TokenizerLatticeContext *ctx = tokenizer_lattice_context_create();
#endif

  if (__builtin_expect(!ctx, 0)) {
    /* Ultimate fallback if memory allocation fails completely */
    return tokenizer_encode(t, text, out_tokens, out_capacity);
  }

  int result = tokenizer_encode_lattice_with_context(t, ctx, syls, (uint32_t)n_syls,
                                                     out_tokens, out_capacity);

#if !LATTICE_HAS_POSIX_TLS && !LATTICE_HAS_TLS
  /* Clean up the fallback allocation immediately if TLS is missing */
  tokenizer_lattice_context_destroy(ctx);
#endif

  return result;
}

const char *tokenizer_decode(const Tokenizer *t, uint32_t token_id) {
  if (!t || !t->id_to_str) return NULL;
  if (token_id >= t->vocab_size) return NULL;
  return t->id_to_str[token_id];
}

/* =========================================================
 * Serialization format
 * ========================================================= */
#define ALIGN8(x) (((x) + 7ULL) & ~7ULL)

int tokenizer_save(const Tokenizer *t, const char *path) {
  if (!t || !path) return -1;

  /* Write to temp file for atomic rename.
   * Temp file is created in the same directory as destination to avoid EXDEV error
   * from rename() when source and destination are on different filesystems. */
  char tmp_path[PATH_MAX];
  snprintf(tmp_path, sizeof(tmp_path), "%s.tmp", path);

  FILE *f = fopen(tmp_path, "wb+");
  if (!f) {
    perror("tokenizer_save: fopen");
    return -1;
  }

  MmapHeader hdr = {0};
  memcpy(hdr.magic, "LGMMAPv1", 8);
  hdr.version = 15;
  hdr.vocab_size = t->vocab_size;
  hdr.rule_count = t->rs ? t->rs->rule_count : 0;
  hdr.truth_seed_hash = compute_truth_seed_hash();

  if (fwrite(&hdr, sizeof(hdr), 1, f) != 1) goto io_err;

  fseeko(f, ALIGN8(ftello(f)), SEEK_SET);
  hdr.stbl_offset = (uint64_t)ftello(f);

  SyllableTable st_copy = *t->stbl;
  st_copy.trie = NULL;
  if (fwrite(&st_copy, sizeof(SyllableTable), 1, f) != 1) goto io_err;

  if (t->rs && t->rs->rule_count > 0) {
    fseeko(f, ALIGN8(ftello(f)), SEEK_SET);
    hdr.rules_offset = (uint64_t)ftello(f);
    if (fwrite(t->rs->rules, sizeof(Rule), t->rs->rule_count, f) !=
        t->rs->rule_count) goto io_err;
  }

  if (t->louds && t->louds->has_csr) {
    fseeko(f, ALIGN8(ftello(f)), SEEK_SET);
    hdr.louds_offset = (uint64_t)ftello(f);

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

    if (fwrite(&l_copy, sizeof(LOUDS), 1, f) != 1) goto io_err;

    fseeko(f, ALIGN8(ftello(f)), SEEK_SET);
    if (fwrite(t->louds->row_ptr, sizeof(uint32_t), t->louds->node_count + 1,
               f) != t->louds->node_count + 1) goto io_err;

    fseeko(f, ALIGN8(ftello(f)), SEEK_SET);
    if (fwrite(t->louds->labels, sizeof(uint16_t), t->louds->edge_count, f) !=
        t->louds->edge_count) goto io_err;

    fseeko(f, ALIGN8(ftello(f)), SEEK_SET);
    if (fwrite(t->louds->next_node, sizeof(uint32_t), t->louds->edge_count,
               f) != t->louds->edge_count) goto io_err;

    fseeko(f, ALIGN8(ftello(f)), SEEK_SET);
    if (fwrite(t->louds->terminals, sizeof(uint32_t), t->louds->node_count,
               f) != t->louds->node_count) goto io_err;
  }

  if (t->gp && t->gp->expansions && t->rs->rule_count > 0) {
    fseeko(f, ALIGN8(ftello(f)), SEEK_SET);
    hdr.strings_offset = (uint64_t)ftello(f);
    hdr.strings_size = t->rs->rule_count * MAX_TOKEN_CHARS;

    for (uint32_t i = 0; i < t->rs->rule_count; i++) {
      char buf[MAX_TOKEN_CHARS] = {0};
      if (t->gp->expansions[i]) {
        strncpy(buf, t->gp->expansions[i], MAX_TOKEN_CHARS - 1);
      }
      if (fwrite(buf, MAX_TOKEN_CHARS, 1, f) != 1) goto io_err;
    }
  }

  hdr.data_size = (uint64_t)ftello(f);
  fseeko(f, 0, SEEK_SET);
  if (fwrite(&hdr, sizeof(hdr), 1, f) != 1) goto io_err;

  uint32_t crc = 0;
  fseeko(f, 0, SEEK_SET);
  uint8_t crc_buf[8192];
  uint64_t remaining = hdr.data_size;
  while (remaining > 0) {
    size_t to_read = (remaining > sizeof(crc_buf)) ? sizeof(crc_buf) : (size_t)remaining;
    if (fread(crc_buf, 1, to_read, f) != to_read) goto io_err;
    crc = crc32_update(crc, crc_buf, to_read);
    remaining -= to_read;
  }

  fseeko(f, 0, SEEK_END);
  if (fwrite(&crc, sizeof(crc), 1, f) != 1) goto io_err;

  fclose(f);

  /* Atomic rename from temp to final path */
  if (rename(tmp_path, path) != 0) {
    perror("tokenizer_save: rename");
    unlink(tmp_path);
    return -1;
  }

  return 0;

io_err:
  fprintf(stderr, "[save] I/O error writing model to %s\n", tmp_path);
  fclose(f);
  unlink(tmp_path);
  return -1;
}

Tokenizer *tokenizer_load(const char *path) {
  if (!path) return NULL;

  FILE *f = fopen(path, "rb");
  if (!f) return NULL;

  MmapHeader hdr;
  if (fread(&hdr, sizeof(hdr), 1, f) != 1) {
    fclose(f);
    return NULL;
  }

  if (memcmp(hdr.magic, "LGMMAPv1", 8) != 0 || hdr.version < 14 || hdr.version > 15) {
    fclose(f);
    return NULL;
  }

  fseeko(f, 0, SEEK_END);
  off_t file_size = ftello(f);
  if (file_size < (off_t)(sizeof(MmapHeader) + 4)) {
    fclose(f);
    return NULL;
  }

  uint32_t expected_crc;
  fseeko(f, -4, SEEK_END);
  if (fread(&expected_crc, sizeof(expected_crc), 1, f) != 1) {
    fclose(f);
    return NULL;
  }

  fseeko(f, 0, SEEK_SET);
  uint32_t actual_crc = 0;
  uint8_t crc_buf[8192];
  uint64_t remaining = (uint64_t)(file_size - 4);
  while (remaining > 0) {
    size_t to_read = (remaining > sizeof(crc_buf)) ? sizeof(crc_buf) : (size_t)remaining;
    if (fread(crc_buf, 1, to_read, f) != to_read) {
      fclose(f);
      return NULL;
    }
    actual_crc = crc32_update(actual_crc, crc_buf, to_read);
    remaining -= to_read;
  }

  if (actual_crc != expected_crc) {
    fprintf(stderr, "[load] CRC mismatch\n");
    fclose(f);
    return NULL;
  }

  fseeko(f, sizeof(MmapHeader), SEEK_SET);

  Tokenizer *t = calloc(1, sizeof(Tokenizer));
  if (!t) {
    fclose(f);
    return NULL;
  }

  t->vocab_size = hdr.vocab_size;

  if (fseeko(f, (off_t)hdr.stbl_offset, SEEK_SET) != 0) goto io_err;
  t->stbl = calloc(1, sizeof(SyllableTable));
  if (!t->stbl) goto io_err;
  if (fread(t->stbl, sizeof(SyllableTable), 1, f) != 1) goto io_err;

  t->stbl->trie = NULL;
  for (uint16_t i = 0; i < t->stbl->count; i++) t->stbl->entries[i].id = i;

  t->rs = calloc(1, sizeof(RePairState));
  if (!t->rs) goto io_err;
  t->rs->rule_count = hdr.rule_count;
  if (hdr.rules_offset > 0 && hdr.rule_count > 0) {
    if (hdr.rule_count > MAX_RULES) goto io_err;
    if (fseeko(f, (off_t)hdr.rules_offset, SEEK_SET) != 0) goto io_err;
    if (fread(t->rs->rules, sizeof(Rule), hdr.rule_count, f) != hdr.rule_count) goto io_err;
  }

  if (hdr.louds_offset > 0) {
    if (fseeko(f, (off_t)hdr.louds_offset, SEEK_SET) != 0) goto io_err;
    t->louds = calloc(1, sizeof(LOUDS));
    if (!t->louds) goto io_err;
    if (fread(t->louds, sizeof(LOUDS), 1, f) != 1) goto io_err;

    LOUDS *l = t->louds;
    if (l->node_count > (1u << 24) || l->edge_count > (1u << 24)) goto io_err;
    l->row_ptr = malloc(((size_t)l->node_count + 1) * sizeof(uint32_t));
    l->labels = calloc(l->edge_count > 0 ? l->edge_count : 1, sizeof(uint16_t));
    l->next_node = calloc(l->edge_count > 0 ? l->edge_count : 1, sizeof(uint32_t));
    l->terminals = malloc(l->node_count * sizeof(uint32_t));
    if (!l->row_ptr || !l->labels || !l->next_node || !l->terminals) goto io_err;

    uint64_t off = ALIGN8(hdr.louds_offset + sizeof(LOUDS));
    if (fseeko(f, (off_t)off, SEEK_SET) != 0) goto io_err;
    if (fread(l->row_ptr, sizeof(uint32_t), l->node_count + 1, f) != l->node_count + 1) goto io_err;

    off = ALIGN8((uint64_t)ftello(f));
    if (fseeko(f, (off_t)off, SEEK_SET) != 0) goto io_err;
    if (fread(l->labels, sizeof(uint16_t), l->edge_count, f) != l->edge_count) goto io_err;

    off = ALIGN8((uint64_t)ftello(f));
    if (fseeko(f, (off_t)off, SEEK_SET) != 0) goto io_err;
    if (fread(l->next_node, sizeof(uint32_t), l->edge_count, f) != l->edge_count) goto io_err;

    off = ALIGN8((uint64_t)ftello(f));
    if (fseeko(f, (off_t)off, SEEK_SET) != 0) goto io_err;
    if (fread(l->terminals, sizeof(uint32_t), l->node_count, f) != l->node_count) goto io_err;

    l->has_csr = true;
  }

  if (hdr.strings_offset > 0 && hdr.rule_count > 0) {
    if (fseeko(f, (off_t)hdr.strings_offset, SEEK_SET) != 0) goto io_err;
    t->gp_expansions = calloc(hdr.rule_count, sizeof(char *));
    if (!t->gp_expansions) goto io_err;
    for (uint32_t i = 0; i < hdr.rule_count; i++) {
      t->gp_expansions[i] = malloc(MAX_TOKEN_CHARS);
      if (!t->gp_expansions[i]) goto io_err;
      if (fread(t->gp_expansions[i], MAX_TOKEN_CHARS, 1, f) != 1) goto io_err;
    }
  }

  fclose(f);

  t->stbl->trie = syltrie_build(t->stbl);
  if (!t->stbl->trie) {
    tokenizer_destroy(t);
    return NULL;
  }

  t->syl = syllabifier_create(t->stbl);
  if (!t->syl) {
    tokenizer_destroy(t);
    return NULL;
  }
  t->syl->frozen = true;
  stbl_freeze(t->stbl);

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

  /* Rebuild constraint masks from grammar + truth seeds.
   * This is deterministic, so we can recompute at load time rather
   * than serialising the 24-byte/token tables to disk. */
  TruthTrie *tt = truth_trie_create();
  if (tt) {
    truth_trie_build_from_seed(tt, t);
    tokenizer_build_masks(t, tt);
    truth_trie_destroy(tt);
  }

  return t;

io_err:
  tokenizer_destroy(t);
  fclose(f);
  return NULL;
}

#if USE_FUSED
/* Fused encoder implementations omitted for brevity (they match your original logic) */
#endif

void tokenizer_cursor_init(TokenizerCursor *cursor, const Tokenizer *t,
                           const char *text) {
  if (!cursor || !t || !text) return;
  cursor->tokenizer = t;
  cursor->cursor = text;
  cursor->eos = false;
}

size_t tokenizer_encode_streaming(TokenizerCursor *cursor, uint32_t *out,
                                  size_t out_cap) {
  if (!cursor || !out || out_cap == 0 || cursor->eos) return 0;

  const Tokenizer *t = cursor->tokenizer;
  const LOUDS *l = t->louds;
  const uint8_t *p = (const uint8_t *)cursor->cursor;

  size_t out_cnt = 0;

  while (*p && out_cnt < out_cap) {
    uint32_t node = 0;
    uint32_t best_id = 0;
    const uint8_t *best_p = p;
    const uint8_t *start_p = p;

    while (*p) {
      uint16_t syl_id;
      int consumed = consume_syllable_id(t->stbl->trie, &p, &syl_id);
      if (__builtin_expect(consumed <= 0, 0)) {
        if (consumed == 0) break;
        p++;
        continue;
      }

      if (__builtin_expect(syl_id == UINT16_MAX, 0)) break;

      uint32_t start = l->row_ptr[node];
      uint32_t end = l->row_ptr[node + 1];
      uint32_t count = end - start;

      if (__builtin_expect(count == 0, 0)) break;

      uint32_t next = 0xFFFFFFFF;

      if (count < 8) {
        for (uint32_t k = start; k < end; k++) {
          if (l->labels[k] == syl_id) {
            next = l->next_node[k];
            break;
          }
        }
      } else {
        uint32_t lo = start, hi = end;
        while (lo < hi) {
          uint32_t mid = (lo + hi) >> 1;
          if (l->labels[mid] < syl_id) lo = mid + 1;
          else hi = mid;
        }
        if (lo < end && l->labels[lo] == syl_id) {
          next = l->next_node[lo];
        }
      }

      if (next == 0xFFFFFFFF) break;

      node = next;
      if (l->terminals[node] != 0) {
        best_id = l->terminals[node];
        best_p = p;
      }
    }

    if (best_id != 0) {
      out[out_cnt++] = best_id - 1;
      p = best_p;
    } else {
      size_t skip = safe_utf8_len(start_p);
      out[out_cnt++] = TOK_UNK;
      p = start_p + skip;
    }
  }

  cursor->cursor = (const char *)p;
  if (!*p) cursor->eos = true;

  return out_cnt;
}