#ifndef TOKENIZER_H
#define TOKENIZER_H

#include "tokenizer_config.h"

#include <stdbool.h>
#include <stdatomic.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>

/* CRC32 table for model integrity verification */
static const uint32_t crc32_table[256] = {
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

static inline uint32_t crc32_update(uint32_t crc, const uint8_t *data, size_t len) {
    crc = ~crc;
    for (size_t i = 0; i < len; i++) {
        crc = (crc >> 8) ^ crc32_table[(crc ^ data[i]) & 0xFF];
    }
    return ~crc;
}

/* Overflow-safe allocation helpers.
 * Both helpers guard against count*size integer overflow before
 * forwarding to the underlying allocator.  safe_malloc also guards
 * the degenerate count==0 case that makes malloc(0) implementation-defined. */
static inline void *safe_calloc(size_t count, size_t size) {
    if (count == 0 || size == 0) return calloc(1, 1); /* return valid ptr */
    if (size > SIZE_MAX / count) {
        fprintf(stderr, "[alloc] size overflow: %zu * %zu\n", count, size);
        return NULL;
    }
    return calloc(count, size);
}

static inline void *safe_malloc(size_t count, size_t size) {
    if (count == 0 || size == 0) return malloc(1);
    if (size > SIZE_MAX / count) {
        fprintf(stderr, "[alloc] size overflow: %zu * %zu\n", count, size);
        return NULL;
    }
    return malloc(count * size);
}

#define DENSE_HASH_SIZE      256
#define MAX_SEQ_LEN          32
#define MAX_RULE_DEPTH       128
#define SPECIAL_TOKENS_COUNT 5
#define TOK_UNK              0

/* Morphological boundary flags */
#define TOKEN_BOUNDARY_LEFT  (1u << 0)
#define TOKEN_BOUNDARY_RIGHT (1u << 1)
#define TOKEN_TRUTH          (1u << 2)
#define TOKEN_PHONO          (1u << 3)
#define TOKEN_BPE            (1u << 4)

/* Compositional span flags for linguistic state machine */
typedef enum {
    TF_NONE         = 0,
    TF_LOCKED_START = (1u << 0), /* Start of protected morpheme/truth span */
    TF_LOCKED_END   = (1u << 1), /* End of protected morpheme/truth span */
    TF_SEEDED       = (1u << 2), /* Interned during stbl_seed_all() */
    TF_TRUTH_MATCH  = (1u << 3), /* Matched by truth layer */
    TF_PHONO_MATCH  = (1u << 4), /* Matched by phonotactic rules */
    TF_LOCK_RIGHT   = (1u << 5), /* Adjacency (i, i+1) is locked (no merge) */
} TokenFlags;

#define NODE_LOCK_RIGHT TF_LOCK_RIGHT

/* Token with boundary metadata for morphological preservation */
typedef struct {
    uint16_t id;
    uint8_t  flags;
} Token;

/* Rule: Re-Pair grammar rule for subword merging */
typedef struct {
    uint32_t lhs;         /* new non-terminal symbol ID              */
    uint32_t rhs[2];      /* two child symbols                       */
    uint32_t freq;        /* occurrence count in corpus              */
    uint32_t eff_freq;    /* effective freq after parent accounting  */
    uint16_t depth;       /* nesting depth from terminals                */
    uint8_t  merge_flags; /* flags propagated from children          */
    bool     dead;        /* marked for elimination                  */
} Rule;

/* RePairState: Training state for grammar rules.
 *
 * NOTE: rules[] is a fixed-size inline array of MAX_RULES entries.
 * sizeof(RePairState) is therefore ~2MB; always heap-allocate this struct
 * via repair_create() — never put it on the stack. */
typedef struct {
    uint32_t rule_count;
    Rule     rules[MAX_RULES];
    uint32_t next_sym;

    /* Internal training tables (not serialized) */
    uint64_t         *pair_keys;
    uint32_t         *pair_freq;
    struct SeqNode  **head_occ;
    struct SeqNode  **tail_occ;
    uint32_t         *heap;
    uint32_t         *heap_pos;
    uint32_t          heap_size;
    uint32_t          pair_ht_size;
    uint32_t          pair_count;
    uint32_t          live_count;
    void             *slab_head;
    bool              fatal_oom;

    /* Mmap support — byte offset of the serialised rules array */
    uint64_t          rules_offset;
} RePairState;

/* MergeMask: bit-packed mask for adjacency-constrained merging.
 *
 * Bit i = 1 means the boundary between positions i and i+1 is LOCKED
 * (merging across it is forbidden).  Bit i = 0 means merging is allowed.
 * This encoding matches set_merge_allowed / is_merge_allowed below. */
typedef struct {
    uint64_t *bits;
    size_t    n_boundaries;
} MergeMask;

MergeMask *merge_mask_create(size_t n_boundaries);
void       merge_mask_destroy(MergeMask *m);

/* Callers must check m != NULL and m->bits != NULL before calling. */
static inline void set_merge_allowed(MergeMask *m, size_t i, bool allowed) {
    if (!m || !m->bits || i >= m->n_boundaries) return;
    if (allowed) m->bits[i >> 6] &= ~((uint64_t)1 << (i & 63));
    else         m->bits[i >> 6] |=  ((uint64_t)1 << (i & 63));
}

static inline bool is_merge_allowed(const MergeMask *m, size_t i) {
    if (!m || !m->bits) return true;  /* NULL mask → everything allowed */
    if (i >= m->n_boundaries) return true;
    return !(m->bits[i >> 6] & ((uint64_t)1 << (i & 63)));
}

/* -----------------------------------------------------------------------
 * Trie Structures (CSR layout)
 * Each node stores its first edge index and edge count; edges are stored
 * in a separate flat array sorted by label for binary-search lookup.
 * ----------------------------------------------------------------------- */
typedef struct {
    uint32_t first_edge;  /* index of first edge in LOUDS::edges[] */
    uint32_t token_id;    /* 0 = not a terminal; >0 = token ID + 1 */
    uint16_t edge_count;
    uint16_t id;          /* node's own index (for debugging) */
} TrieNode;

typedef TrieNode SylTrieNode; /* Alias for compatibility */

typedef struct {
    uint16_t label;      /* syllable ID on this edge */
    uint32_t next_node;  /* index of child TrieNode */
} TrieEdge;

typedef TrieEdge SylTrieEdge; /* Alias for compatibility */

typedef struct {
    TrieNode *nodes;
    TrieEdge *edges;
    uint32_t  node_count;
    uint32_t  edge_count;
    /* root_next[byte] = first node reachable from root via a byte-indexed
     * fast path (valid only for single-byte syllable IDs < 256). */
    uint32_t  root_next[256];
} SylTrie;

/* -----------------------------------------------------------------------
 * SyllableTable
 * Fixed-size hash table mapping UTF-8 strings to syllable IDs.
 * Entries are also stored in a dense array for O(1) ID→string lookup.
 * ----------------------------------------------------------------------- */
typedef struct {
    char     text[MAX_TOKEN_CHARS];
    uint16_t id;
    uint16_t flags;
} SyllableEntry;

typedef struct {
    uint32_t      count;
    SyllableEntry entries[MAX_RULES];   /* dense array, index = syllable ID */

    /* Open-addressing hash table (size 8192, load ≤ 50%) */
    uint8_t  ht_used[8192];
    uint16_t ht_keys[8192];
    char     ht_text[8192][MAX_TOKEN_CHARS];

    bool      frozen; /* true after stbl_seed_phonotactic(); no new entries */
    SylTrie  *trie;
} SyllableTable;

typedef struct Syllabifier {
    const SyllableTable *stbl;
    bool is_vowel[256];  /* byte-indexed vowel fast path */
    bool is_nasal[256];  /* byte-indexed nasal fast path */
    bool frozen;
} Syllabifier;

/* GrammarPruner: post-training analysis pass */
typedef struct {
    RePairState        *rs;
    const SyllableTable *stbl;
    char              **expansions;      /* rule_id → UTF-8 expansion string */
    uint32_t           *expansion_lens;
    bool               *expanded;
    bool                is_frozen;
    bool                in_expand[MAX_RULES]; /* cycle detection */
    bool                fatal_oom;
} GrammarPruner;

/* TokenizerStats: lightweight counters.
 * Incremented via STAT_INC/STAT_ADD macros which become no-ops when
 * ENABLE_STATS == 0. */
typedef struct {
    uint64_t syllabify_calls;
    uint64_t syllables_produced;
    uint64_t tokenize_calls;
    uint64_t tokens_produced;

    /* LOUDS traversal telemetry */
    atomic_uint_fast64_t louds_single_syl_fallbacks;
    atomic_uint_fast64_t louds_multi_syl_matches;
    atomic_uint_fast64_t louds_single_syl_matches;
    atomic_uint_fast64_t fast_path_hits;
    atomic_uint_fast64_t trie_traversals;
} TokenizerStats;

/* -----------------------------------------------------------------------
 * LOUDS CSR Trie
 * The primary token-lookup structure used during inference.
 * "CSR" = Compressed Sparse Row — each node stores (first_edge, edge_count)
 * into a flat edge array, similar to CSR sparse-matrix format.
 * ----------------------------------------------------------------------- */
typedef struct {
    /* Layer 2: Split CSR trie (production inference path) */
    uint32_t *row_ptr;     /* node_count + 1 entries */
    uint16_t *labels;      /* edge_count entries */
    uint32_t *next_node;   /* edge_count entries */
    uint32_t *terminals;   /* edge_count entries (0 = none) */
    
    uint32_t  node_count;
    uint32_t  edge_count;
    bool      has_csr;

    /* Layer 1: legacy flat representation (used by training and stream_tokenizer) */
    uint8_t  *klens;
    uint16_t *labels_legacy; /* Renamed to avoid collision */
    uint32_t *token_ids;
    uint32_t *koffs;
    uint32_t *rank1_sb;
    uint32_t  bv_len;
    uint32_t  sb_count;

    /* Fast-path tables (indexed by syllable ID) */
    uint32_t fast_token[8192];
    bool     can_extend[8192];

    /* O(1) root-level acceleration */
    uint32_t first_index[BASE_SYMBOL_OFFSET];
    uint32_t first_count[BASE_SYMBOL_OFFSET];

    uint32_t  n_entries;
    uint32_t  label_count;
    uint32_t  csr_node_count; /* used by mmap loader to compute sub-section sizes */
    uint32_t  csr_edge_count;
    void     *louds_bv;
} LOUDS;

/* -----------------------------------------------------------------------
 * Tokenizer
 * ----------------------------------------------------------------------- */
typedef struct {
    SyllableTable *stbl;
    Syllabifier   *syl;
    RePairState   *rs;
    GrammarPruner *gp;
    LOUDS         *louds;
    uint32_t       vocab_size;
    char         **gp_expansions;
    const char   **id_to_str;

    TokenizerStats stats;

    /* Fast-path lookup tables (populated during tokenizer_build).
     * cv_to_token / space_cv_to_token are indexed by a packed 16-bit key
     * formed from two consecutive syllable bytes; byte_* tables are for
     * single-byte syllable ID fast paths. */
    uint16_t cv_to_token[65536];
    uint32_t space_cv_to_token[65536];
    uint16_t byte_to_syl[65536];
    uint16_t byte_to_token[256];
    uint32_t byte_direct[256];

    uint16_t cv_dense_keys[DENSE_HASH_SIZE];
    uint32_t cv_dense_vals[DENSE_HASH_SIZE];
    uint16_t sp_dense_keys[DENSE_HASH_SIZE];
    uint32_t sp_dense_vals[DENSE_HASH_SIZE];
    char   **morph_strings;
} Tokenizer;

/* Mutable symbol sequence node used during Re-Pair training.
 * This struct is allocated from a slab; its exact size is asserted below
 * so that slab-block sizing can be computed at compile time. */
typedef struct SeqNode {
    uint32_t sym;
    uint16_t span_id;
    uint8_t  flags;
    uint8_t  _pad;
    struct SeqNode *prev;
    struct SeqNode *next;
    struct SeqNode *prev_occ;
    struct SeqNode *next_occ;
} SeqNode;

_Static_assert(sizeof(SeqNode) == 40,
    "SeqNode layout changed — update slab block sizing and serialization docs");

typedef struct SlabBlock {
    struct SlabBlock *next;
    uint32_t          used;
} SlabBlock;

/* =========================================================
 *  Public API
 * ========================================================= */

/* SyllableTable operations */
uint32_t  stbl_hash(const char *s);
int       stbl_add(SyllableTable *stbl, const char *s, uint16_t flags);
int       stbl_find(const SyllableTable *stbl, const char *s);
uint16_t  stbl_lookup(const SyllableTable *t, const char *text);
uint16_t  stbl_intern(SyllableTable *t, const char *text);
uint16_t  stbl_intern_fixed(SyllableTable *t, const char *text, uint16_t id);

/* SylTrie */
SylTrie  *syltrie_build(const SyllableTable *stbl);
void      syltrie_destroy(SylTrie *trie);
int       consume_syllable_id(const SylTrie *t, const uint8_t **pp, uint16_t *id_out);

/* Syllabifier */
Syllabifier *syllabifier_create(SyllableTable *stbl);
void         syllabifier_destroy(Syllabifier *s);
int          syllabify(Syllabifier *s, const char *text, uint16_t *out, int out_cap);

/* Re-Pair */
RePairState *repair_create(void);
void         repair_destroy(RePairState *rs);
int          repair_train(RePairState *rs,
                          const uint16_t **corpus,
                          const uint32_t *lengths,
                          uint32_t n_seqs);
int          repair_train_with_mask(RePairState *rs,
                                    const uint16_t **corpus,
                                    const uint32_t *lengths,
                                    uint32_t n_seqs,
                                    const MergeMask **masks);

typedef struct RePairCompressor RePairCompressor;
RePairCompressor *repair_compressor_init(void);
void              repair_compressor_destroy(RePairCompressor *comp);
int               repair_compress_with_context(RePairCompressor *comp, const RePairState *rs, uint32_t *seq, uint32_t *len);
int               repair_compress(const RePairState *rs, uint32_t **seq, uint32_t *len);

/* Grammar pruner */
GrammarPruner *pruner_create(RePairState *rs, const SyllableTable *stbl);
void           pruner_destroy(GrammarPruner *gp);
bool           pruner_compute_eff_freq(GrammarPruner *gp);
void           pruner_mark_dead(GrammarPruner *gp);
void           pruner_expand_all(GrammarPruner *gp, const SyllableTable *stbl);
const char    *pruner_expand(GrammarPruner *gp, uint32_t sym, const SyllableTable *stbl);

/* LOUDS trie */
LOUDS      *louds_build(const char **strings, const uint32_t *ids, uint32_t n,
                        const SyllableTable *stbl, const Syllabifier *s);
void        louds_destroy(LOUDS *l);
int         louds_tokenize(const LOUDS *l, const uint16_t *sids, uint32_t n_sids,
                           uint32_t *out, uint32_t out_cap);
void        louds_rebuild_csr(LOUDS *l);

/* ID / string mapping */
const char *id_to_str(const Tokenizer *t, uint32_t id);
uint32_t    str_to_id(const Tokenizer *t, const char *s);

/* Top-level tokenizer lifecycle */
Tokenizer  *tokenizer_build(const char **docs, uint32_t n_docs);
void        tokenizer_destroy(Tokenizer *t);

/* Encoding */
int tokenizer_encode(const Tokenizer *t, const char *text,
                     uint32_t *out, uint32_t out_cap);

#if USE_FUSED
int tokenizer_encode_fused(const Tokenizer *t, const char *text,
                           uint32_t *out, uint32_t out_cap);
int tokenizer_encode_fused_v2(const Tokenizer *t, const char *text,
                              uint32_t *out, uint32_t out_cap);
#endif

const char *tokenizer_decode(const Tokenizer *t, uint32_t token_id);

/* Simple cursor-based streaming API */
typedef struct {
    const Tokenizer *tokenizer;
    const char      *cursor;
    bool             eos;
} TokenizerCursor;

void   tokenizer_cursor_init(TokenizerCursor *cursor,
                             const Tokenizer *t, const char *text);
size_t tokenizer_encode_streaming(TokenizerCursor *cursor,
                                  uint32_t *out, size_t out_cap);

/* Full streaming tokenizer (fd or memory input) */
typedef enum {
    TOKENIZER_OK,
    TOKENIZER_MORE,
    TOKENIZER_EOF,
    TOKENIZER_ERROR
} TokenizerStatus;

/* StreamTokenizer buffer is 65536+16 bytes:
 *  65536 = STREAM_BUF_SIZE (must match stream_tokenizer.c)
 *     16 = overflow sentinel (largest possible UTF-8 sequence × 4) */
typedef struct {
    uint8_t           buf[65536 + 16];
    size_t            head;
    size_t            tail;
    bool              eof;
    uint32_t          trie_node;
    uint32_t          best_token;
    const uint8_t    *best_pos;
    bool              in_token;
    bool              output_full;
    const Tokenizer  *t;
    uint64_t          bytes_processed;
    uint32_t          best_id;
} StreamTokenizer;

void stream_tokenizer_init(StreamTokenizer *st, const Tokenizer *t);
TokenizerStatus stream_tokenizer_file(StreamTokenizer *st, int fd,
                                      uint32_t *out, size_t out_cap,
                                      size_t *out_used);
TokenizerStatus stream_tokenizer_mem(StreamTokenizer *st,
                                     const uint8_t *data, size_t len,
                                     bool is_final,
                                     uint32_t *out, size_t out_cap,
                                     size_t *out_used,
                                     size_t *bytes_consumed);

/* Persistence */
int        tokenizer_save(const Tokenizer *t, const char *path);
Tokenizer *tokenizer_load(const char *path);

/* Seed helpers */
uint32_t vocab_validate_morphemes(Tokenizer *tok);
int      stbl_seed_bytes(SyllableTable *stbl);
uint32_t stbl_seed_phonotactic(SyllableTable *stbl);
int      stbl_seed_special(SyllableTable *stbl);
int      stbl_seed_orphan_ascii(SyllableTable *stbl);

/* UTF-8 / syllable utilities */
size_t utf8_seq_len(const uint8_t *p);
int    consume_syllable_fast(const Tokenizer *t, const uint8_t **pp,
                              uint16_t *syl_id_out);
extern const uint8_t is_letter_tbl[256];

/* -----------------------------------------------------------------------
 * Memory-mapped tokenizer
 * The MmapTokenizer wraps a base Tokenizer whose data structures point
 * directly into an mmap'd file region — zero copy, read-only at runtime.
 * ----------------------------------------------------------------------- */
typedef struct {
    Tokenizer base;        /* must be first — cast-compatible with Tokenizer* */
    void     *mmap_base;
    size_t    mmap_size;
    int       fd;          /* -1 when loaded from an Android asset */
    void     *asset_handle; /* AAsset* when loaded via Android asset manager */
    bool      owns_buffer; /* true → munmap on destroy; false → caller owns */
} MmapTokenizer;

MmapTokenizer *tokenizer_load_mmap(const char *path);
MmapTokenizer *tokenizer_load_mmap_from_buffer(const void *buf, size_t size);
void           tokenizer_destroy_mmap(MmapTokenizer *mt);

/* -----------------------------------------------------------------------
 * Mmap file header (v15 layout)
 *
 * IMPORTANT: data_size is uint64_t here (matching the file format) but was
 * mistakenly declared uint32_t in the original, causing silent truncation
 * of files larger than 4 GiB.  The on-disk field is always 64-bit.
 * ----------------------------------------------------------------------- */
typedef struct {
    char     magic[8];        /* "LGMMAPv1" */
    uint32_t version;         /* 14 = CSR layout; 15 = Truth layer added */
    uint32_t _pad0;           /* reserved — keeps offsets 8-byte aligned */
    uint64_t data_size;       /* total file size in bytes (for validation) */
    uint32_t vocab_size;
    uint32_t rule_count;
    uint64_t stbl_offset;
    uint64_t rules_offset;
    uint64_t louds_offset;
    uint64_t strings_offset;
    uint32_t strings_size;
    uint32_t idmap_size;
    uint64_t idmap_offset;
    uint64_t truth_seed_hash; /* FNV-64 of seed data at build time */
} MmapHeader;

/* Compile-time assertion: the header must be a multiple of 8 bytes so that
 * all uint64_t fields remain naturally aligned inside a mmap'd file. */
_Static_assert(sizeof(MmapHeader) % 8 == 0,
    "MmapHeader size must be a multiple of 8 bytes for natural alignment");

#endif /* TOKENIZER_H */
