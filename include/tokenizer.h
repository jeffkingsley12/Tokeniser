#ifndef TOKENIZER_H
#define TOKENIZER_H

#include "tokenizer_config.h"

#include <stdbool.h>
#include <stdatomic.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

/* CRC32 table for model integrity verification */
extern const uint32_t crc32_table[256];

uint32_t crc32_update(uint32_t crc, const uint8_t *data, size_t len);

/* Overflow-safe allocation helpers.
 * Both helpers guard against count*size integer overflow before
 * forwarding to the underlying allocator.  safe_malloc also guards
 * the degenerate count==0 case that makes malloc(0) implementation-defined. */
static inline void *safe_calloc(size_t count, size_t size) {
    if (count == 0 || size == 0) return NULL;
    if (size > SIZE_MAX / count) {
        fprintf(stderr, "[alloc] size overflow: %zu * %zu\n", count, size);
        return NULL;
    }
    return calloc(count, size);
}

static inline void *safe_malloc(size_t count, size_t size) {
    if (count == 0 || size == 0) return NULL;
    if (size > SIZE_MAX / count) {
        fprintf(stderr, "[alloc] size overflow: %zu * %zu\n", count, size);
        return NULL;
    }
    return malloc(count * size);
}

#define DENSE_HASH_SIZE      1024
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
typedef struct SylTrieNode {
    uint32_t first_edge;  /* index of first edge in LOUDS::edges[] */
    uint32_t token_id;    /* 0 = not a terminal; >0 = token ID + 1 */
    uint16_t edge_count;
    uint16_t id;          /* node's own index (for debugging) */
} SylTrieNode;

typedef struct SylTrieEdge {
    uint16_t label;      /* syllable ID on this edge */
    uint32_t next_node;  /* index of child SylTrieNode */
} SylTrieEdge;

typedef struct {
    SylTrieNode *nodes;
    SylTrieEdge *edges;
    uint32_t     node_count;
    uint32_t     edge_count;
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
    SyllableEntry entries[BASE_SYMBOL_OFFSET];   /* dense array, index = syllable ID */

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
    uint16_t cv_to_token[65536];
    uint32_t space_cv_to_token[65536];
} __attribute__((aligned(64))) TokenizerFastPaths;

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

    TokenizerFastPaths *fast_paths;
    uint16_t byte_to_syl[65536];
    uint16_t byte_to_token[256];
    uint32_t byte_direct[256];

    uint16_t cv_dense_keys[DENSE_HASH_SIZE];
    uint32_t cv_dense_vals[DENSE_HASH_SIZE];
    uint16_t sp_dense_keys[DENSE_HASH_SIZE];
    uint32_t sp_dense_vals[DENSE_HASH_SIZE];
    char   **morph_strings;

    void     *mmap_addr;
    size_t    mmap_len;
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

typedef struct RePairSlabBlock {
    struct RePairSlabBlock *next;
    uint32_t                used;
} RePairSlabBlock;

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
