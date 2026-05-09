#ifndef REPAIR_TOKENIZER_H
#define REPAIR_TOKENIZER_H

/*
 * repair_tokenizer.h
 * ==================
 * Luganda morphology-aware tokenizer using:
 *   Syllabifier → Re-Pair → Grammar Pruning → LOUDS Succinct Automaton
 *
 * Symbol encoding (uint32_t):
 *   [0,   BASE_SYMBOLS)  → raw syllable IDs
 *   [BASE_SYMBOLS, ...)  → grammar rule IDs (rule_id + BASE_SYMBOLS)
 */

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>
#include <stdio.h>

/* ──────────────────────────────────────────────────────────── */
/*  Compile-time limits                                         */
/* ──────────────────────────────────────────────────────────── */
#define BASE_SYMBOLS      1024u   /* max distinct syllables          */
#define MAX_RULES         65536u  /* max Re-Pair grammar rules        */
#define MAX_RULE_DEPTH    12u     /* max nesting depth before flatten */
#define MIN_PAIR_FREQ     3u      /* break-even threshold (f > 2)     */
#define MAX_VOCAB         40960u  /* BASE_SYMBOLS + MAX_RULES         */
#define MAX_FLAT_LEN      64u     /* max syllables in one flattened rule */
#define MAX_SEQ_LEN       (1 << 24) /* max symbols in training corpus */
#define PAIR_HTABLE_SIZE  (1 << 20) /* hash table buckets (power of 2) */

/* sentinel – marks deleted/invalid slots */
#define INVALID_SYM  UINT32_MAX

/* ──────────────────────────────────────────────────────────── */
/*  Syllabifier                                                 */
/* ──────────────────────────────────────────────────────────── */

typedef struct {
    char   *table[BASE_SYMBOLS]; /* syllable_id → UTF-8 string    */
    uint32_t count;              /* number of distinct syllables   */
} SyllableTable;

/* Initialise table; returns 0 on success. */
int  syllable_table_init(SyllableTable *st);
void syllable_table_free(SyllableTable *st);

/*
 * Tokenise one UTF-8 word into syllable IDs.
 * out_ids must hold at least 64 entries.
 * Returns number of syllable IDs written, or -1 on error.
 */
int syllabify_word(SyllableTable *st,
                   const char   *word,
                   uint32_t     *out_ids,
                   int           out_cap);

/* ──────────────────────────────────────────────────────────── */
/*  Re-Pair sequence (doubly-linked list over symbol array)     */
/* ──────────────────────────────────────────────────────────── */

typedef struct SeqNode {
    uint32_t sym;       /* symbol value                 */
    int32_t  prev;      /* index of previous node (-1)  */
    int32_t  next;      /* index of next node    (-1)   */
    bool     active;    /* false = logically deleted    */
} SeqNode;

typedef struct {
    SeqNode  *nodes;    /* flat node pool               */
    uint32_t  len;      /* total allocated nodes        */
    int32_t   head;     /* index of first active node   */
} Sequence;

int  seq_init(Sequence *s, uint32_t capacity);
void seq_free(Sequence *s);

/* Append symbol; returns node index or -1 on OOM. */
int32_t seq_append(Sequence *s, uint32_t sym);

/* Logically remove node at index i from the linked list. */
void seq_remove(Sequence *s, int32_t i);

/* ──────────────────────────────────────────────────────────── */
/*  Pair frequency table (open-addressing hash map)             */
/* ──────────────────────────────────────────────────────────── */

typedef struct PairEntry {
    uint32_t left;
    uint32_t right;
    uint32_t freq;
    bool     occupied;
} PairEntry;

typedef struct {
    PairEntry *buckets;
    uint32_t   capacity; /* must be power of 2 */
    uint32_t   count;
} PairTable;

int  pair_table_init(PairTable *pt, uint32_t capacity);
void pair_table_free(PairTable *pt);

/* Increment count for (left,right); returns new count or 0 on OOM. */
uint32_t pair_table_inc(PairTable *pt, uint32_t left, uint32_t right);

/* Decrement count; returns new count (saturates at 0). */
uint32_t pair_table_dec(PairTable *pt, uint32_t left, uint32_t right);

/* Look up without modification; returns 0 if absent. */
uint32_t pair_table_get(const PairTable *pt, uint32_t left, uint32_t right);

/* ──────────────────────────────────────────────────────────── */
/*  Max-heap priority queue for (score, pair)                   */
/* ──────────────────────────────────────────────────────────── */

typedef struct {
    uint64_t score;     /* packed score – higher is better */
    uint32_t left;
    uint32_t right;
} HeapEntry;

typedef struct {
    HeapEntry *data;
    uint32_t   size;
    uint32_t   cap;
} MaxHeap;

int  heap_init(MaxHeap *h, uint32_t capacity);
void heap_free(MaxHeap *h);
int  heap_push(MaxHeap *h, HeapEntry e);         /* 0 ok, -1 OOM */
HeapEntry heap_pop(MaxHeap *h);                  /* UB if empty  */
bool heap_empty(const MaxHeap *h);

/* ──────────────────────────────────────────────────────────── */
/*  Grammar rule                                                */
/* ──────────────────────────────────────────────────────────── */

typedef struct {
    uint32_t left;        /* left  child symbol             */
    uint32_t right;       /* right child symbol             */
    uint32_t freq;        /* observed frequency in corpus   */
    uint32_t eff_freq;    /* effective freq (after nesting) */
    uint8_t  depth;       /* nesting depth                  */
    bool     dead;        /* true = pruned / inlined        */
} GrammarRule;

typedef struct {
    GrammarRule *rules;
    uint32_t     count;   /* number of rules created so far */
    uint32_t     cap;
    uint32_t     base;    /* = BASE_SYMBOLS (symbol offset)  */
} Grammar;

int  grammar_init(Grammar *g);
void grammar_free(Grammar *g);

/* Add rule X → left right; returns new symbol ID for X. */
uint32_t grammar_add(Grammar *g, uint32_t left, uint32_t right, uint32_t freq);

/* ──────────────────────────────────────────────────────────── */
/*  Re-Pair trainer                                             */
/* ──────────────────────────────────────────────────────────── */

typedef struct {
    uint32_t base_vocab;   /* = BASE_SYMBOLS                      */
    uint32_t max_rules;    /* stop after this many rules          */
    uint32_t min_freq;     /* only replace pairs with freq ≥ this */
    uint32_t max_depth;    /* flatten rules deeper than this      */
} RePairConfig;

/*
 * Run Re-Pair on `seq` using `st` for syllable info.
 * Fills `g` with learned rules.
 * Returns 0 on success, -1 on error.
 */
int repair_train(Sequence          *seq,
                 const SyllableTable *st,
                 Grammar           *g,
                 const RePairConfig *cfg);

/* ──────────────────────────────────────────────────────────── */
/*  Grammar post-processing                                     */
/* ──────────────────────────────────────────────────────────── */

/* Compute effective frequencies (subtract parent usages). */
void grammar_compute_eff_freq(Grammar *g);

/* Mark rules whose eff_freq < threshold as dead. */
void grammar_prune(Grammar *g, uint32_t min_eff_freq);

/* Inline dead rules into their parents (grammar flattening).
 * After this, each live rule's children are either base syllables
 * or other live rules with depth ≤ max_depth.
 */
void grammar_flatten(Grammar *g, uint8_t max_depth);

/* ──────────────────────────────────────────────────────────── */
/*  Trie                                                        */
/* ──────────────────────────────────────────────────────────── */

#define TRIE_NULL UINT32_MAX

typedef struct TrieNode {
    uint32_t  child;    /* first child index (TRIE_NULL = leaf) */
    uint32_t  sibling;  /* next sibling index                   */
    uint32_t  label;    /* symbol on edge from parent           */
    uint32_t  token_id; /* assigned token if terminal           */
    bool      terminal;
} TrieNode;

typedef struct {
    TrieNode *nodes;
    uint32_t  count;
    uint32_t  cap;
} Trie;

int  trie_init(Trie *t);
void trie_free(Trie *t);

/*
 * Insert a token path (sequence of symbols) into the trie.
 * token_id is stored at the terminal node.
 */
int trie_insert(Trie *t,
                const uint32_t *path,
                uint32_t        path_len,
                uint32_t        token_id);

/* ──────────────────────────────────────────────────────────── */
/*  LOUDS succinct representation                               */
/* ──────────────────────────────────────────────────────────── */

/*
 * LOUDS (Level-Order Unary Degree Sequence):
 *   For each node in BFS order, write deg(v) ones followed by one zero.
 *   Total bits = 2*N + 1 where N = number of nodes.
 *
 *   rank1(i)  = number of 1-bits in bits[0..i]
 *   select1(k) = position of k-th 1-bit (1-indexed)
 *   select0(k) = position of k-th 0-bit
 *
 * Node v (1-indexed) occupies position select1(v) in the bitstring.
 * Its children span from select0(v)+1 to select0(v+1).
 * The i-th child of v is node rank1(select0(v) + i).
 */

#define LOUDS_BLOCK 64u

typedef struct {
    uint64_t *bits;         /* topology bitvector                  */
    uint32_t  bits_len;     /* number of uint64_t blocks           */
    uint32_t  n_nodes;      /* total nodes (root = node 1)         */

    uint32_t *labels;       /* labels[i] = label on edge to node i */
    uint32_t *token_ids;    /* token_ids[i] = token at node i      */
    bool     *terminal;     /* terminal[i] = is node i a terminal? */

    /* Rank/select acceleration: one entry per LOUDS_BLOCK bits */
    uint32_t *rank_table;   /* rank_table[i] = popcount(bits[0..i*64-1]) */
    uint32_t  rank_table_len;
} LOUDS;

int  louds_build(LOUDS *L, const Trie *t);
void louds_free(LOUDS *L);

/* Returns token_id for the sequence, or INVALID_SYM if not found. */
uint32_t louds_lookup(const LOUDS    *L,
                      const uint32_t *syms,
                      uint32_t        len);

/* ──────────────────────────────────────────────────────────── */
/*  Top-level tokenizer                                         */
/* ──────────────────────────────────────────────────────────── */

typedef struct {
    SyllableTable  syllables;
    Grammar        grammar;
    LOUDS          automaton;
    uint32_t       vocab_size;
} Tokenizer;

/* Build tokenizer from a training corpus file.  Returns 0 on success. */
int tokenizer_train(Tokenizer  *tok,
                    const char *corpus_path,
                    const RePairConfig *cfg);

/* Free all tokenizer memory. */
void tokenizer_free(Tokenizer *tok);

/*
 * Tokenize a UTF-8 string.
 * out_ids must have capacity for at least `out_cap` token IDs.
 * Returns number of tokens, or -1 on error.
 */
int tokenizer_encode(const Tokenizer *tok,
                     const char      *text,
                     uint32_t        *out_ids,
                     int              out_cap);

/* Serialize / deserialize model to binary file */
int tokenizer_save(const Tokenizer *tok, const char *path);
int tokenizer_load(Tokenizer *tok, const char *path);

#endif /* REPAIR_TOKENIZER_H */
