#ifndef REPAIR_TRAIN_H
#define REPAIR_TRAIN_H

/*
 * repair_train.h
 * ===============
 * Training-only definitions for the Luganda Re-Pair tokenizer.
 * 
 * This header contains types and macros used ONLY during training.
 * All symbols are prefixed with TRAIN_ to avoid ODR violations with
 * the production tokenizer.h header.
 *
 * NEVER include this header from tokenizer.h - it should only be
 * included by repair_tokenizer.h and training implementation files.
 */

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>
#include <stdio.h>

/* ──────────────────────────────────────────────────────────── */
/*  Compile-time limits (training-only)                         */
/* ──────────────────────────────────────────────────────────── */
#ifndef TRAIN_BASE_SYMBOLS
#define TRAIN_BASE_SYMBOLS      1024u   /* max distinct syllables          */
#endif

#ifndef TRAIN_MAX_RULES
#define TRAIN_MAX_RULES         65536u  /* max Re-Pair grammar rules        */
#endif

#ifndef TRAIN_MAX_RULE_DEPTH
#define TRAIN_MAX_RULE_DEPTH    12u     /* max nesting depth before flatten */
#endif

#ifndef TRAIN_MIN_PAIR_FREQ
#define TRAIN_MIN_PAIR_FREQ     3u      /* break-even threshold (f > 2)     */
#endif

#ifndef TRAIN_MAX_VOCAB
#define TRAIN_MAX_VOCAB         40960u  /* BASE_SYMBOLS + MAX_RULES         */
#endif

#ifndef TRAIN_MAX_FLAT_LEN
#define TRAIN_MAX_FLAT_LEN      64u     /* max syllables in one flattened rule */
#endif

#ifndef TRAIN_MAX_SEQ_LEN
#define TRAIN_MAX_SEQ_LEN       (1 << 24) /* max symbols in training corpus */
#endif

#ifndef TRAIN_PAIR_HTABLE_SIZE
#define TRAIN_PAIR_HTABLE_SIZE  (1 << 20) /* hash table buckets (power of 2) */
#endif

/* sentinel – marks deleted/invalid slots */
#ifndef TRAIN_INVALID_SYM
#define TRAIN_INVALID_SYM  UINT32_MAX
#endif

/* ──────────────────────────────────────────────────────────── */
/*  Syllabifier (training-only)                                 */
/* ──────────────────────────────────────────────────────────── */

typedef struct TrainSyllableTable {
    char *table[TRAIN_BASE_SYMBOLS]; /* syllable_id → UTF-8 string    */
    uint32_t count;                  /* number of distinct syllables   */
} TrainSyllableTable;

/* Initialise table; returns 0 on success. */
int  train_syllable_table_init(TrainSyllableTable *st);
void train_syllable_table_free(TrainSyllableTable *st);

/*
 * Tokenise one UTF-8 word into syllable IDs.
 * out_ids must hold at least 64 entries.
 * Returns number of syllable IDs written, or -1 on error.
 */
int train_syllabify_word(TrainSyllableTable *st,
                         const char         *word,
                         uint32_t           *out_ids,
                         int                 out_cap);

/* ──────────────────────────────────────────────────────────── */
/*  Re-Pair sequence (doubly-linked list over symbol array)     */
/* ──────────────────────────────────────────────────────────── */

typedef struct TrainSeqNode {
    uint32_t sym;       /* symbol value                 */
    int32_t  prev;      /* index of previous node (-1)  */
    int32_t  next;      /* index of next node    (-1)   */
    bool     active;    /* false = logically deleted    */
} TrainSeqNode;

typedef struct TrainSequence {
    TrainSeqNode *nodes;    /* flat node pool               */
    uint32_t     len;       /* total allocated nodes        */
    int32_t      head;      /* index of first active node   */
} TrainSequence;

int  train_seq_init(TrainSequence *s, uint32_t capacity);
void train_seq_free(TrainSequence *s);

/* Append symbol; returns node index or -1 on OOM. */
int32_t train_seq_append(TrainSequence *s, uint32_t sym);

/* Logically remove node at index i from the linked list. */
void train_seq_remove(TrainSequence *s, int32_t i);

/* ──────────────────────────────────────────────────────────── */
/*  Pair frequency table (open-addressing hash map)             */
/* ──────────────────────────────────────────────────────────── */

typedef struct TrainRepairPairEntry {
    uint32_t left;
    uint32_t right;
    uint32_t freq;
    bool     occupied;
} TrainRepairPairEntry;

typedef struct TrainPairTable {
    TrainRepairPairEntry *buckets;
    uint32_t              capacity; /* must be power of 2 */
    uint32_t              count;
} TrainPairTable;

int  train_pair_table_init(TrainPairTable *pt, uint32_t capacity);
void train_pair_table_free(TrainPairTable *pt);

/* Increment count for (left,right); returns new count or 0 on OOM. */
uint32_t train_pair_table_inc(TrainPairTable *pt, uint32_t left, uint32_t right);

/* Decrement count; returns new count (saturates at 0). */
uint32_t train_pair_table_dec(TrainPairTable *pt, uint32_t left, uint32_t right);

/* Look up without modification; returns 0 if absent. */
uint32_t train_pair_table_get(const TrainPairTable *pt, uint32_t left, uint32_t right);

/* ──────────────────────────────────────────────────────────── */
/*  Max-heap priority queue for (score, pair)                   */
/* ──────────────────────────────────────────────────────────── */

typedef struct TrainHeapEntry {
    uint64_t score;     /* packed score – higher is better */
    uint32_t left;
    uint32_t right;
} TrainHeapEntry;

typedef struct TrainMaxHeap {
    TrainHeapEntry *data;
    uint32_t        size;
    uint32_t        cap;
} TrainMaxHeap;

int  train_heap_init(TrainMaxHeap *h, uint32_t capacity);
void train_heap_free(TrainMaxHeap *h);
int  train_heap_push(TrainMaxHeap *h, TrainHeapEntry e);         /* 0 ok, -1 OOM */
TrainHeapEntry train_heap_pop(TrainMaxHeap *h);                  /* UB if empty - caller must guarantee !train_heap_empty(h) */
bool train_heap_empty(const TrainMaxHeap *h);

/* Checked variant: returns false if heap is empty, true if pop succeeded */
bool train_heap_try_pop(TrainMaxHeap *h, TrainHeapEntry *out);

/* ──────────────────────────────────────────────────────────── */
/*  Grammar rule                                                */
/* ──────────────────────────────────────────────────────────── */

typedef struct TrainGrammarRule {
    uint32_t left;        /* left  child symbol             */
    uint32_t right;       /* right child symbol             */
    uint32_t freq;        /* observed frequency in corpus   */
    uint32_t eff_freq;    /* effective freq (after nesting) */
    uint8_t  depth;       /* nesting depth                  */
    bool     dead;        /* true = pruned / inlined        */
} TrainGrammarRule;

typedef struct TrainGrammar {
    TrainGrammarRule *rules;
    uint32_t          count;   /* number of rules created so far */
    uint32_t          cap;
    uint32_t          base;    /* = TRAIN_BASE_SYMBOLS (symbol offset)  */
} TrainGrammar;

int  train_grammar_init(TrainGrammar *g);
void train_grammar_free(TrainGrammar *g);

/* Add rule X → left right; returns new symbol ID for X, or TRAIN_INVALID_SYM on failure.
 * Caller must check for TRAIN_INVALID_SYM before using the returned symbol. */
uint32_t train_grammar_add(TrainGrammar *g, uint32_t left, uint32_t right, uint32_t freq);

/* ──────────────────────────────────────────────────────────── */
/*  Re-Pair trainer                                             */
/* ──────────────────────────────────────────────────────────── */

typedef struct {
    uint32_t base_vocab;   /* = TRAIN_BASE_SYMBOLS                      */
    uint32_t max_rules;    /* stop after this many rules          */
    uint32_t min_freq;     /* only replace pairs with freq ≥ this */
    uint32_t max_depth;    /* flatten rules deeper than this      */
} TrainRePairConfig;

/*
 * Run Re-Pair on `seq` using `st` for syllable info.
 * Fills `g` with learned rules.
 * Returns 0 on success, -1 on error.
 */
int train_repair_train(TrainSequence          *seq,
                        const TrainSyllableTable *st,
                        TrainGrammar           *g,
                        const TrainRePairConfig *cfg);

/* ──────────────────────────────────────────────────────────── */
/*  Grammar post-processing                                     */
/* ──────────────────────────────────────────────────────────── */

/* Compute effective frequencies (subtract parent usages). */
void train_grammar_compute_eff_freq(TrainGrammar *g);

/* Mark rules whose eff_freq < threshold as dead. */
void train_grammar_prune(TrainGrammar *g, uint32_t min_eff_freq);

/* Inline dead rules into their parents (grammar flattening).
 * After this, each live rule's children are either base syllables
 * or other live rules with depth ≤ max_depth.
 */
void train_grammar_flatten(TrainGrammar *g, uint8_t max_depth);

/* ──────────────────────────────────────────────────────────── */
/*  Trie                                                        */
/* ──────────────────────────────────────────────────────────── */

#define TRAIN_TRIE_NULL UINT32_MAX

typedef struct TrainRepairTrieNode {
    uint32_t  child;    /* first child index (TRIE_NULL = leaf) */
    uint32_t  sibling;  /* next sibling index                   */
    uint32_t  label;    /* symbol on edge from parent           */
    uint32_t  token_id; /* assigned token if terminal           */
    bool      terminal;
} TrainRepairTrieNode;

typedef struct TrainTrie {
    TrainRepairTrieNode *nodes;
    uint32_t             count;
    uint32_t             cap;
} TrainTrie;

int  train_trie_init(TrainTrie *t);
void train_trie_free(TrainTrie *t);

/*
 * Insert a token path (sequence of symbols) into the trie.
 * token_id is stored at the terminal node.
 */
int train_trie_insert(TrainTrie       *t,
                      const uint32_t   *path,
                      uint32_t         path_len,
                      uint32_t         token_id);

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

#define TRAIN_LOUDS_BLOCK 64u

typedef struct TrainLOUDS {
    uint64_t *bits;         /* topology bitvector                  */
    uint32_t  bits_len;     /* number of uint64_t blocks           */
    uint32_t  n_nodes;      /* total nodes (root = node 1)         */

    uint32_t *labels;       /* labels[i] = label on edge to node i */
    uint32_t *token_ids;    /* token_ids[i] = token at node i      */
    bool     *terminal;     /* terminal[i] = is node i a terminal? */

    /* Rank/select acceleration: one entry per LOUDS_BLOCK bits */
    uint32_t *rank_table;   /* rank_table[i] = popcount(bits[0..i*64-1]) */
    uint32_t  rank_table_len;
} TrainLOUDS;

int  train_louds_build(TrainLOUDS *L, const TrainTrie *t);
void train_louds_free(TrainLOUDS *L);

/* Returns token_id for the sequence, or TRAIN_INVALID_SYM if not found. */
uint32_t train_louds_lookup(const TrainLOUDS *L,
                            const uint32_t   *syms,
                            uint32_t          len);

/* ──────────────────────────────────────────────────────────── */
/*  Top-level tokenizer (training-only)                         */
/* ──────────────────────────────────────────────────────────── */

typedef struct TrainTokenizer {
    TrainSyllableTable  syllables;
    TrainGrammar        grammar;
    TrainLOUDS          automaton;
    uint32_t            vocab_size;
} TrainTokenizer;

/* Build tokenizer from a training corpus file.  Returns 0 on success. */
int train_tokenizer_train(TrainTokenizer      *tok,
                           const char          *corpus_path,
                           const TrainRePairConfig *cfg);

/* Free all tokenizer memory. */
void train_tokenizer_free(TrainTokenizer *tok);

/*
 * Tokenize a UTF-8 string.
 * out_ids must have capacity for at least `out_cap` token IDs.
 * Returns number of tokens, or -1 on error.
 */
int train_tokenizer_encode(const TrainTokenizer *tok,
                           const char            *text,
                           uint32_t              *out_ids,
                           int                    out_cap);

/* Serialize / deserialize model to binary file */
int train_tokenizer_save(const TrainTokenizer *tok, const char *path);
int train_tokenizer_load(TrainTokenizer *tok, const char *path);

#endif /* REPAIR_TRAIN_H */
