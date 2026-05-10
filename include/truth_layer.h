#ifndef TRUTH_LAYER_H
#define TRUTH_LAYER_H

#include <stdint.h>
#include "tokenizer.h"

/* -----------------------------------------------------------------------
 * Maximum number of TruthEntry records in the TruthTrie.
 * This is a soft cap used for pre-allocation; the trie itself can hold
 * any number of terminal nodes as long as memory allows.
 * ----------------------------------------------------------------------- */
#define MAX_TRUTH_ENTRIES 65536u

/* -----------------------------------------------------------------------
 * Category bitfield for multi-role linguistic atoms.
 *
 * A single surface form (e.g. "mu") can serve multiple grammatical roles
 * simultaneously (noun-class prefix for cl.1, cl.3, cl.18).  Using a
 * bitfield rather than an enum lets the trie store all roles in one node.
 * ----------------------------------------------------------------------- */
#define TRUTH_UNKNOWN     (1u << 0)
#define TRUTH_PREFIX      (1u << 1)
#define TRUTH_ROOT        (1u << 2)
#define TRUTH_PARTICLE    (1u << 3)
#define TRUTH_SUFFIX      (1u << 4)
#define TRUTH_REDUP       (1u << 5)
#define TRUTH_LOAN        (1u << 6)
#define TRUTH_PHONOTACTIC (1u << 7)  /* canonical CV syllable */
#define TRUTH_LOCATIVE    (1u << 8)  /* locative marker */
#define TRUTH_NOUN_CLASS  (1u << 9)  /* noun-class prefix */
#define TRUTH_INFIX       (1u << 10) /* morphological infix */

/* Bit 31 is reserved as the "terminal" sentinel inside TruthTrie::nodes[].
 * It must not be set in externally visible category masks. */
#define TRUTH_TERMINAL_BIT (1u << 31)

/* Legacy enum retained for callers that use the old API.
 * New code should use the TRUTH_* bitmask constants above. */
typedef enum {
    TRUTH_CAT_UNKNOWN    = 0,
    TRUTH_CAT_PREFIX     = 1,
    TRUTH_CAT_ROOT       = 2,
    TRUTH_CAT_PARTICLE   = 3,
    TRUTH_CAT_SUFFIX     = 4,
    TRUTH_CAT_REDUP      = 5,
    TRUTH_CAT_LOAN       = 6,
} TruthCategory;

/* Per-token linguistic metadata stored at trie terminals. */
typedef struct {
    uint32_t token_id;
    uint32_t category_mask;  /* OR of TRUTH_* bits */
    uint8_t  can_standalone;
    uint8_t  attaches_to;
} TruthEntry;

/* Unified seed metadata for multi-role linguistic atoms. */
typedef struct {
    uint16_t syllable_id;
    uint32_t flags;          /* OR of TRUTH_* bits */
} SeedMeta;

/* Seed flag aliases (maps TRUTH_* names for use in seed tables). */
#define SEED_PHONOTACTIC  TRUTH_PHONOTACTIC
#define SEED_PREFIX       TRUTH_PREFIX
#define SEED_LOCATIVE     TRUTH_LOCATIVE
#define SEED_NOUN_CLASS   TRUTH_NOUN_CLASS
#define SEED_INFIX        TRUTH_INFIX
#define SEED_TRUTH        (TRUTH_PREFIX | TRUTH_ROOT | TRUTH_PARTICLE | \
                           TRUTH_SUFFIX | TRUTH_REDUP | TRUTH_LOAN)
/* SEED_FROZEN is stored in the upper bit of SeedMeta::flags to mark
 * seeds that must not be overwritten during corpus-driven updates. */
#define SEED_FROZEN       (1u << 31)

/* -----------------------------------------------------------------------
 * TruthTrie
 *
 * Aho-Corasick automaton built over syllable ID sequences.
 * node_cap and edge_cap track the *allocated* capacity of the nodes/edges
 * arrays; node_count and edge_count track the *used* count.
 *
 * The first_syl[] table provides O(1) lookup for depth-1 nodes from the
 * root indexed by syllable ID.  Only valid for syl_id < BASE_SYMBOL_OFFSET.
 * ----------------------------------------------------------------------- */
typedef struct {
    SylTrieNode   *nodes;
    SylTrieEdge   *edges;
    uint32_t    node_count;
    uint32_t    edge_count;
    uint32_t    node_cap;   /* allocated capacity of nodes[] */
    uint32_t    edge_cap;   /* allocated capacity of edges[] */

    TruthEntry *entries;
    uint32_t    n_entries;

    /* O(1) root-level lookup: first_syl[syl_id] = child node index, or 0
     * if no such child exists.  Index 0 is the root and is never a child,
     * so 0 serves as the "not found" sentinel. */
    uint32_t    first_syl[BASE_SYMBOL_OFFSET];

    uint32_t   *failure_links; /* Aho-Corasick failure function, size = node_count */
    uint16_t   *depths;        /* node depths for length calculation */
    bool        has_csr;       /* true once failure links are computed */
} TruthTrie;

/* -----------------------------------------------------------------------
 * API
 * ----------------------------------------------------------------------- */

/* Lifecycle */
TruthTrie *truth_trie_create(void);
void       truth_trie_destroy(TruthTrie *tt);

/* Build from an explicit word list (used by the Java/Kotlin bridge) */
int truth_trie_build(TruthTrie          *tt,
                     const char        **sub_words,
                     const TruthCategory *categories,
                     const uint32_t     *token_ids,
                     uint32_t            n_entries);

/* Longest-match lookup in byte-string domain (for the Java bridge) */
int truth_trie_longest_match(const TruthTrie *tt,
                             const uint8_t   *text,
                             TruthEntry      *out_entry);

/* Serialisation */
int            truth_trie_save(const TruthTrie *tt, int fd);
const uint8_t *truth_trie_load(TruthTrie *tt,
                               const uint8_t *base, size_t remaining);

/* -----------------------------------------------------------------------
 * Syllable-ID domain matching (used by the tokenizer inference path)
 * ----------------------------------------------------------------------- */

/* Maximum number of syllables in a single truth-layer match window. */
#define MAX_TRUTH_WINDOW 16u

typedef struct {
    uint32_t token_id;
    uint32_t category_mask;
} TruthMatchInfo;

typedef struct {
    uint32_t token_id;
    uint32_t start_pos;
    uint32_t length;
    uint32_t category_mask;
} TruthMatch;

/* Longest-prefix match starting at syllable_ids[0].
 * Returns the number of syllables matched (0 = no match).
 * out_match is written only on a non-zero return. */
int truth_match_ids(const TruthTrie *tt,
                    const uint16_t  *syllable_ids,
                    size_t           n_syllables,
                    TruthMatchInfo  *out_match);

/* Compute Aho-Corasick failure links.  Must be called after all insertions
 * and before any search.  Returns 0 on success, -1 on OOM. */
int truth_trie_compute_failure_links(TruthTrie *tt);

/* Find all matches in syllable_ids[0..n_syllables) using Aho-Corasick.
 * Returns the number of matches found (≤ max_matches).
 * out_matches must point to an array of at least max_matches TruthMatch. */
int aho_corasick_find_matches(const TruthTrie *tt,
                              const uint16_t  *syllable_ids,
                              size_t           n_syllables,
                              TruthMatch      *out_matches,
                              uint32_t         max_matches);

/* Seed / corpus update */
int truth_trie_build_from_seed(TruthTrie *tt, const Tokenizer *tokenizer);
int truth_trie_update_from_corpus(TruthTrie  *tt,
                                  const char **corpus_docs,
                                  uint32_t    n_docs,
                                  uint32_t    min_freq);

/* FNV-64 fingerprint of the compiled-in TRUTH_SEED_DATA.
 * Stored in MmapHeader::truth_seed_hash so that a mmap'd model can be
 * checked against the current binary's seed set at load time. */
uint64_t compute_truth_seed_hash(void);

#endif /* TRUTH_LAYER_H */
