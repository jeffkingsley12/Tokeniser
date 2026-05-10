/*
 * louds.c
 *
 * Two-layer design:
 *
 *   Layer 1 – Working store: a flat sorted array of (syllable_seq[], token_id)
 *     pairs.  Tokenization uses a binary-search–assisted longest-match scan.
 *     This is correct by construction and straightforward to verify.
 *
 *   Layer 2 – Compact store: LOUDS bitvector trie.
 *     Built from the same data.  Used for serialisation (smaller on disk).
 *     Navigation helpers (louds_rank1 / louds_select1) are also used by the
 *     trie builder to verify the topology.
 *
 * For the tokenisation hot-path we use Layer 1.
 */

#include "tokenizer.h"

#include <assert.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdatomic.h>
#include <unistd.h>  /* for write() */
#include <endian.h>  /* for htole32/le32toh endian conversion */



/* =========================================================
 *  Flat token entry (working lookup table)
 *
 *  `key`  – syllable ID sequence packed as uint16_t[MAX_SEQ_LEN]
 *  `klen` – number of valid syllable IDs in key
 *  `id`   – token ID
 * ========================================================= */

typedef struct {
  uint16_t key[MAX_SEQ_LEN];
  uint8_t klen;
  uint32_t id;
} TokenEntry;

/*
 * Compare two token entries lexicographically by their syllable-ID sequences.
 * Used for qsort and bsearch.
 */
static int token_entry_cmp(const void *a, const void *b) {
  const TokenEntry *ta = (const TokenEntry *)a;
  const TokenEntry *tb = (const TokenEntry *)b;
  uint8_t min_len = ta->klen < tb->klen ? ta->klen : tb->klen;
  for (uint8_t i = 0; i < min_len; i++) {
    if (ta->key[i] != tb->key[i])
      return (int)ta->key[i] - (int)tb->key[i];
  }
  return (int)ta->klen - (int)tb->klen;
}

/* =========================================================
 *  Intermediate pointer trie (build-time only)
 * ========================================================= */

typedef struct PtrTrieNode {
  uint16_t label;
  uint32_t token_id;
  struct PtrTrieNode **children;
  uint32_t n_children;
  uint32_t cap_children;
} PtrTrieNode;

static PtrTrieNode *trie_node_new(uint16_t label) {
  PtrTrieNode *n = calloc(1, sizeof *n);
  if (n)
    n->label = label;
  return n;
}

static int trie_insert(PtrTrieNode *root, const uint16_t *syls, uint32_t n,
                       uint32_t tok_id) {
  PtrTrieNode *cur = root;
  for (uint32_t i = 0; i < n; i++) {
    uint16_t lbl = syls[i];
    int lo = 0, hi = (int)cur->n_children - 1, found = -1;
    while (lo <= hi) {
      int mid = (lo + hi) / 2;
      if (cur->children[mid]->label == lbl) {
        found = mid;
        break;
      } else if (cur->children[mid]->label < lbl)
        lo = mid + 1;
      else
        hi = mid - 1;
    }
    if (found >= 0) {
      cur = cur->children[found];
    } else {
      if (cur->n_children == cur->cap_children) {
        uint32_t nc = cur->cap_children ? cur->cap_children * 2 : 4;
        PtrTrieNode **arr = realloc(cur->children, nc * sizeof *arr);
        if (!arr)
          return -1;
        /* Zero-initialize new entries to prevent garbage pointers */
        if (nc > cur->cap_children) {
          memset(&arr[cur->cap_children], 0,
                 (nc - cur->cap_children) * sizeof *arr);
        }
        cur->children = arr;
        cur->cap_children = nc;
      }
      PtrTrieNode *child = trie_node_new(lbl);
      if (!child)
        return -1;
      uint32_t ins = (uint32_t)lo;
      memmove(&cur->children[ins + 1], &cur->children[ins],
              (cur->n_children - ins) * sizeof *cur->children);
      cur->children[ins] = child;
      cur->n_children++;
      cur = child;
    }
  }
  cur->token_id = tok_id + 1;
  // fprintf(stderr, "[DEBUG-TRIE-INS] ID=%u sequence=[", tok_id);
  // for (uint32_t j = 0; j < n; j++) fprintf(stderr, "%u%s", syls[j], (j == n -
  // 1) ? "" : ", "); fprintf(stderr, "]\n");
  return 0;
}

static void trie_free(PtrTrieNode *n) {
  if (!n)
    return;
  for (uint32_t i = 0; i < n->n_children; i++)
    trie_free(n->children[i]);
  free(n->children);
  free(n);
}

static void trie_node_add_child(PtrTrieNode *parent, PtrTrieNode *child) {
  if (parent->n_children == parent->cap_children) {
    uint32_t nc = parent->cap_children ? parent->cap_children * 2 : 4;
    PtrTrieNode **arr = realloc(parent->children, nc * sizeof *arr);
    if (!arr) return;
    parent->children = arr;
    parent->cap_children = nc;
  }
  
  /* Keep children sorted by label for binary search */
  int insert_pos = 0;
  while (insert_pos < (int)parent->n_children && parent->children[insert_pos]->label < child->label) {
    insert_pos++;
  }
  
  for (int m = (int)parent->n_children - 1; m >= insert_pos; m--) {
    parent->children[m + 1] = parent->children[m];
  }
  parent->children[insert_pos] = child;
  parent->n_children++;
}

/* =========================================================
 *  CSR Trie Conversion (pointer trie → compact arrays)
 * ========================================================= */

/* Count total nodes and edges in the trie */
static void trie_count_nodes_edges(PtrTrieNode *n, uint32_t *n_nodes,
                                   uint32_t *n_edges) {
  if (!n)
    return;
  (*n_nodes)++;
  (*n_edges) += n->n_children;
  for (uint32_t i = 0; i < n->n_children; i++) {
    if (n->children[i])
      trie_count_nodes_edges(n->children[i], n_nodes, n_edges);
  }
}

/* BFS-assign node IDs and build Split CSR arrays */
static int trie_build_csr(PtrTrieNode *root, uint32_t max_nodes, uint32_t max_edges,
                           uint32_t *row_ptr, uint16_t *labels, 
                           uint32_t *next_node, uint32_t *terminals,
                           uint32_t *node_count_out, uint32_t *edge_count_out) {
  typedef struct {
    PtrTrieNode *ptr;
    uint32_t id;
  } QueueItem;
  QueueItem *queue = malloc(max_nodes * sizeof *queue);
  if (!queue) return -1;

  uint32_t q_head = 0, q_tail = 0;
  uint32_t next_node_id = 0;
  uint32_t next_edge_idx = 0;

  /* Enqueue root */
  queue[q_tail++] = (QueueItem){root, next_node_id++};
  row_ptr[0] = 0;

  while (q_head < q_tail) {
    QueueItem item = queue[q_head++];
    PtrTrieNode *pn = item.ptr;
    uint32_t node_id = item.id;

    terminals[node_id] = pn->token_id;

    /* Add edges and enqueue children */
    for (uint32_t i = 0; i < pn->n_children; i++) {
      if (!pn->children[i]) continue;
      
      if (next_edge_idx >= max_edges) {
          fprintf(stderr, "[trie_build_csr] edge overflow\n");
          free(queue);
          return -1;
      }
      
      labels[next_edge_idx] = pn->children[i]->label;
      next_node[next_edge_idx] = next_node_id;
      
      if (q_tail >= max_nodes) {
          fprintf(stderr, "[trie_build_csr] node overflow\n");
          free(queue);
          return -1;
      }
      
      queue[q_tail++] = (QueueItem){pn->children[i], next_node_id++};
      next_edge_idx++;
    }
    row_ptr[node_id + 1] = next_edge_idx;
  }

  free(queue);
  *node_count_out = next_node_id;
  *edge_count_out = next_edge_idx;
  return 0;
}

/* =========================================================
 *  Collect token entries from pointer trie (DFS)
 * ========================================================= */

static int collect_entries(PtrTrieNode *node, uint16_t *path, uint8_t depth,
                           TokenEntry *entries, uint32_t *count, uint32_t cap) {
  if (node->token_id != 0 && depth > 0) {
    if (*count >= cap)
      return -1;
    TokenEntry *e = &entries[(*count)++];
    memcpy(e->key, path, depth * sizeof *path);
    e->klen = depth;
    e->id = node->token_id - 1;
  }
  for (uint32_t i = 0; i < node->n_children; i++) {
    if (depth >= MAX_SEQ_LEN)
      continue;
    path[depth] = node->children[i]->label;
    if (collect_entries(node->children[i], path, depth + 1, entries, count,
                        cap) != 0)
      return -1;
  }
  return 0;
}

/* =========================================================
 *  LOUDS build
 *
 *  Builds the flat working table from a token vocabulary.
 *  The "LOUDS bitvector" layer is not needed at runtime —
 *  the flat sorted table + first-syllable index is the hot path.
 * ========================================================= */

LOUDS *louds_build(const char **tokens, const uint32_t *token_ids,
                   uint32_t n_tokens, const SyllableTable *stbl,
                   const Syllabifier *syl_obj) {
  (void)stbl;

  /* Step 1: Build pointer trie */
  PtrTrieNode *root = trie_node_new(0);
  if (!root)
    return NULL;

  /* syllabify() takes a non-const Syllabifier* because it may call
   * stbl_intern() when not frozen.  louds_build() is always called before
   * the tokenizer freezes the syllabifier, so the cast is safe.  If that
   * invariant changes, move the freeze() call to AFTER louds_build().  */
  Syllabifier *s = (Syllabifier *)syl_obj;
  uint16_t syls[MAX_SYLLABLES];

  uint32_t skipped_unk = 0;

  for (uint32_t t = 0; t < n_tokens; t++) {
    /* Skip special tokens (ID 0..SPECIAL_TOKENS_COUNT-1) because they are
     * fallbacks/meta-tokens, not searchable text. ID 0 (<unk>) is the fallback
     * token and doesn't need to be in the LOUDS trie. */
    if (token_ids[t] < SPECIAL_TOKENS_COUNT)
      continue;

    int n = syllabify(s, tokens[t], syls, MAX_SYLLABLES);
    if (n <= 0)
      continue;

    /* Unified-admission guard: a syllable ID of 0 (TOK_UNK) in the sequence
     * means the scanner emitted a symbol that was never interned into the
     * SyllableTable.  Inserting such a sequence into the LOUDS trie creates
     * an edge labelled 0 which is indistinguishable from "no edge found" in
     * the binary-search lookup, making this token permanently unreachable.
     *
     * Root cause: stbl was not a superset of the scanner's output domain at
     * the time louds_build() was called.  stbl_seed_orphan_ascii() in
     * tokenizer_build() should prevent this; this check surfaces regressions. */
    bool has_unk = false;
    for (int si = 0; si < n; si++) {
      if (syls[si] == 0) { /* TOK_UNK */
        has_unk = true;
        break;
      }
    }
    if (has_unk) {
      fprintf(stderr,
              "[louds_build] WARNING: skipping token \"%s\" (id=%u) — "
              "syllabification contains TOK_UNK; fix: ensure stbl covers all "
              "characters before louds_build() is called.\n",
              tokens[t], token_ids[t]);
      skipped_unk++;
      continue;
    }

    if (trie_insert(root, syls, (uint32_t)n, token_ids[t]) != 0) {
      trie_free(root);
      return NULL;
    }
  }

  if (skipped_unk > 0) {
    fprintf(stderr,
            "[louds_build] %u token(s) skipped due to TOK_UNK in syllable "
            "sequence — vocabulary is incomplete.\n", skipped_unk);
  }

  /* Step 2: Collect all token entries into a sorted flat array.
   *
   * entry_cap = n_tokens + 64 is always sufficient because:
   *   - trie_insert sets token_id on exactly one node per inserted token.
   *   - collect_entries records only nodes where (token_id != 0 && depth>0).
   *   - Duplicate syllable-sequences overwrite the same trie node, so
   *     collected entries ≤ distinct syllable paths ≤ n_tokens.
   * The +64 is conservative slack.  If the invariant is ever violated
   * by a future change to the trie logic, the hard error below ensures
   * the failure is visible immediately rather than silently truncating
   * the vocabulary and producing a tokenizer that misses tokens with no
   * observable error at call sites.                                     */
  uint32_t entry_cap = n_tokens + 64;
  TokenEntry *entries = calloc(entry_cap, sizeof *entries);
  if (!entries) {
    trie_free(root);
    return NULL;
  }

  uint32_t n_entries = 0;
  uint16_t path[MAX_SEQ_LEN];
  if (collect_entries(root, path, 0, entries, &n_entries, entry_cap) != 0) {
    /* FIX-4: This path should be unreachable per the invariant above.
     * Treat it as a hard error — a partial vocabulary silently missing
     * tokens is far worse than a clear NULL return to the caller.    */
    fprintf(stderr,
            "[louds_build] Fatal: Trie capacity exceeded (%u nodes). "
            "This indicates a logic error in trie_insert or collect_entries "
            "or the vocabulary is too large for current limits.\n",
            entry_cap);
    trie_free(root);
    free(entries);
    return NULL;
  }

  if (n_entries == 0) {
    trie_free(root);
    free(entries);
    return NULL;
  }

  qsort(entries, n_entries, sizeof *entries, token_entry_cmp);

  /* Step 3: Allocate LOUDS and populate flat table */
  LOUDS *l = calloc(1, sizeof *l);
  if (!l)
    goto oom;

  /* C-02: allocate exactly n_entries, not n_nodes+2 */
  l->klens = calloc(n_entries, sizeof *l->klens);
  l->token_ids = calloc(n_entries, sizeof *l->token_ids);
  l->koffs = calloc(n_entries + 1, sizeof *l->koffs);

  /* Total syllables across all keys */
  uint32_t total_syls = 0;
  for (uint32_t i = 0; i < n_entries; i++)
    total_syls += entries[i].klen;

  l->labels_legacy = calloc(total_syls + 1, sizeof *l->labels_legacy);

  if (!l->klens || !l->token_ids || !l->koffs || !l->labels_legacy)
    goto oom;

  /* Populate */
  uint32_t off = 0;
  for (uint32_t i = 0; i < n_entries; i++) {
    l->klens[i] = entries[i].klen;
    l->token_ids[i] = entries[i].id;
    l->koffs[i] = off;
    memcpy(&l->labels_legacy[off], entries[i].key,
           entries[i].klen * sizeof *l->labels_legacy);
    off += entries[i].klen;
  }
  l->koffs[n_entries] = off;
  l->n_entries = n_entries;

  /* Legacy aliases used by serialisation code */
  l->node_count = total_syls;
  l->label_count = n_entries;
  l->sb_count = n_entries + 1;

  /* rank1_sb aliases koffs — same pointer, same type (uint32_t *).
   * louds_destroy() frees only koffs; rank1_sb must not be freed. */
  l->rank1_sb = l->koffs;

  /* louds_bv is not used at runtime (tokenization uses the flat klens
   * table directly).  Set to NULL so any accidental dereference is
   * caught immediately rather than reading through a type-punned alias. */
  l->louds_bv = NULL;
  l->bv_len = n_entries;

  /* Step 4: Build first-syllable acceleration index (H-01) */
  memset(l->first_index, 0xFF, sizeof l->first_index); /* 0xFF = "no entry" */
  memset(l->first_count, 0, sizeof l->first_count);

  for (uint32_t i = 0; i < n_entries; i++) {
    uint16_t fs = l->labels_legacy[l->koffs[i]];
    if (fs < BASE_SYMBOL_OFFSET) {
      if (l->first_count[fs] == 0) {
        l->first_index[fs] = i;
      } else {
        /* Entries sharing the same first syllable must be contiguous
         * (guaranteed by the qsort above).  If this fires the sort
         * contract was broken or entry_cap truncation scrambled data. */
        if (i != l->first_index[fs] + l->first_count[fs]) {
          fprintf(stderr,
                  "[louds_build] BUG: first-syllable contiguity violated "
                  "at entry %u (syllable %u) — sort contract broken\n",
                  i, fs);
          goto oom;
        }
      }
      l->first_count[fs]++;
    }
  }

  /* Step 5: Build Split CSR trie for fast runtime tokenization */
  uint32_t n_trie_nodes = 0, n_trie_edges = 0;
  trie_count_nodes_edges(root, &n_trie_nodes, &n_trie_edges);

  if (n_trie_nodes == 0) goto oom;

  l->row_ptr   = calloc(n_trie_nodes + 1, sizeof(uint32_t));
  l->labels    = calloc(n_trie_edges > 0 ? n_trie_edges : 1, sizeof(uint16_t));
  l->next_node = calloc(n_trie_edges > 0 ? n_trie_edges : 1, sizeof(uint32_t));
  l->terminals = calloc(n_trie_nodes, sizeof(uint32_t));

  if (!l->row_ptr || !l->labels || !l->next_node || !l->terminals) goto oom;

  l->node_count = n_trie_nodes;
  l->edge_count = n_trie_edges;
  l->has_csr    = true;

  if (trie_build_csr(root, n_trie_nodes, n_trie_edges,
                     l->row_ptr, l->labels, l->next_node, l->terminals,
                     &l->node_count, &l->edge_count) != 0) {
    goto oom;
  }

  /* Step 6: Build fast-path acceleration tables */
  memset(l->fast_token, 0, sizeof l->fast_token);
  memset(l->can_extend, 0, sizeof l->can_extend);
  
  uint32_t start = l->row_ptr[0];
  uint32_t end   = l->row_ptr[1];
  for (uint32_t i = start; i < end; i++) {
    uint16_t lbl  = l->labels[i];
    uint32_t next = l->next_node[i];
    if (lbl < BASE_SYMBOL_OFFSET) {
      if (l->terminals[next] != 0)
        l->fast_token[lbl] = l->terminals[next];
      if (l->row_ptr[next+1] > l->row_ptr[next])
        l->can_extend[lbl] = true;
    }
  }

  fprintf(stderr, "[louds_build] Split CSR: %u nodes, %u edges\n", 
          l->node_count, l->edge_count);
  free(entries);
  trie_free(root);
  return l;

oom:
  free(entries);
  trie_free(root); /* Free pointer trie on error */
  if (l) {
    /* louds_bv is always NULL here (set above); louds_destroy is safe. */
    louds_destroy(l);
  }
  return NULL;
}

/* Rebuild Split CSR trie from flat table (used after loading legacy models) */
void louds_rebuild_csr(LOUDS *l) {
  if (!l || l->n_entries == 0) return;

  /* Free existing CSR arrays */
  if (l->has_csr) {
    free(l->row_ptr);   l->row_ptr   = NULL;
    free(l->labels);    l->labels    = NULL;
    free(l->next_node); l->next_node = NULL;
    free(l->terminals); l->terminals = NULL;
    l->has_csr = false;
  }

  PtrTrieNode *root = trie_node_new(0);
  if (!root) return;

  for (uint32_t i = 0; i < l->n_entries; i++) {
    uint8_t klen = l->klens[i];
    uint32_t koff = l->koffs[i];
    uint32_t tok_id = l->token_ids[i];

    PtrTrieNode *cur = root;
    for (uint8_t k = 0; k < klen; k++) {
      uint16_t lbl = l->labels_legacy[koff + k];
      PtrTrieNode *next = NULL;
      for (uint32_t j = 0; j < cur->n_children; j++) {
        if (cur->children[j]->label == lbl) {
          next = cur->children[j];
          break;
        }
      }
      if (!next) {
        next = trie_node_new(lbl);
        trie_node_add_child(cur, next);
      }
      cur = next;
    }
    cur->token_id = tok_id;
  }

  uint32_t n_nodes = 0, n_edges = 0;
  trie_count_nodes_edges(root, &n_nodes, &n_edges);
  
  l->row_ptr   = calloc(n_nodes + 1, sizeof(uint32_t));
  l->labels    = calloc(n_edges > 0 ? n_edges : 1, sizeof(uint16_t));
  l->next_node = calloc(n_edges > 0 ? n_edges : 1, sizeof(uint32_t));
  l->terminals = calloc(n_nodes, sizeof(uint32_t));
  
  if (l->row_ptr && l->labels && l->next_node && l->terminals) {
    if (trie_build_csr(root, n_nodes, n_edges, 
                       l->row_ptr, l->labels, l->next_node, l->terminals,
                       &l->node_count, &l->edge_count) == 0) {
      l->has_csr = true;
    }
  }

  /* Step 6: Build fast-path tables */
  memset(l->fast_token, 0, sizeof l->fast_token);
  memset(l->can_extend, 0, sizeof l->can_extend);
  if (l->has_csr) {
      uint32_t start = l->row_ptr[0];
      uint32_t end   = l->row_ptr[1];
      for (uint32_t i = start; i < end; i++) {
        uint16_t lbl  = l->labels[i];
        uint32_t next = l->next_node[i];
        if (lbl < BASE_SYMBOL_OFFSET) {
          if (l->terminals[next] != 0) l->fast_token[lbl] = l->terminals[next];
          if (l->row_ptr[next+1] > l->row_ptr[next]) l->can_extend[lbl] = true;
        }
      }
  }

  trie_free(root);
}


void louds_destroy(LOUDS *l) {
  if (!l)
    return;
  
  /* Layer 1: Legacy flat representation */
  free(l->klens);
  free(l->labels_legacy);
  free(l->token_ids);
  free(l->koffs);

  /* Layer 2: Split CSR arrays */
  free(l->row_ptr);
  free(l->labels);
  free(l->next_node);
  free(l->terminals);

  free(l->louds_bv);

  free(l);
}

/* =========================================================
 *  Tokenization: longest-match with binary search
 *
 *  The flat sorted entry table (l->labels_legacy / l->token_ids / klens)
 *  enables O(k * log V) tokenization where k = average token length
 *  and V = vocabulary size.
 *
 *  For each position `pos` in syls[]:
 *    1. Binary search to find the range of entries whose prefix matches
 *       syls[pos..pos+i].
 *    2. Track the last complete match (entry with klen == i+1 and full match).
 *    3. Emit longest match and advance pos.
 * ========================================================= */

/* =========================================================
 *  Tokenization: longest-match with first-syllable index
 *
 *  For each position `pos`:
 *    1. Look up first_index[syls[pos]] to find the contiguous range of
 *       entries that share the same first syllable.  O(1) lookup.
 *    2. Scan only that range (typically << 1% of vocabulary).
 *    3. Track the longest complete match.
 *
 *  Total cost: O(N * avg_bucket_size * max_token_len)
 *  For a 30k vocabulary over a 1024-syllable alphabet, avg bucket ≈ 30.
 * ========================================================= */

int louds_tokenize(const LOUDS *l, const uint16_t *syls, uint32_t n,
                   uint32_t *out, uint32_t out_cap) {
  if (!l || !syls || !out) return -1;

  uint32_t pos = 0;
  uint32_t out_cnt = 0;

  while (pos < n && out_cnt < out_cap) {
    uint16_t syl_id = syls[pos];

    /* Syllable fast-path */
    uint32_t ft = l->fast_token[syl_id];
    if (__builtin_expect(ft != 0 && !l->can_extend[syl_id], 1)) {
        out[out_cnt++] = ft - 1;
        pos++;
        continue;
    }

    uint32_t node = 0; /* root */
    uint32_t best_id = 0;
    uint32_t best_len = 0;
    uint32_t i = pos;

    while (i < n) {
      uint32_t start = l->row_ptr[node];
      uint32_t end   = l->row_ptr[node + 1];
      uint32_t count = end - start;

      if (count == 0) break;

      uint16_t target = syls[i];
      uint32_t next = 0xFFFFFFFF;

      /* Binary search for target label */
      uint32_t lo = start, hi = end;
      while (lo < hi) {
          uint32_t mid = (lo + hi) >> 1;
          if (l->labels[mid] < target) lo = mid + 1;
          else hi = mid;
      }
      
      if (lo < end && l->labels[lo] == target) {
          next = l->next_node[lo];
      }

      if (next == 0xFFFFFFFF) break;

      node = next;
      i++;

      /* Track last terminal node (longest match) */
      if (l->terminals[node] != 0) {
        best_id = l->terminals[node];
        best_len = i - pos;
      }
    }

    if (best_id == 0) {
      best_len = 1;
      best_id = (uint32_t)syls[pos] + 1;
    }

    out[out_cnt++] = best_id - 1;
    pos += best_len;
  }

  return (int)out_cnt;
}
