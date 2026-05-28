/*
 * louds.c
 *
 * Upgraded Design: Minimized Directed Acyclic Word Graph (DAWG).
 * * To elegantly handle Luganda's intense agglutinative morphology, this implementation
 * completely replaces the standard prefix trie with a minimal DAFSA / DAWG. 
 * Suffix structures and shared morphological extensions are merged bottom-up 
 * during construction using an incremental minimization register.
 *
 * The resulting minimized graph structure is compiled cleanly into the existing 
 * Split Compressed Sparse Row (CSR) format, meaning the runtime hot path tokenization 
 * benefits transparently from the compaction without modification.
 */

#include "tokenizer.h"

#include <assert.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <string.h>
#include <stdatomic.h>
#include <unistd.h>  /* for write() */
#include <endian.h>  /* for htole32/le32toh endian conversion */

/* =========================================================
 * Flat token entry (working lookup table)
 * ========================================================= */

typedef struct {
  uint16_t key[MAX_SEQ_LEN];
  uint8_t klen;
  uint32_t id;
} TokenEntry;

/*
 * Compare two token entries lexicographically by their syllable-ID sequences.
 * Mandated for incremental DAWG minimization.
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
 * DAWG Builder Structures and Register (Build-time only)
 * ========================================================= */

typedef struct DawgNode {
  uint16_t label;            /* Incoming/identifying edge label */
  uint32_t token_id;         /* 0 if non-terminal, token_id + 1 if terminal */
  struct DawgNode **children;
  uint32_t n_children;
  uint32_t cap_children;
  uint32_t id;               /* Assigned during CSR layout mapping */
  uint32_t last_visit;       /* Visited pass-ID tracking to handle graph sharing safely */
} DawgNode;

#define REGISTER_SIZE 131071 /* Large prime for the compile-time state register hash map */

typedef struct RegisterEntry {
  DawgNode *node;
  struct RegisterEntry *next;
} RegisterEntry;

static RegisterEntry *dawg_register[REGISTER_SIZE];
static uint32_t global_visit_id = 0;
static pthread_mutex_t louds_build_mutex = PTHREAD_MUTEX_INITIALIZER;

/* Compute hash of a node based purely on its deterministic future language */
static uint32_t dawg_node_hash(const DawgNode *n) {
  uint32_t hash = 5381;
  hash = ((hash << 5) + hash) + n->token_id;
  hash = ((hash << 5) + hash) + n->n_children;
  for (uint32_t i = 0; i < n->n_children; i++) {
    hash = ((hash << 5) + hash) + n->children[i]->label;
    uintptr_t ptr_val = (uintptr_t)n->children[i];
    hash = ((hash << 5) + hash) + (uint32_t)(ptr_val & 0xFFFFFFFF);
    if (sizeof(uintptr_t) > 4) {
      hash = ((hash << 5) + hash) + (uint32_t)(ptr_val >> 32);
    }
  }
  return hash % REGISTER_SIZE;
}

/* Structural equivalence check for bottom-up state minimization */
static bool dawg_node_equals(const DawgNode *a, const DawgNode *b) {
  if (a->token_id != b->token_id) return false;
  if (a->n_children != b->n_children) return false;
  for (uint32_t i = 0; i < a->n_children; i++) {
    if (a->children[i]->label != b->children[i]->label) return false;
    if (a->children[i] != b->children[i]) return false; /* Enforce identity of already canonicalized children */
  }
  return true;
}

static void dawg_node_add_child(DawgNode *parent, DawgNode *child) {
  if (parent->n_children == parent->cap_children) {
    uint32_t nc = parent->cap_children ? parent->cap_children * 2 : 4;
    DawgNode **arr = realloc(parent->children, nc * sizeof *arr);
    if (!arr) return;
    parent->children = arr;
    parent->cap_children = nc;
  }
  /* Sorted insertion is naturally guaranteed since incoming strings are pre-sorted lexicographically */
  parent->children[parent->n_children++] = child;
}

/* Performs bottom-up minimization on parts of the graph no longer in the frontier */
static void minimize_node(DawgNode *parent, uint32_t child_idx) {
  DawgNode *child = parent->children[child_idx];
  uint32_t h = dawg_node_hash(child);
  RegisterEntry *curr = dawg_register[h];
  
  while (curr) {
    if (dawg_node_equals(curr->node, child)) {
      /* Equivalent canonical state found! Redirect parent edge, purge redundant allocation */
      parent->children[child_idx] = curr->node;
      free(child->children);
      free(child);
      return;
    }
    curr = curr->next;
  }
  
  /* State is unique in the current graph prefix; register it */
  RegisterEntry *entry = malloc(sizeof *entry);
  if (entry) {
    entry->node = child;
    entry->next = dawg_register[h];
    dawg_register[h] = entry;
  }
}

static void dawg_clear_register(void) {
  for (int i = 0; i < REGISTER_SIZE; i++) {
    RegisterEntry *curr = dawg_register[i];
    while (curr) {
      RegisterEntry *next = curr->next;
      free(curr);
      curr = next;
    }
    dawg_register[i] = NULL;
  }
}

/* =========================================================
 * Graph-Safe Visual/CSR Navigation Conversions
 * ========================================================= */

/* Graph traversal counting unique nodes and edges using visit IDs */
static void dawg_count_nodes_edges(DawgNode *n, uint32_t visit_id, uint32_t *n_nodes, uint32_t *n_edges) {
  if (!n || n->last_visit == visit_id) return;
  n->last_visit = visit_id;
  (*n_nodes)++;
  (*n_edges) += n->n_children;
  for (uint32_t i = 0; i < n->n_children; i++) {
    dawg_count_nodes_edges(n->children[i], visit_id, n_nodes, n_edges);
  }
}

/* BFS-assign node IDs and flatten DAWG topology into Split CSR storage arrays */
static int dawg_build_csr(DawgNode *root, uint32_t visit_id, uint32_t max_nodes, uint32_t max_edges,
                          uint32_t *row_ptr, uint16_t *labels, 
                          uint32_t *next_node, uint32_t *terminals,
                          uint32_t *node_count_out, uint32_t *edge_count_out) {
  typedef struct {
    DawgNode *ptr;
    uint32_t id;
  } QueueItem;
  QueueItem *queue = malloc(max_nodes * sizeof *queue);
  if (!queue) return -1;

  uint32_t q_head = 0, q_tail = 0;
  uint32_t next_node_id = 0;
  uint32_t next_edge_idx = 0;

  root->id = next_node_id++;
  root->last_visit = visit_id;
  queue[q_tail++] = (QueueItem){root, root->id};
  row_ptr[0] = 0;

  while (q_head < q_tail) {
    QueueItem item = queue[q_head++];
    DawgNode *pn = item.ptr;
    uint32_t node_id = item.id;

    terminals[node_id] = pn->token_id;

    for (uint32_t i = 0; i < pn->n_children; i++) {
      DawgNode *child = pn->children[i];
      if (!child) continue;
      
      if (next_edge_idx >= max_edges) {
        fprintf(stderr, "[dawg_build_csr] edge overflow\n");
        free(queue);
        return -1;
      }
      
      labels[next_edge_idx] = child->label;
      
      /* Enqueue state only if it hasn't been encountered yet during this layout pass */
      if (child->last_visit != visit_id) {
        if (next_node_id >= max_nodes) {
          fprintf(stderr, "[dawg_build_csr] node overflow\n");
          free(queue);
          return -1;
        }
        child->id = next_node_id++;
        child->last_visit = visit_id;
        queue[q_tail++] = (QueueItem){child, child->id};
      }
      
      next_node[next_edge_idx] = child->id;
      next_edge_idx++;
    }
    row_ptr[node_id + 1] = next_edge_idx;
  }

  free(queue);
  *node_count_out = next_node_id;
  *edge_count_out = next_edge_idx;
  return 0;
}

static void dawg_collect_nodes(DawgNode *n, uint32_t visit_id, DawgNode **list, uint32_t *count) {
  if (!n || n->last_visit == visit_id) return;
  n->last_visit = visit_id;
  list[(*count)++] = n;
  for (uint32_t i = 0; i < n->n_children; i++) {
    dawg_collect_nodes(n->children[i], visit_id, list, count);
  }
}

/* Linear graph cleanup preventing double frees over converged suffix structures */
static void dawg_free_graph(DawgNode *root, uint32_t visit_id, uint32_t max_nodes) {
  DawgNode **list = malloc(max_nodes * sizeof *list);
  if (!list) return;
  uint32_t count = 0;
  dawg_collect_nodes(root, visit_id, list, &count);
  for (uint32_t i = 0; i < count; i++) {
    free(list[i]->children);
    free(list[i]);
  }
  free(list);
}

/* =========================================================
 * DAWG Build and Serialization Interleaving
 * ========================================================= */

LOUDS *louds_build(const char **tokens, const uint32_t *token_ids,
                   uint32_t n_tokens, const SyllableTable *stbl,
                   const Syllabifier *syl_obj) {
  pthread_mutex_lock(&louds_build_mutex);

  (void)stbl;
  Syllabifier *s = (Syllabifier *)syl_obj;
  uint16_t syls[MAX_SYLLABLES];
  uint32_t skipped_unk = 0;

  uint32_t entry_cap = n_tokens + 64;
  TokenEntry *entries = calloc(entry_cap, sizeof *entries);
  if (!entries) { pthread_mutex_unlock(&louds_build_mutex); return NULL; }

  uint32_t n_entries = 0;

  /* Step 1: Pre-collect and ingest sequences */
  for (uint32_t t = 0; t < n_tokens; t++) {
    if (token_ids[t] < SPECIAL_TOKENS_COUNT)
      continue;

    int n = syllabify(s, tokens[t], syls, MAX_SYLLABLES);
    if (n <= 0)
      continue;

    bool has_unk = false;
    for (int si = 0; si < n; si++) {
      if (syls[si] == 0) {
        has_unk = true;
        break;
      }
    }
    if (has_unk) {
      fprintf(stderr, "[louds_build] WARNING: skipping token \"%s\" (id=%u) — contains TOK_UNK.\n", tokens[t], token_ids[t]);
      skipped_unk++;
      continue;
    }

    if (n_entries >= entry_cap) {
      entry_cap *= 2;
      TokenEntry *new_entries = realloc(entries, (size_t)entry_cap * sizeof *new_entries);
      if (!new_entries) { free(entries); pthread_mutex_unlock(&louds_build_mutex); return NULL; }
      entries = new_entries;
    }

    TokenEntry *e = &entries[n_entries++];
    memcpy(e->key, syls, n * sizeof(uint16_t));
    e->klen = n;
    e->id = token_ids[t];
  }

  if (n_entries == 0) {
    free(entries);
    pthread_mutex_unlock(&louds_build_mutex);
    return NULL;
  }

  /* Step 2: Sort strings lexicographically to fulfill Daciuk's invariant */
  qsort(entries, n_entries, sizeof *entries, token_entry_cmp);

  /* Step 3: Incremental DAWG Construction Loop */
  memset(dawg_register, 0, sizeof(dawg_register));
  DawgNode *root = calloc(1, sizeof *root);
  if (!root) { free(entries); pthread_mutex_unlock(&louds_build_mutex); return NULL; }

  DawgNode *frontier[MAX_SEQ_LEN + 1];
  frontier[0] = root;
  uint32_t prev_len = 0;

  for (uint32_t i = 0; i < n_entries; i++) {
    TokenEntry *e = &entries[i];
    uint32_t len = e->klen;
    uint32_t common_len = 0;

    if (i > 0) {
      TokenEntry *prev_e = &entries[i - 1];
      uint32_t min_l = len < prev_e->klen ? len : prev_e->klen;
      while (common_len < min_l && e->key[common_len] == prev_e->key[common_len]) {
        common_len++;
      }
      if (common_len == len && len == prev_e->klen) {
        continue; /* De-duplicate identical sequence entries safely */
      }
    }

    /* Minimize from the bottom up starting from the rightmost edge down to the divergence point */
    for (uint32_t d = prev_len; d > common_len; d--) {
      minimize_node(frontier[d - 1], frontier[d - 1]->n_children - 1);
    }

    /* Spawn new unminimized frontier segments for the current suffix */
    for (uint32_t d = common_len + 1; d <= len; d++) {
      DawgNode *child = calloc(1, sizeof *child);
      child->label = e->key[d - 1];
      dawg_node_add_child(frontier[d - 1], child);
      frontier[d] = child;
    }

    frontier[len]->token_id = e->id + 1;
    prev_len = len;
  }

  /* Complete minimization pass over the residual right edge of the graph */
  for (uint32_t d = prev_len; d > 0; d--) {
    minimize_node(frontier[d - 1], frontier[d - 1]->n_children - 1);
  }

  dawg_clear_register();

  /* Compute topological statistics */
  uint32_t visit_id = ++global_visit_id;
  uint32_t n_dawg_nodes = 0, n_dawg_edges = 0;
  dawg_count_nodes_edges(root, visit_id, &n_dawg_nodes, &n_dawg_edges);

  /* Step 4: Instantiation of runtime LOUDS wrapping container */
  LOUDS *l = calloc(1, sizeof *l);
  if (!l) goto oom;

  l->klens = calloc(n_entries, sizeof *l->klens);
  l->token_ids = calloc(n_entries, sizeof *l->token_ids);
  l->koffs = calloc(n_entries + 1, sizeof *l->koffs);

  uint32_t total_syls = 0;
  for (uint32_t i = 0; i < n_entries; i++)
    total_syls += entries[i].klen;

  l->labels_legacy = calloc(total_syls + 1, sizeof *l->labels_legacy);

  if (!l->klens || !l->token_ids || !l->koffs || !l->labels_legacy)
    goto oom;

  uint32_t off = 0;
  for (uint32_t i = 0; i < n_entries; i++) {
    l->klens[i] = entries[i].klen;
    l->token_ids[i] = entries[i].id;
    l->koffs[i] = off;
    memcpy(&l->labels_legacy[off], entries[i].key, entries[i].klen * sizeof *l->labels_legacy);
    off += entries[i].klen;
  }
  l->koffs[n_entries] = off;
  l->n_entries = n_entries;

  l->legacy_label_count = total_syls;
  l->label_count = n_entries;
  l->sb_count = n_entries + 1;
  l->rank1_sb = l->koffs;
  l->louds_bv = NULL;
  l->bv_len = n_entries;

  /* Step 5: Build first-syllable acceleration index */
  memset(l->first_index, 0xFF, sizeof l->first_index);
  memset(l->first_count, 0, sizeof l->first_count);

  for (uint32_t i = 0; i < n_entries; i++) {
    uint16_t fs = l->labels_legacy[l->koffs[i]];
    if (fs < BASE_SYMBOL_OFFSET) {
      if (l->first_count[fs] == 0) {
        l->first_index[fs] = i;
      }
      l->first_count[fs]++;
    }
  }

  /* Step 6: Map minimized DAWG into runtime CSR Layout arrays */
  l->row_ptr   = calloc(n_dawg_nodes + 1, sizeof(uint32_t));
  l->labels    = calloc(n_dawg_edges > 0 ? n_dawg_edges : 1, sizeof(uint16_t));
  l->next_node = calloc(n_dawg_edges > 0 ? n_dawg_edges : 1, sizeof(uint32_t));
  l->terminals = calloc(n_dawg_nodes, sizeof(uint32_t));

  if (!l->row_ptr || !l->labels || !l->next_node || !l->terminals) goto oom;

  l->node_count = n_dawg_nodes;
  l->edge_count = n_dawg_edges;
  l->has_csr    = true;

  visit_id = ++global_visit_id;
  if (dawg_build_csr(root, visit_id, n_dawg_nodes, n_dawg_edges,
                     l->row_ptr, l->labels, l->next_node, l->terminals,
                     &l->node_count, &l->edge_count) != 0) {
    goto oom;
  }

  /* Step 7: Fast-path initialization */
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

  fprintf(stderr, "[louds_build] Minimized DAWG CSR Architecture: %u nodes, %u edges\n",
          l->node_count, l->edge_count);
  free(entries);
  visit_id = ++global_visit_id;
  dawg_free_graph(root, visit_id, n_dawg_nodes + 16);
  pthread_mutex_unlock(&louds_build_mutex);
  return l;

oom:
  free(entries);
  visit_id = ++global_visit_id;
  dawg_free_graph(root, visit_id, n_dawg_nodes + 16);
  if (l) louds_destroy(l);
  pthread_mutex_unlock(&louds_build_mutex);
  return NULL;
}

/* * Rebuild Split CSR array representation from legacy models.
 * Upgraded to inject suffix minimization automatically upon loading.
 */
void louds_rebuild_csr(LOUDS *l) {
  if (!l || l->n_entries == 0) return;

  pthread_mutex_lock(&louds_build_mutex);

  if (l->has_csr) {
    free(l->row_ptr);   l->row_ptr   = NULL;
    free(l->labels);    l->labels    = NULL;
    free(l->next_node); l->next_node = NULL;
    free(l->terminals); l->terminals = NULL;
    l->has_csr = false;
  }

  memset(dawg_register, 0, sizeof(dawg_register));
  DawgNode *root = calloc(1, sizeof *root);
  if (!root) { pthread_mutex_unlock(&louds_build_mutex); return; }

  DawgNode *frontier[MAX_SEQ_LEN + 1];
  frontier[0] = root;
  uint32_t prev_len = 0;

  for (uint32_t i = 0; i < l->n_entries; i++) {
    uint8_t klen = l->klens[i];
    uint32_t koff = l->koffs[i];
    uint32_t tok_id = l->token_ids[i];
    uint32_t common_len = 0;

    if (i > 0) {
      uint8_t prev_klen = l->klens[i - 1];
      uint32_t prev_koff = l->koffs[i - 1];
      uint32_t min_l = klen < prev_klen ? klen : prev_klen;
      while (common_len < min_l && 
             l->labels_legacy[koff + common_len] == l->labels_legacy[prev_koff + common_len]) {
        common_len++;
      }
      if (common_len == klen && klen == prev_klen) continue;
    }

    for (uint32_t d = prev_len; d > common_len; d--) {
      minimize_node(frontier[d - 1], frontier[d - 1]->n_children - 1);
    }

    for (uint32_t d = common_len + 1; d <= klen; d++) {
      DawgNode *child = calloc(1, sizeof *child);
      child->label = l->labels_legacy[koff + d - 1];
      dawg_node_add_child(frontier[d - 1], child);
      frontier[d] = child;
    }

    frontier[klen]->token_id = tok_id + 1;
    prev_len = klen;
  }

  for (uint32_t d = prev_len; d > 0; d--) {
    minimize_node(frontier[d - 1], frontier[d - 1]->n_children - 1);
  }

  dawg_clear_register();

  uint32_t visit_id = ++global_visit_id;
  uint32_t n_nodes = 0, n_edges = 0;
  dawg_count_nodes_edges(root, visit_id, &n_nodes, &n_edges);
  
  l->row_ptr   = calloc(n_nodes + 1, sizeof(uint32_t));
  l->labels    = calloc(n_edges > 0 ? n_edges : 1, sizeof(uint16_t));
  l->next_node = calloc(n_edges > 0 ? n_edges : 1, sizeof(uint32_t));
  l->terminals = calloc(n_nodes, sizeof(uint32_t));

  if (l->row_ptr && l->labels && l->next_node && l->terminals) {
    visit_id = ++global_visit_id;
    if (dawg_build_csr(root, visit_id, n_nodes, n_edges,
                       l->row_ptr, l->labels, l->next_node, l->terminals,
                       &l->node_count, &l->edge_count) == 0) {
      l->has_csr = true;
    }
  } else {
    /* Free any successful allocations if combined check fails */
    free(l->row_ptr);
    free(l->labels);
    free(l->next_node);
    free(l->terminals);
    l->row_ptr = NULL;
    l->labels = NULL;
    l->next_node = NULL;
    l->terminals = NULL;
  }

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

  visit_id = ++global_visit_id;
  dawg_free_graph(root, visit_id, n_nodes + 16);

  pthread_mutex_unlock(&louds_build_mutex);
}

void louds_destroy(LOUDS *l) {
  if (!l) return;
  
  if (!l->is_mmap_backed) {
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
  }

  free(l->louds_bv);
  free(l);
}

/* =========================================================
 * Tokenization Hot Path Navigation Loop
 * ========================================================= */

int louds_tokenize(const LOUDS *l, const uint16_t *syls, uint32_t n,
                   uint32_t *out, uint32_t out_cap) {
  if (!l || !syls || !out) return -1;

  uint32_t pos = 0;
  uint32_t out_cnt = 0;

  while (pos < n && out_cnt < out_cap) {
    uint16_t syl_id = syls[pos];

    /* Syllable fast-path shortcuts */
    uint32_t ft = l->fast_token[syl_id];
    if (__builtin_expect(ft != 0 && !l->can_extend[syl_id], 1)) {
      out[out_cnt++] = ft - 1;
      pos++;
      continue;
    }

    uint32_t node = 0; /* Index representing root */
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

      /* Binary search across outbound transitions */
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

      /* Track terminal status across consolidated nodes */
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

  /* H-3: signal truncation — caller cannot distinguish a full-buffer success
   * from a genuinely-complete tokenization without this check. */
  if (out_cnt == out_cap && pos < n) {
    return -1;   /* output buffer exhausted before input was consumed */
  }
  return (int)out_cnt;
}