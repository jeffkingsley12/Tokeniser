#include "truth_layer.h"
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
  const char *text;
  uint32_t category;
} TruthSeed;

/* NOTE: Duplicate prefix entries (e.g. "mu"/TRUTH_PREFIX appears for
 * classes 1, 3, and 18) are intentional — they represent distinct
 * morphological noun classes that happen to share surface forms.
 * The trie de-duplicates them at insertion time; the hash includes all
 * entries so the fingerprint covers the full seed definition. */
static const TruthSeed TRUTH_SEED_DATA[] = {
    /* Class 1 (person singular) */
    {"mu", TRUTH_PREFIX},
    {"omu", TRUTH_PREFIX},

    /* Class 2 (person plural) */
    {"ba", TRUTH_PREFIX},
    {"aba", TRUTH_PREFIX},

    /* Class 3 (tree singular) */
    {"mu", TRUTH_PREFIX},
    {"omu", TRUTH_PREFIX},

    /* Class 4 (tree plural) */
    {"mi", TRUTH_PREFIX},
    {"emi", TRUTH_PREFIX},

    /* Class 5 (fruit singular) */
    {"li", TRUTH_PREFIX},
    {"eri", TRUTH_PREFIX},

    /* Class 6 (fruit plural) */
    {"ma", TRUTH_PREFIX},
    {"ama", TRUTH_PREFIX},

    /* Class 7 (thing singular) */
    {"ki", TRUTH_PREFIX},
    {"eki", TRUTH_PREFIX},

    /* Class 8 (thing plural) */
    {"bi", TRUTH_PREFIX},
    {"ebi", TRUTH_PREFIX},

    /* Class 9 (animal singular) */
    {"n", TRUTH_PREFIX},
    {"en", TRUTH_PREFIX},

    /* Class 10 (animal plural) */
    {"zi", TRUTH_PREFIX},
    {"enzi", TRUTH_PREFIX},

    /* Class 11 (long thing) */
    {"lu", TRUTH_PREFIX},
    {"olu", TRUTH_PREFIX},

    /* Class 12 (diminutive) */
    {"ka", TRUTH_PREFIX},
    {"aka", TRUTH_PREFIX},

    /* Class 13 (diminutive plural) */
    {"tu", TRUTH_PREFIX},
    {"otu", TRUTH_PREFIX},

    /* Class 14 (abstract) */
    {"bu", TRUTH_PREFIX},
    {"obu", TRUTH_PREFIX},

    /* Class 15 (infinitive) */
    {"ku", TRUTH_PREFIX},
    {"oku", TRUTH_PREFIX},

    /* Class 16 (locative) */
    {"wa", TRUTH_PREFIX},
    {"awa", TRUTH_PREFIX},

    /* Class 17 (locative) */
    {"ku", TRUTH_PREFIX},
    {"oku", TRUTH_PREFIX},

    /* Class 18 (locative) */
    {"mu", TRUTH_PREFIX},
    {"omu", TRUTH_PREFIX},

    /* Class 20 (augmentative) */
    {"gu", TRUTH_PREFIX},
    {"ogu", TRUTH_PREFIX},

    /* Class 22 (plural augmentative) */
    {"ga", TRUTH_PREFIX},
    {"aga", TRUTH_PREFIX},

    /* Particles */
    {"ne", TRUTH_PARTICLE},
    {"na", TRUTH_PARTICLE},
    {"ye", TRUTH_PARTICLE},
    {"e", TRUTH_PARTICLE},

    /* === COMMON ROOTS (stems) === */
    {"lim", TRUTH_ROOT},
    {"som", TRUTH_ROOT},
    {"lya", TRUTH_ROOT},
    {"nya", TRUTH_ROOT},
    {"gul", TRUTH_ROOT},
    {"tun", TRUTH_ROOT},
    {"lab", TRUTH_ROOT},
    {"kub", TRUTH_ROOT},
    {"zib", TRUTH_ROOT},
    {"vub", TRUTH_ROOT},
    {"tamb", TRUTH_ROOT},
    {"sinz", TRUTH_ROOT},
    {"kola", TRUTH_ROOT},
    {"yimb", TRUTH_ROOT},
    {"zany", TRUTH_ROOT},

    /* === SUFFIXES === */
    {"vu", TRUTH_SUFFIX},
    {"fu", TRUTH_SUFFIX},
    {"ye", TRUTH_SUFFIX},
};

static const size_t N_SEEDS =
    sizeof(TRUTH_SEED_DATA) / sizeof(TRUTH_SEED_DATA[0]);

/* ============================================================================
 * Internal resize helpers
 * ============================================================================
 */

/* Initial capacities — chosen to avoid early reallocation for the current
 * seed set while remaining small enough for low-memory targets. */
#define TRIE_INITIAL_NODES 1024
#define TRIE_INITIAL_EDGES 2048

/* Grow tt->nodes to at least `need` entries.  Returns 0 on success. */
static int grow_nodes(TruthTrie *tt, uint32_t need) {
  uint32_t new_cap = tt->node_cap ? tt->node_cap : TRIE_INITIAL_NODES;
  while (new_cap < need) {
    if (new_cap > UINT32_MAX / 2)
      return -1;
    new_cap *= 2;
  }
  SylTrieNode *p = realloc(tt->nodes, new_cap * sizeof(SylTrieNode));
  if (!p)
    return -1;
  memset(p + tt->node_cap, 0, (new_cap - tt->node_cap) * sizeof(SylTrieNode));

  uint16_t *d = realloc(tt->depths, new_cap * sizeof(uint16_t));
  if (!d) {
    /* Roll back nodes allocation to maintain consistency with depths */
    SylTrieNode *rollback = realloc(p, tt->node_cap * sizeof(SylTrieNode));
    if (rollback) tt->nodes = rollback;
    else tt->nodes = p; /* Keep p if rollback fails, but cap stays old */
    return -1;
  }
  memset(d + tt->node_cap, 0, (new_cap - tt->node_cap) * sizeof(uint16_t));
  
  tt->nodes = p;
  tt->depths = d;
  tt->node_cap = new_cap;
  return 0;
}

/* Grow tt->edges to at least `need` entries.  Returns 0 on success. */
static int grow_edges(TruthTrie *tt, uint32_t need) {
  uint32_t new_cap = tt->edge_cap ? tt->edge_cap : TRIE_INITIAL_EDGES;
  while (new_cap < need) {
    if (new_cap > UINT32_MAX / 2)
      return -1;
    new_cap *= 2;
  }
  SylTrieEdge *p = realloc(tt->edges, new_cap * sizeof(SylTrieEdge));
  if (!p)
    return -1;
  memset(p + tt->edge_cap, 0, (new_cap - tt->edge_cap) * sizeof(SylTrieEdge));
  tt->edges = p;
  tt->edge_cap = new_cap;
  return 0;
}

/* ============================================================================
 * TruthTrie Implementation (Syllable-ID based AC Automaton)
 * ============================================================================
 */

TruthTrie *truth_trie_create(void) {
  TruthTrie *tt = calloc(1, sizeof *tt);
  if (!tt)
    return NULL;

  tt->nodes = calloc(TRIE_INITIAL_NODES, sizeof(SylTrieNode));
  if (!tt->nodes) {
    free(tt);
    return NULL;
  }
  tt->node_count = 1; /* root = node 0 */
  tt->node_cap = TRIE_INITIAL_NODES;

  tt->edges = calloc(TRIE_INITIAL_EDGES, sizeof(SylTrieEdge));
  if (!tt->edges) {
    free(tt->nodes);
    free(tt);
    return NULL;
  }
  tt->edge_count = 0;
  tt->edge_cap = TRIE_INITIAL_EDGES;

  memset(tt->first_syl, 0xFF, sizeof(tt->first_syl));

  tt->depths = calloc(TRIE_INITIAL_NODES, sizeof(uint16_t));
  if (!tt->depths) {
    free(tt->edges);
    free(tt->nodes);
    free(tt);
    return NULL;
  }
  tt->depths[0] = 0; /* root depth */

  return tt;
}

void truth_trie_destroy(TruthTrie *tt) {
  if (!tt)
    return;
  if (tt->nodes)
    free(tt->nodes);
  if (tt->edges)
    free(tt->edges);
  if (tt->entries)
    free(tt->entries);
  if (tt->failure_links)
    free(tt->failure_links);
  if (tt->depths)
    free(tt->depths);
  free(tt);
}

/* Internal trie builder (operates on syllable IDs).
 * Returns 0 on success, -1 on allocation failure. */
int truth_trie_insert(TruthTrie *tt, const uint16_t *sids, size_t len,
                      uint32_t category) {
  if (!tt || !sids || len == 0)
    return 0;

  uint32_t node = 0;
  for (size_t i = 0; i < len; i++) {
    uint16_t sid = sids[i];
    uint32_t next = UINT32_MAX;

    /* Fast path: root-level lookup via first_syl table */
    if (node == 0 && sid < BASE_SYMBOL_OFFSET) {
      next = tt->first_syl[sid];
    }

    /* General path: linear scan of the current node's edges.
     * (Edge count is small for this trie — linear scan is fine.) */
    if (next == UINT32_MAX) {
      SylTrieNode *nd = &tt->nodes[node];
      for (uint16_t j = 0; j < nd->edge_count; j++) {
        if (tt->edges[nd->first_edge + j].label == sid) {
          next = tt->edges[nd->first_edge + j].next_node;
          break;
        }
      }
    }

    if (next == UINT32_MAX) {
      /* Allocate a new node */
      if (tt->node_count >= tt->node_cap) {
        if (grow_nodes(tt, tt->node_count + 1) != 0)
          return -1;
      }
      next = tt->node_count++;
      tt->depths[next] = tt->depths[node] + 1;

      if (node == 0 && sid < BASE_SYMBOL_OFFSET) {
        /* Register in the O(1) root fast-path table */
        tt->first_syl[sid] = next;
      } else {
        /* Append edge to the current node.
         * Edges for a node must be contiguous — only valid when
         * added sequentially during construction (which is true
         * here since we build depth-first, left-to-right). */
        SylTrieNode *nd = &tt->nodes[node];
        if (nd->edge_count == 0) {
          /* First edge for this node: claim the next free slot */
          nd->first_edge = tt->edge_count;
        }

        uint32_t insert_pos = nd->first_edge + nd->edge_count;

        if (tt->edge_count >= tt->edge_cap) {
          if (grow_edges(tt, tt->edge_count + 1) != 0)
            return -1;
          /* Re-load nd after realloc (pointer may have moved) */
          nd = &tt->nodes[node];
        }

        if (insert_pos < tt->edge_count) {
          memmove(&tt->edges[insert_pos + 1], &tt->edges[insert_pos],
                  (tt->edge_count - insert_pos) * sizeof(SylTrieEdge));
          for (uint32_t n = 0; n < tt->node_count; n++) {
            if (tt->nodes[n].edge_count > 0 && tt->nodes[n].first_edge >= insert_pos && n != node) {
              tt->nodes[n].first_edge++;
            }
          }
        }

        tt->edges[insert_pos].label = sid;
        tt->edges[insert_pos].next_node = next;
        tt->edge_count++;
        nd->edge_count++;
      }
    }
    node = next;
  }

  /* Mark terminal — encode category in the upper bit of token_id.
   * token_id == 0 means "not terminal", so we set bit 31 unconditionally. */
  tt->nodes[node].token_id = (category & ~0x80000000u) | 0x80000000u;
  return 0;
}

int truth_trie_build_from_seed(TruthTrie *tt, const Tokenizer *tok) {
  if (!tt || !tok)
    return -1;

  for (size_t i = 0; i < N_SEEDS; i++) {
    uint16_t sids[MAX_SEQ_LEN];
    int n = syllabify(tok->syl, TRUTH_SEED_DATA[i].text, sids, MAX_SEQ_LEN);
    if (n > 0) {
      truth_trie_insert(tt, sids, (size_t)n, TRUTH_SEED_DATA[i].category);
    }
  }

  int rc = truth_trie_compute_failure_links(tt);
  if (rc == 0)
    tt->has_csr = true;
  return rc;
}

int truth_trie_compute_failure_links(TruthTrie *tt) {
  if (!tt || tt->node_count == 0)
    return -1;

  tt->failure_links = calloc(tt->node_count, sizeof(uint32_t));
  if (!tt->failure_links)
    return -1;
  uint32_t *queue = malloc(tt->node_count * sizeof(uint32_t));
  if (!queue) {
    free(tt->failure_links);
    tt->failure_links = NULL;
    tt->has_csr = false;
    return -1;
  }

  uint32_t head = 0, tail = 0;

  /* Depth 1 nodes */
  for (uint32_t i = 0; i < BASE_SYMBOL_OFFSET; i++) {
    if (tt->first_syl[i] != UINT32_MAX) {
      queue[tail++] = tt->first_syl[i];
      tt->failure_links[tt->first_syl[i]] = 0;
    }
  }

  while (head < tail) {
    uint32_t u = queue[head++];
    SylTrieNode *nd_u = &tt->nodes[u];

    for (uint16_t i = 0; i < nd_u->edge_count; i++) {
      uint16_t lbl = tt->edges[nd_u->first_edge + i].label;
      uint32_t v = tt->edges[nd_u->first_edge + i].next_node;

      uint32_t f = tt->failure_links[u];
      while (f != 0) {
        /* Search lbl in f's edges */
        uint32_t next_f = UINT32_MAX;
        SylTrieNode *nd_f = &tt->nodes[f];
        for (uint16_t j = 0; j < nd_f->edge_count; j++) {
          if (tt->edges[nd_f->first_edge + j].label == lbl) {
            next_f = tt->edges[nd_f->first_edge + j].next_node;
            break;
          }
        }
        if (next_f != UINT32_MAX) {
          tt->failure_links[v] = next_f;
          break;
        }
        f = tt->failure_links[f];
      }

      if (f == 0) {
        if (lbl < BASE_SYMBOL_OFFSET && tt->first_syl[lbl] != UINT32_MAX) {
          tt->failure_links[v] = tt->first_syl[lbl];
        } else {
          tt->failure_links[v] = 0;
        }
      }

      queue[tail++] = v;
    }
  }

  free(queue);
  return 0;
}

int truth_match_ids(const TruthTrie *tt, const uint16_t *syllable_ids,
                    size_t n_syllables, TruthMatchInfo *out_match) {
  if (!tt || !syllable_ids || n_syllables == 0)
    return 0;

  uint32_t node = 0;
  int best_len = 0;

  for (size_t i = 0; i < n_syllables; i++) {
    uint16_t sid = syllable_ids[i];
    uint32_t next = UINT32_MAX;

    if (node == 0 && sid < BASE_SYMBOL_OFFSET) {
      next = tt->first_syl[sid];
    } else {
      SylTrieNode *nd = &tt->nodes[node];
      for (uint16_t j = 0; j < nd->edge_count; j++) {
        if (tt->edges[nd->first_edge + j].label == sid) {
          next = tt->edges[nd->first_edge + j].next_node;
          break;
        }
      }
    }

    while (next == UINT32_MAX && node != 0) {
      node = tt->failure_links[node];
      if (node == 0 && sid < BASE_SYMBOL_OFFSET) {
        next = tt->first_syl[sid];
      } else {
        SylTrieNode *nd = &tt->nodes[node];
        for (uint16_t j = 0; j < nd->edge_count; j++) {
          if (tt->edges[nd->first_edge + j].label == sid) {
            next = tt->edges[nd->first_edge + j].next_node;
            break;
          }
        }
      }
    }

    if (next == UINT32_MAX) {
      node = 0;
    } else {
      node = next;
    }

    uint32_t out_node = node;
    while (out_node != 0) {
      if (tt->nodes[out_node].token_id != 0) {
        if (out_node == node) {
            best_len = (int)(i + 1);
        }
        if (out_match) {
          out_match->token_id = 0; /* Not used for truth layer usually */
          out_match->category_mask |= tt->nodes[out_node].token_id & ~0x80000000;
        }
      }
      out_node = tt->failure_links[out_node];
    }
  }

  return best_len;
}

int aho_corasick_find_matches(const TruthTrie *tt,
                              const uint16_t  *syllable_ids,
                              size_t           n_syllables,
                              TruthMatch      *out_matches,
                              uint32_t         max_matches) {
  if (!tt || !syllable_ids || n_syllables == 0 || !out_matches || max_matches == 0)
    return 0;

  uint32_t node = 0;
  uint32_t found = 0;

  for (size_t i = 0; i < n_syllables; i++) {
    uint16_t sid = syllable_ids[i];
    uint32_t next = UINT32_MAX;

    if (node == 0 && sid < BASE_SYMBOL_OFFSET) {
      next = tt->first_syl[sid];
    } else {
      SylTrieNode *nd = &tt->nodes[node];
      for (uint16_t j = 0; j < nd->edge_count; j++) {
        if (tt->edges[nd->first_edge + j].label == sid) {
          next = tt->edges[nd->first_edge + j].next_node;
          break;
        }
      }
    }

    while (next == UINT32_MAX && node != 0) {
      node = tt->failure_links[node];
      if (node == 0 && sid < BASE_SYMBOL_OFFSET) {
        next = tt->first_syl[sid];
      } else {
        SylTrieNode *nd = &tt->nodes[node];
        for (uint16_t j = 0; j < nd->edge_count; j++) {
          if (tt->edges[nd->first_edge + j].label == sid) {
            next = tt->edges[nd->first_edge + j].next_node;
            break;
          }
        }
      }
    }

    if (next == UINT32_MAX) {
      node = 0;
    } else {
      node = next;
    }

    uint32_t out_node = node;
    while (out_node != 0) {
      if (tt->nodes[out_node].token_id != 0) {
        if (found < max_matches) {
          out_matches[found].token_id = 0;
          out_matches[found].length = tt->depths[out_node];
          out_matches[found].start_pos = (uint32_t)(i + 1 - out_matches[found].length);
          out_matches[found].category_mask = tt->nodes[out_node].token_id & ~0x80000000;
          found++;
        }
      }
      out_node = tt->failure_links[out_node];
    }
  }

  return (int)found;
}

uint64_t compute_truth_seed_hash(void) {
  uint64_t h = 0xcbf29ce484222325ULL; /* FNV offset basis */
  for (size_t i = 0; i < N_SEEDS; i++) {
    const char *s = TRUTH_SEED_DATA[i].text;
    while (*s) {
      h ^= (uint64_t)(unsigned char)*s++;
      h *= 0x100000001b3ULL;
    }
    uint32_t cat = TRUTH_SEED_DATA[i].category;
    h ^= (uint64_t)cat;
    h *= 0x100000001b3ULL;
  }
  return h;
}
