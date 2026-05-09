/*
 * repair.c
 *
 * Re-Pair grammar compression over syllable-ID sequences.
 * Optimized with Max-Heap and Occurrence Lists for O(N) training.
 *
 * THREADING CONTRACT:
 * - Training (repair_train) is NOT thread-safe (uses global slab and mutable state).
 * - Inference/Compression is thread-safe when using repair_compress_with_context.
 */

#include "tokenizer.h"
#include "tokenizer_utils.h"

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <assert.h>
#include <stdbool.h>

#define PAIR_HT_INIT_SIZE  65536u
#define PAIR_HT_MAX_SIZE   (1u << 26)

typedef struct {
    uint64_t key;
    uint32_t lhs;
} CompressEntry;

typedef struct RePairCompressor {
    CompressEntry *ht;
    uint32_t ht_size;
} RePairCompressor;
static uint32_t pair_hash(uint64_t key, uint32_t mask) {
    uint64_t h = key ^ (key >> 30);
    h *= 0xbf58476d1ce4e5b9ULL;
    h ^= (h >> 27);
    h *= 0x94d049bb133111ebULL;
    h ^= (h >> 31);
    return (uint32_t)(h & mask);
}

static inline uint64_t pair_key(uint32_t l, uint32_t r) {
    return ((uint64_t)(l + 1) << 32) | (uint64_t)(r + 1);
}

/* Heap functions */
static inline void heap_swap(RePairState *rs, uint32_t i, uint32_t j) {
    uint32_t slot_i = rs->heap[i];
    uint32_t slot_j = rs->heap[j];
    rs->heap[i] = slot_j;
    rs->heap[j] = slot_i;
    rs->heap_pos[slot_i] = j;
    rs->heap_pos[slot_j] = i;
}

static inline bool heap_less(RePairState *rs, uint32_t i, uint32_t j) {
    uint32_t slot_i = rs->heap[i];
    uint32_t slot_j = rs->heap[j];
    if (rs->pair_freq[slot_i] == rs->pair_freq[slot_j]) {
        /* Deterministic tie-breaking: compare the actual pair components */
        uint64_t key_i = rs->pair_keys[slot_i];
        uint64_t key_j = rs->pair_keys[slot_j];
        uint32_t left_i = (uint32_t)((key_i >> 32) - 1);
        uint32_t right_i = (uint32_t)(key_i & 0xFFFFFFFF) - 1;
        uint32_t left_j = (uint32_t)((key_j >> 32) - 1);
        uint32_t right_j = (uint32_t)(key_j & 0xFFFFFFFF) - 1;
        
        /* Compare left syllable ID first, then right syllable ID */
        if (left_i != left_j) {
            return left_i < left_j;
        }
        return right_i < right_j;
    }
    return rs->pair_freq[slot_i] < rs->pair_freq[slot_j];
}

static void heap_sift_up(RePairState *rs, uint32_t idx) {
    while (idx > 0) {
        uint32_t parent = (idx - 1) / 2;
        if (!heap_less(rs, parent, idx)) break;
        heap_swap(rs, parent, idx);
        idx = parent;
    }
}

static void heap_sift_down(RePairState *rs, uint32_t idx) {
    uint32_t size = rs->heap_size;
    while (true) {
        uint32_t left = 2 * idx + 1;
        uint32_t right = 2 * idx + 2;
        uint32_t largest = idx;
        if (left < size && heap_less(rs, largest, left)) largest = left;
        if (right < size && heap_less(rs, largest, right)) largest = right;
        if (largest == idx) break;
        heap_swap(rs, idx, largest);
        idx = largest;
    }
}

static void heap_update(RePairState *rs, uint32_t slot) {
    uint32_t idx = rs->heap_pos[slot];
    if (idx == UINT32_MAX) {
        if (rs->pair_freq[slot] > 0) {
            idx = rs->heap_size++;
            rs->heap[idx] = slot;
            rs->heap_pos[slot] = idx;
            heap_sift_up(rs, idx);
        }
        return;
    }
    if (rs->pair_freq[slot] == 0) {
        uint32_t last_slot = rs->heap[--rs->heap_size];
        rs->heap_pos[slot] = UINT32_MAX;
        if (idx < rs->heap_size) {
            rs->heap[idx] = last_slot;
            rs->heap_pos[last_slot] = idx;
            heap_sift_up(rs, idx);
            heap_sift_down(rs, idx);
        }
        return;
    }
    heap_sift_up(rs, idx);
    heap_sift_down(rs, idx);
}

/* Hash table functions */
static uint32_t pair_ht_find(RePairState *rs, uint64_t key, bool *found) {
    uint32_t mask = rs->pair_ht_size - 1;
    uint32_t slot = pair_hash(key, mask);
    uint32_t probe = 0;
    uint32_t first_tombstone = UINT32_MAX;
    while (probe < rs->pair_ht_size) {
        if (rs->pair_keys[slot] == 0 && rs->pair_freq[slot] == 0) {
            *found = false;
            return (first_tombstone != UINT32_MAX) ? first_tombstone : slot;
        }
        if (rs->pair_keys[slot] == key) {
            *found = true;
            return slot;
        }
        if (rs->pair_freq[slot] == 0 && first_tombstone == UINT32_MAX) {
            first_tombstone = slot;
        }
        slot = (slot + 1) & mask;
        probe++;
    }
    *found = false;
    return (first_tombstone != UINT32_MAX) ? first_tombstone : UINT32_MAX;
}

static int pair_ht_resize(RePairState *rs, uint32_t new_size) {
    uint64_t *new_keys = calloc(new_size, sizeof *new_keys);
    uint32_t *new_freq = calloc(new_size, sizeof *new_freq);
    SeqNode **new_head = calloc(new_size, sizeof *new_head);
    uint32_t *new_heap = calloc(new_size, sizeof *new_heap);
    uint32_t *new_heap_pos = malloc(new_size * sizeof *new_heap_pos);
    SeqNode **new_tail = calloc(new_size, sizeof *new_tail);

    if (!new_keys || !new_freq || !new_head || !new_heap || !new_heap_pos || !new_tail) {
        free(new_keys); free(new_freq); free(new_head); free(new_heap); free(new_heap_pos); free(new_tail);
        rs->fatal_oom = true;
        return -1;
    }

    memset(new_heap_pos, 0xFF, new_size * sizeof *new_heap_pos); // UINT32_MAX

    uint32_t mask = new_size - 1;
    uint32_t new_count = 0;

    for (uint32_t i = 0; i < rs->pair_ht_size; i++) {
        if (rs->pair_keys[i] == 0 || rs->pair_freq[i] == 0) continue;
        uint64_t key = rs->pair_keys[i];
        uint32_t slot = pair_hash(key, mask);
        while (new_keys[slot] != 0) slot = (slot + 1) & mask;
        new_keys[slot] = key;
        new_freq[slot] = rs->pair_freq[i];
        new_head[slot] = rs->head_occ[i];
        new_tail[slot] = rs->tail_occ[i];
        new_count++;
    }

    free(rs->pair_keys); free(rs->pair_freq); free(rs->head_occ); free(rs->tail_occ); free(rs->heap); free(rs->heap_pos);

    rs->pair_keys = new_keys;
    rs->pair_freq = new_freq;
    rs->head_occ = new_head;
    rs->tail_occ = new_tail;
    rs->heap = new_heap;
    rs->heap_pos = new_heap_pos;
    rs->pair_ht_size = new_size;
    rs->pair_count = new_count;
    rs->live_count = new_count;
    rs->heap_size = 0;

    for (uint32_t i = 0; i < new_size; i++) {
        if (new_keys[i] != 0 && new_freq[i] > 0) {
            uint32_t idx = rs->heap_size++;
            rs->heap[idx] = i;
            rs->heap_pos[i] = idx;
            heap_sift_up(rs, idx);
            
            /* Rebuild tail_occ */
            SeqNode *tail = rs->head_occ[i];
            if (tail) {
                while (tail->next_occ) tail = tail->next_occ;
                rs->tail_occ[i] = tail;
            }
        }
    }
    return 0;
}

static uint32_t pair_ht_add(RePairState *rs, uint64_t key, int32_t delta) {
    if (key == 0) return UINT32_MAX;

    if (rs->pair_count >= (rs->pair_ht_size * 65u / 100u)) {
        uint32_t new_size = (rs->live_count > rs->pair_ht_size / 2) ? rs->pair_ht_size * 2 : rs->pair_ht_size;
        if (pair_ht_resize(rs, new_size) != 0) {
            fprintf(stderr, "[repair] OOM during hash table resize\n");
            rs->fatal_oom = true;
            return UINT32_MAX;
        }
    }

    bool found;
    uint32_t slot = pair_ht_find(rs, key, &found);
    if (slot == UINT32_MAX) return UINT32_MAX;

    if (!found) {
        if (delta <= 0) return UINT32_MAX;
        if (rs->pair_keys[slot] == 0) {
            rs->pair_count++;
        }
        rs->pair_keys[slot] = key;
        rs->pair_freq[slot] = (uint32_t)delta;
        rs->live_count++;
    } else {
        bool was_live = (rs->pair_freq[slot] > 0);
        if (delta < 0 && (uint32_t)(-delta) > rs->pair_freq[slot]) rs->pair_freq[slot] = 0;
        else rs->pair_freq[slot] = (uint32_t)((int32_t)rs->pair_freq[slot] + delta);
        bool now_live = (rs->pair_freq[slot] > 0);
        if (was_live && !now_live) rs->live_count--;
        else if (!was_live && now_live) rs->live_count++;
    }
    heap_update(rs, slot);
    return slot;
}

static void add_occurrence(RePairState *rs, uint32_t slot, SeqNode *node) {
    if (slot == UINT32_MAX) return;
    node->next_occ = NULL;
    
    SeqNode *tail = rs->tail_occ[slot];
    if (!tail) {
        /* Empty list - this is the first occurrence */
        node->prev_occ = NULL;
        rs->head_occ[slot] = node;
        rs->tail_occ[slot] = node;
    } else {
        /* O(1) append using tail pointer */
        tail->next_occ = node;
        node->prev_occ = tail;
        rs->tail_occ[slot] = node;
    }
}

static void remove_occurrence(RePairState *rs, uint32_t slot, SeqNode *node) {
    if (slot == UINT32_MAX) return;
    if (node->prev_occ) node->prev_occ->next_occ = node->next_occ;
    else if (rs->head_occ[slot] == node) rs->head_occ[slot] = node->next_occ;
    
    if (node->next_occ) node->next_occ->prev_occ = node->prev_occ;
    else if (rs->tail_occ[slot] == node) rs->tail_occ[slot] = node->prev_occ;
    node->prev_occ = NULL;
    node->next_occ = NULL;
}

/* SlabAllocator */
#define SLAB_BLOCK  65536
typedef struct SlabBlockBody {
    SlabBlock hdr;
    SeqNode nodes[SLAB_BLOCK];
} SlabBlockBody;

static SeqNode *slab_alloc(RePairState *rs) {
    SlabBlockBody *body = (SlabBlockBody *)rs->slab_head;
    if (!body || body->hdr.used >= SLAB_BLOCK) {
        SlabBlockBody *nb = calloc(1, sizeof *nb);
        if (!nb) return NULL;
        nb->hdr.next = rs->slab_head;
        nb->hdr.used = 0;
        rs->slab_head = &nb->hdr;
        body = nb;
    }
    return &body->nodes[body->hdr.used++];
}

static void slab_free_all(RePairState *rs) {
    SlabBlock *b = rs->slab_head;
    while (b) {
        SlabBlock *nx = b->next;
        free(b);
        b = nx;
    }
    rs->slab_head = NULL;
}

static SeqNode *build_list(RePairState *rs, const uint16_t *syms, const uint16_t *flags, uint32_t n) {
    SeqNode *head = slab_alloc(rs);
    if (!head) return NULL;
    head->sym = UINT32_MAX;
    head->flags = 0;
    head->prev = NULL;

    SeqNode *prev = head;
    for (uint32_t i = 0; i < n; i++) {
        SeqNode *node = slab_alloc(rs);
        if (!node) return NULL;
        node->sym = (uint32_t)syms[i];
        node->flags = flags ? flags[i] : 0;
        node->prev = prev;
        node->next = NULL;
        node->prev_occ = NULL;
        node->next_occ = NULL;
        prev->next = node;
        prev = node;
    }
    SeqNode *tail = slab_alloc(rs);
    if (!tail) return NULL;
    tail->sym = UINT32_MAX;
    tail->flags = 0;
    tail->prev = prev;
    tail->next = NULL;
    prev->next = tail;
    return head;
}

RePairState *repair_create(void) {
    RePairState *rs = calloc(1, sizeof *rs);
    if (!rs) return NULL;

    rs->pair_ht_size = PAIR_HT_INIT_SIZE;
    rs->pair_keys = calloc(rs->pair_ht_size, sizeof *rs->pair_keys);
    rs->pair_freq = calloc(rs->pair_ht_size, sizeof *rs->pair_freq);
    rs->head_occ = calloc(rs->pair_ht_size, sizeof *rs->head_occ);
    rs->tail_occ = calloc(rs->pair_ht_size, sizeof *rs->tail_occ);
    rs->heap = calloc(rs->pair_ht_size, sizeof *rs->heap);
    rs->heap_pos = malloc(rs->pair_ht_size * sizeof *rs->heap_pos);

    if (!rs->pair_keys || !rs->pair_freq || !rs->head_occ || !rs->tail_occ || !rs->heap || !rs->heap_pos) {
        repair_destroy(rs);
        return NULL;
    }
    memset(rs->heap_pos, 0xFF, rs->pair_ht_size * sizeof *rs->heap_pos); // UINT32_MAX
    rs->next_sym = BASE_SYMBOL_OFFSET;
    return rs;
}

void repair_destroy(RePairState *rs) {
    if (!rs) return;
    free(rs->pair_keys);
    free(rs->pair_freq);
    free(rs->head_occ);
    free(rs->tail_occ);
    free(rs->heap);
    free(rs->heap_pos);
    slab_free_all(rs);
    free(rs);
}

static void count_pairs(RePairState *rs, SeqNode **heads, uint32_t n_seqs) {
    for (uint32_t s = 0; s < n_seqs; s++) {
        SeqNode *cur = heads[s]->next;
        while (cur && cur->next && cur->next->sym != UINT32_MAX) {
            /* Skip if this boundary is locked (NODE_LOCK_RIGHT on current node) */
            if (!(cur->flags & NODE_LOCK_RIGHT)) {
                uint64_t key = pair_key(cur->sym, cur->next->sym);
                uint32_t slot = pair_ht_add(rs, key, +1);
                if (rs->fatal_oom) return;
                if (slot != UINT32_MAX) {
                    add_occurrence(rs, slot, cur);
                }
            }
            cur = cur->next;
        }
    }
}

static uint64_t find_best_pair(RePairState *rs, uint32_t *out_freq) {
    if (rs->heap_size == 0) {
        *out_freq = 0;
        return 0;
    }
    uint32_t best_slot = rs->heap[0];
    *out_freq = rs->pair_freq[best_slot];
    return rs->pair_keys[best_slot];
}

static void replace_pair(RePairState *rs, uint32_t L, uint32_t R, uint32_t X) {
    uint64_t lr_key = pair_key(L, R);
    bool found;
    uint32_t lr_slot = pair_ht_find(rs, lr_key, &found);
    if (!found || lr_slot == UINT32_MAX) return;

    SeqNode *cur = rs->head_occ[lr_slot];
    while (cur) {
        if (rs->fatal_oom) return;
        SeqNode *next_occ = cur->next_occ;

        if (cur->sym == L && cur->next && cur->next->sym == R) {
            SeqNode *right = cur->next;
            SeqNode *after = right->next;
            SeqNode *before = cur->prev;

            if (before && before->sym != UINT32_MAX) {
                uint64_t left_key = pair_key(before->sym, L);
                bool found_left;
                uint32_t left_slot = pair_ht_find(rs, left_key, &found_left);
                if (found_left) {
                    remove_occurrence(rs, left_slot, before);
                    pair_ht_add(rs, left_key, -1);
                }
            }
            if (after && after->sym != UINT32_MAX) {
                uint64_t right_key = pair_key(R, after->sym);
                bool found_right;
                uint32_t right_slot = pair_ht_find(rs, right_key, &found_right);
                if (found_right) {
                    remove_occurrence(rs, right_slot, right);
                    pair_ht_add(rs, right_key, -1);
                }
            }

            remove_occurrence(rs, lr_slot, cur);
            pair_ht_add(rs, lr_key, -1);

            /* Propagate flags: C.left = A.left, C.right = B.right
             * TF_LOCKED_START is a left-edge flag.
             * TF_LOCKED_END and NODE_LOCK_RIGHT are right-edge flags. */
            cur->flags = (cur->flags & TF_LOCKED_START) | 
                         (right->flags & (TF_LOCKED_END | NODE_LOCK_RIGHT | TF_SEEDED));
            
            cur->sym = X;
            cur->next = after;
            if (after) after->prev = cur;
            
            if (before && before->sym != UINT32_MAX) {
                /* Only add if the boundary is NOT locked */
                if (!(before->flags & NODE_LOCK_RIGHT)) {
                    uint64_t new_left_key = pair_key(before->sym, X);
                    uint32_t new_left_slot = pair_ht_add(rs, new_left_key, +1);
                    if (new_left_slot != UINT32_MAX) add_occurrence(rs, new_left_slot, before);
                }
            }
            if (after && after->sym != UINT32_MAX) {
                /* Only add if the boundary is NOT locked */
                if (!(cur->flags & NODE_LOCK_RIGHT)) {
                    uint64_t new_right_key = pair_key(X, after->sym);
                    uint32_t new_right_slot = pair_ht_add(rs, new_right_key, +1);
                    if (new_right_slot != UINT32_MAX) add_occurrence(rs, new_right_slot, cur);
                }
            }
            
            if (right == next_occ) {
                next_occ = right->next_occ;
            }
            right->sym = UINT32_MAX;
        }
        cur = next_occ;
    }
}

static uint8_t rule_depth(RePairState *rs, uint32_t sym) {
    if (sym < BASE_SYMBOL_OFFSET) return 0;
    uint32_t idx = sym - BASE_SYMBOL_OFFSET;
    if (idx >= rs->rule_count) return 0;
    return rs->rules[idx].depth;
}

int repair_train(RePairState *rs, const uint16_t **corpus, const uint32_t *lengths, uint32_t n_seqs) {
    if (!rs || !corpus || !lengths || n_seqs == 0) return -1;

    SeqNode **heads = calloc(n_seqs, sizeof *heads);
    if (!heads) return -1;

    for (uint32_t i = 0; i < n_seqs; i++) {
        heads[i] = build_list(rs, corpus[i], NULL, lengths[i]);
        if (!heads[i]) goto oom;
    }

    size_t total_symbols = 0;
    for (uint32_t i = 0; i < n_seqs; i++) {
        total_symbols += lengths[i];
    }
    uint32_t estimated_pairs = (uint32_t)(total_symbols / 4);
    uint32_t target_size = next_pow2(estimated_pairs * 2);
    if (target_size < PAIR_HT_INIT_SIZE) target_size = PAIR_HT_INIT_SIZE;
    if (target_size > PAIR_HT_MAX_SIZE) target_size = PAIR_HT_MAX_SIZE;
    
    if (target_size > rs->pair_ht_size) {
        if (pair_ht_resize(rs, target_size) == 0) {
            fprintf(stderr, "[repair] Pre-sized hash table to %u buckets\n", target_size);
        }
    }

    count_pairs(rs, heads, n_seqs);
    if (rs->fatal_oom) goto oom;

    while (rs->rule_count < MAX_RULES) {
        if (rs->fatal_oom) goto oom;
        uint32_t freq = 0;
        uint64_t best = find_best_pair(rs, &freq);

        if (freq < (uint32_t)MIN_PAIR_FREQ) break;

        uint32_t L = (uint32_t)(best >> 32) - 1;
        uint32_t R = (uint32_t)(best & 0xFFFFFFFF) - 1;
        uint32_t X = rs->next_sym++;

        uint8_t d_l = rule_depth(rs, L);
        uint8_t d_r = rule_depth(rs, R);
        uint8_t new_depth = 1 + (d_l > d_r ? d_l : d_r);
        if (new_depth > MAX_RULE_DEPTH) {
            bool found;
            uint32_t slot = pair_ht_find(rs, best, &found);
            if (found) {
                rs->pair_freq[slot] = 0;
                heap_update(rs, slot);
            }
            rs->next_sym--;
            continue;
        }

        Rule *rule = &rs->rules[rs->rule_count++];
        rule->lhs = X;
        rule->rhs[0] = L;
        rule->rhs[1] = R;
        rule->freq = freq;
        rule->eff_freq = freq;
        rule->depth = new_depth;
        rule->dead = false;

        replace_pair(rs, L, R, X);
    }

    free(heads);
    return 0;

oom:
    fprintf(stderr, "[repair] out of memory\n");
    free(heads);
    return -1;
}

#ifndef COMPRESS_HT_SIZE
#define COMPRESS_HT_SIZE  65536u
#endif

static int build_compress_map(const RePairState *rs, CompressEntry *ht, uint32_t ht_size) {
    uint32_t mask = ht_size - 1;
    memset(ht, 0, ht_size * sizeof *ht);

    for (uint32_t ri = 0; ri < rs->rule_count; ri++) {
        const Rule *r = &rs->rules[ri];
        if (r->dead) continue;

        uint64_t key = pair_key(r->rhs[0], r->rhs[1]);
        if (key == 0) continue;

        uint32_t slot = pair_hash(key, mask);
        uint32_t probe = 0;
        while (ht[slot].key != 0 && ht[slot].key != key) {
            slot = (slot + 1) & mask;
            if (++probe >= ht_size) return -1;
        }
        ht[slot].key = key;
        ht[slot].lhs = r->lhs;
    }
    return 0;
}

static uint32_t compress_lookup(const CompressEntry *ht, uint32_t ht_size, uint32_t l, uint32_t r) {
    uint32_t mask = ht_size - 1;
    uint64_t key = pair_key(l, r);
    if (key == 0) return 0;

    uint32_t slot = pair_hash(key, mask);
    uint32_t probe = 0;
    while (ht[slot].key != 0) {
        if (ht[slot].key == key) return ht[slot].lhs;
        slot = (slot + 1) & mask;
        if (++probe >= ht_size) break;
    }
    return 0;
}

/*
 * Shared compress pass — applies grammar rules to a symbol sequence
 * in-place using a stack to achieve O(N) linear time complexity.
 */
static void compress_pass(const CompressEntry *ht, uint32_t ht_size,
                           uint32_t *seq, uint16_t *flags, uint32_t *len) {
    if (*len < 2) return;

    /* We use the input sequence itself as a stack by tracking top index `w`.
     * This is safe because `w` is always <= `r`. */
    uint32_t w = 0; /* stack top index */
    for (uint32_t r = 0; r < *len; r++) {
        seq[w] = seq[r];
        if (flags) flags[w] = flags[r];
        w++;

        /* Try to merge the top two symbols on the stack */
        while (w >= 2) {
            /* Check if merge is allowed (no lock on the boundary) */
            if (flags && (flags[w - 2] & NODE_LOCK_RIGHT)) break;

            uint32_t lhs = compress_lookup(ht, ht_size, seq[w - 2], seq[w - 1]);
            if (lhs == 0) break;

            /* Merge! */
            w--; /* Pop the second child */
            seq[w - 1] = lhs; /* Replace first child with parent */
            
            if (flags) {
                /* Propagate flags: C.left = A.left, C.right = B.right.
                 * flags[w-1] was A.flags, flags[w] was B.flags. */
                flags[w - 1] = (flags[w - 1] & TF_LOCKED_START) | 
                               (flags[w] & (TF_LOCKED_END | NODE_LOCK_RIGHT | TF_SEEDED));
            }
        }
    }
    *len = w;
}

/* DEPRECATED: Allocates memory per call. Prefer repair_compress_with_context()
 * which reuses a persistent hash table across calls. */
__attribute__((deprecated("use repair_compress_with_context")))
int repair_compress(const RePairState *rs, uint32_t **seq, uint32_t *len) {
    if (!rs || !seq || !*seq || !len) return -1;
    if (*len == 0) return 0;

    uint32_t ht_size = COMPRESS_HT_SIZE;
    if (ht_size < rs->rule_count * 2) {
        ht_size = next_pow2(rs->rule_count * 2);
    }

    CompressEntry *ht = calloc(ht_size, sizeof *ht);
    if (!ht) return -1;

    if (build_compress_map(rs, ht, ht_size) != 0) {
        free(ht);
        return -1;
    }

    compress_pass(ht, ht_size, *seq, NULL, len);

    free(ht);
    return 0;
}

RePairCompressor *repair_compressor_init(void) {
    RePairCompressor *comp = malloc(sizeof *comp);
    if (!comp) return NULL;
    
    comp->ht_size = COMPRESS_HT_SIZE;
    comp->ht = calloc(comp->ht_size, sizeof *comp->ht);
    if (!comp->ht) {
        free(comp);
        return NULL;
    }
    return comp;
}

void repair_compressor_destroy(RePairCompressor *comp) {
    if (!comp) return;
    free(comp->ht);
    free(comp);
}

int repair_compress_with_context(RePairCompressor *comp, const RePairState *rs, uint32_t *seq, uint32_t *len) {
    if (!comp || !rs || !seq || !len) return -1;
    if (*len == 0) return 0;
    
    if (comp->ht_size < rs->rule_count * 2) {
        uint32_t new_size = next_pow2(rs->rule_count * 2);
        if (new_size < COMPRESS_HT_SIZE) new_size = COMPRESS_HT_SIZE;
        CompressEntry *new_ht = calloc(new_size, sizeof(*new_ht));
        if (!new_ht) return -1;
        free(comp->ht);
        comp->ht = new_ht;
        comp->ht_size = new_size;
    }

    if (build_compress_map(rs, comp->ht, comp->ht_size) != 0)
        return -1;
    
    compress_pass(comp->ht, comp->ht_size, seq, NULL, len);
    
    return 0;
}

int repair_train_with_mask(RePairState *rs, const uint16_t **corpus,
                          const uint32_t *lengths, uint32_t n_seqs,
                          const MergeMask **masks) {
    if (!rs || !corpus || !lengths || n_seqs == 0) return -1;

    SeqNode **heads = calloc(n_seqs, sizeof *heads);
    if (!heads) return -1;

    for (uint32_t i = 0; i < n_seqs; i++) {
        uint32_t len = lengths[i];
        uint16_t *flags = NULL;
        
        if (masks && masks[i] && len > 1) {
            flags = calloc(len, sizeof(uint16_t));
            if (!flags) {
                for (uint32_t j = 0; j < i; j++) free(heads[j]);
                free(heads);
                return -1;
            }
            for (uint32_t j = 0; j < len - 1; j++) {
                if (!is_merge_allowed(masks[i], j)) {
                    flags[j] |= NODE_LOCK_RIGHT;
                }
            }
        }
        
        heads[i] = build_list(rs, corpus[i], flags, len);
        free(flags);
        
        if (!heads[i]) {
            for (uint32_t j = 0; j < i; j++) free(heads[j]);
            free(heads);
            return -1;
        }
    }

    /* Rest of the logic is identical to repair_train */
    /* Presize hash table */
    size_t total_symbols = 0;
    for (uint32_t i = 0; i < n_seqs; i++) total_symbols += lengths[i];
    uint32_t target_size = next_pow2((uint32_t)(total_symbols / 4) * 2);
    if (target_size < PAIR_HT_INIT_SIZE) target_size = PAIR_HT_INIT_SIZE;
    if (target_size > rs->pair_ht_size) pair_ht_resize(rs, target_size);

    count_pairs(rs, heads, n_seqs);
    if (rs->fatal_oom) goto oom;

    while (rs->rule_count < MAX_RULES) {
        if (rs->fatal_oom) goto oom;
        uint32_t freq = 0;
        uint64_t best = find_best_pair(rs, &freq);
        if (freq < (uint32_t)MIN_PAIR_FREQ) break;

        uint32_t L = (uint32_t)(best >> 32) - 1;
        uint32_t R = (uint32_t)(best & 0xFFFFFFFF) - 1;
        uint32_t X = rs->next_sym++;

        Rule *rule = &rs->rules[rs->rule_count++];
        rule->lhs = X;
        rule->rhs[0] = L;
        rule->rhs[1] = R;
        rule->freq = freq;
        rule->eff_freq = freq;
        rule->depth = 1 + (rule_depth(rs, L) > rule_depth(rs, R) ? rule_depth(rs, L) : rule_depth(rs, R));
        rule->dead = false;

        replace_pair(rs, L, R, X);
    }

    free(heads);
    return 0;

oom:
    free(heads);
    return -1;
}

/* =========================================================
 *  MERGE MASK IMPLEMENTATION
 * ========================================================= */

MergeMask *merge_mask_create(size_t n_adjacencies) {
    MergeMask *mask = safe_malloc(1, sizeof(MergeMask));
    if (!mask) return NULL;
    
    size_t bytes = (n_adjacencies + 7) / 8;
    mask->bits = safe_calloc(bytes, 1);
    if (!mask->bits) {
        free(mask);
        return NULL;
    }
    
    mask->n_boundaries = n_adjacencies;
    return mask;
}

void merge_mask_destroy(MergeMask *mask) {
    if (mask) {
        free(mask->bits);
        free(mask);
    }
}

/*
 * Build a merge mask from token boundary flags.
 *
 * Locking semantics (boundary-only):
 *   - A span [lo, hi) is marked with TF_LOCKED_START on token lo and
 *     TF_LOCKED_END on token hi-1.
 *   - This locks adjacencies at lo-1 (between lo-1 and lo) and hi-1
 *     (between hi-1 and hi), preventing external merges into or out of
 *     the span.
 *   - Internal adjacencies [lo, hi-2] are left UNlocked, allowing tokens
 *     within the span to merge (e.g., [mu][ntu] → [muntu]).
 *   - The merged token inherits TF_LOCK_RIGHT from the right child,
 *     preventing further merges across the now-internal boundary.
 *
 * If full span atomicity is needed (no internal merges), set every
 * adjacency from lo-1 through hi-1.  Boundary-only locking is the
 * default because it yields better compression and reusable subword
 * units while still respecting truth/morpheme boundaries.
 */
MergeMask *merge_mask_from_tokens(const Token *tokens, size_t n_tokens) {
    if (n_tokens < 2) return NULL;
    
    MergeMask *mask = merge_mask_create(n_tokens - 1);
    if (!mask) return NULL;
    
    /* Initialize all adjacencies as allowed */
    for (size_t i = 0; i < mask->n_boundaries; i++) {
        set_merge_allowed(mask, i, true);
    }
    
    /* Lock boundaries: prevent merges across locked spans */
    for (size_t i = 0; i < n_tokens - 1; i++) {
        if ((tokens[i].flags & TF_LOCKED_END) || 
            (tokens[i + 1].flags & TF_LOCKED_START)) {
            set_merge_allowed(mask, i, false);
        }
    }
    
    return mask;
}
