/*
 * compress_context.h
 *
 * Reusable compression context to prevent heap thrashing
 * in Re-Pair compression. The context contains pre-allocated
 * hash tables and slab allocators that can be reused across
 * multiple compression calls.
 */

#ifndef COMPRESS_CONTEXT_H
#define COMPRESS_CONTEXT_H

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>

/* Hash table entry for pair frequencies */
typedef struct PairEntry {
    uint32_t pair;          /* packed pair: (left << 16) | right */
    uint32_t freq;          /* frequency of this pair */
    uint32_t new_symbol;    /* symbol assigned to this pair */
    struct PairEntry *next;  /* next entry in hash bucket */
} PairEntry;

/* Hash table for pair counting */
typedef struct {
    PairEntry **buckets;
    uint32_t size;
    uint32_t count;
    uint32_t mask;  /* size - 1 for fast modulo */
} PairHashTable;

/* Slab allocator for frequent small allocations */
typedef struct SlabBlock {
    struct SlabBlock *next;
    uint8_t data[4096];  /* 4KB slabs */
    size_t used;
} SlabBlock;

typedef struct {
    SlabBlock *current;
    SlabBlock *head;  /* For cleanup */
} SlabAllocator;

/* Reusable compression context */
typedef struct {
    PairHashTable *pair_table;
    SlabAllocator *slab;
    uint32_t *symbol_stack;  /* For symbol replacement stack */
    size_t stack_capacity;
    bool is_initialized;
} CompressContext;

/* Create/destroy compression context */
CompressContext *compress_context_create(uint32_t hash_size);
void compress_context_destroy(CompressContext *ctx);

/* Reset context for reuse (O(1) operation) */
void compress_context_reset(CompressContext *ctx);

/* Accessors for compression routines */
PairHashTable *compress_get_pair_table(CompressContext *ctx);
SlabAllocator *compress_get_slab(CompressContext *ctx);
uint32_t *compress_get_symbol_stack(CompressContext *ctx, size_t min_capacity);

/* Hash table operations */
void pair_table_clear(PairHashTable *ht);
PairEntry *pair_table_find(PairHashTable *ht, uint32_t pair);
PairEntry *pair_table_insert(PairHashTable *ht, uint32_t pair, uint32_t freq);

/* Slab allocator operations */
void slab_reset(SlabAllocator *slab);
void *slab_alloc(SlabAllocator *slab, size_t size);

#endif /* COMPRESS_CONTEXT_H */
