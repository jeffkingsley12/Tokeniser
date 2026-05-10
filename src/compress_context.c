/*
 * compress_context.c
 *
 * Implementation of reusable compression context for Re-Pair
 * to prevent heap thrashing and improve performance.
 */

#include "compress_context.h"
#include "tokenizer_utils.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

/* Hash function for pair tables — single definition used by both
 * pair_table_find and pair_table_insert (previously duplicated). */
static inline uint32_t ht_hash(uint32_t pair, uint32_t mask) {
    return (pair * 0x9E3779B1u) & mask;
}

/* Create hash table */
static PairHashTable *pair_table_create(uint32_t size) {
    PairHashTable *ht = malloc(sizeof(PairHashTable));
    if (!ht) return NULL;
    
    size = next_pow2(size);
    ht->buckets = calloc(size, sizeof(CompressPairEntry *));
    if (!ht->buckets) {
        free(ht);
        return NULL;
    }
    
    ht->size = size;
    ht->count = 0;
    ht->mask = size - 1;
    return ht;
}

/* Destroy hash table */
static void pair_table_destroy(PairHashTable *ht) {
    if (!ht) return;
    
    for (uint32_t i = 0; i < ht->size; i++) {
        CompressPairEntry *entry = ht->buckets[i];
        while (entry) {
            CompressPairEntry *next = entry->next;
            free(entry);
            entry = next;
        }
    }
    
    free(ht->buckets);
    free(ht);
}

/* Clear hash table — O(size + count) due to per-bucket scan and per-entry free.
 * TODO: allocate CompressPairEntry from the slab for true O(1) clear via slab_reset. */
void pair_table_clear(PairHashTable *ht) {
    if (!ht) return;
    
    for (uint32_t i = 0; i < ht->size; i++) {
        CompressPairEntry *entry = ht->buckets[i];
        while (entry) {
            CompressPairEntry *next = entry->next;
            free(entry);
            entry = next;
        }
        ht->buckets[i] = NULL;
    }
    ht->count = 0;
}

/* Find pair in hash table */
CompressPairEntry *pair_table_find(PairHashTable *ht, uint32_t pair) {
    if (!ht) return NULL;
    
    uint32_t hash = ht_hash(pair, ht->mask);
    CompressPairEntry *entry = ht->buckets[hash];
    
    while (entry) {
        if (entry->pair == pair) {
            return entry;
        }
        entry = entry->next;
    }
    return NULL;
}

/* Insert pair into hash table */
CompressPairEntry *pair_table_insert(PairHashTable *ht, uint32_t pair, uint32_t freq) {
    if (!ht) return NULL;
    
    uint32_t hash = ht_hash(pair, ht->mask);
    
    /* Check if already exists */
    CompressPairEntry *entry = ht->buckets[hash];
    while (entry) {
        if (entry->pair == pair) {
            entry->freq += freq;
            return entry;
        }
        entry = entry->next;
    }
    
    /* Create new entry */
    if (ht->count >= ht->size * 3 / 4) {
        uint32_t new_size = ht->size * 2;
        CompressPairEntry **new_buckets = calloc(new_size, sizeof(CompressPairEntry *));
        if (new_buckets) {
            uint32_t new_mask = new_size - 1;
            for (uint32_t i = 0; i < ht->size; i++) {
                CompressPairEntry *curr = ht->buckets[i];
                while (curr) {
                    CompressPairEntry *next = curr->next;
                    uint32_t h = ht_hash(curr->pair, new_mask);
                    curr->next = new_buckets[h];
                    new_buckets[h] = curr;
                    curr = next;
                }
            }
            free(ht->buckets);
            ht->buckets = new_buckets;
            ht->size = new_size;
            ht->mask = new_mask;
            hash = ht_hash(pair, ht->mask);
        }
    }

    entry = malloc(sizeof(CompressPairEntry));
    if (!entry) return NULL;
    
    entry->pair = pair;
    entry->freq = freq;
    entry->new_symbol = 0;
    entry->next = ht->buckets[hash];
    ht->buckets[hash] = entry;
    ht->count++;
    
    return entry;
}

/* Create slab allocator */
static SlabAllocator *slab_create(void) {
    SlabAllocator *slab = malloc(sizeof(SlabAllocator));
    if (!slab) return NULL;
    
    slab->current = malloc(sizeof(CompressSlabBlock));
    if (!slab->current) {
        free(slab);
        return NULL;
    }
    
    slab->current->next = NULL;
    slab->current->used = 0;
    slab->head = slab->current;
    
    return slab;
}

/* Destroy slab allocator */
static void slab_destroy(SlabAllocator *slab) {
    if (!slab) return;
    
    CompressSlabBlock *block = slab->head;
    while (block) {
        CompressSlabBlock *next = block->next;
        free(block);
        block = next;
    }
    
    free(slab);
}

/* Reset slab allocator (keep first slab, free others) */
void slab_reset(SlabAllocator *slab) {
    if (!slab) return;
    
    /* Free all but the first slab */
    CompressSlabBlock *block = slab->head->next;
    while (block) {
        CompressSlabBlock *next = block->next;
        free(block);
        block = next;
    }
    
    /* Reset first slab */
    slab->head->next = NULL;
    slab->head->used = 0;
    slab->current = slab->head;
}

/* Allocate from slab */
void *slab_alloc(SlabAllocator *slab, size_t size) {
    if (!slab) return NULL;
    
    /* Reject requests that can never fit in any single slab block. */
    if (size > sizeof(slab->current->data)) return NULL;

    /* Align to 8-byte boundary. Note: does not handle SIZE_MAX wrap, 
     * but current callers never pass near-limit sizes. */
    size = (size + 7) & ~7;
    if (size > sizeof(slab->current->data)) return NULL;
    
    if (slab->current->used + size > sizeof(slab->current->data)) {
        /* Need new slab */
        CompressSlabBlock *new_block = malloc(sizeof(CompressSlabBlock));
        if (!new_block) return NULL;
        
        new_block->next = NULL;
        new_block->used = 0;
        slab->current->next = new_block;
        slab->current = new_block;
    }
    
    void *ptr = slab->current->data + slab->current->used;
    slab->current->used += size;
    return ptr;
}

/* Create compression context */
CompressContext *compress_context_create(uint32_t hash_size) {
    CompressContext *ctx = malloc(sizeof(CompressContext));
    if (!ctx) return NULL;
    
    ctx->pair_table = pair_table_create(hash_size);
    if (!ctx->pair_table) {
        free(ctx);
        return NULL;
    }
    
    ctx->slab = slab_create();
    if (!ctx->slab) {
        pair_table_destroy(ctx->pair_table);
        free(ctx);
        return NULL;
    }
    
    ctx->stack_capacity = 1024;
    ctx->symbol_stack = malloc(ctx->stack_capacity * sizeof(uint32_t));
    if (!ctx->symbol_stack) {
        slab_destroy(ctx->slab);
        pair_table_destroy(ctx->pair_table);
        free(ctx);
        return NULL;
    }
    
    ctx->is_initialized = true;
    return ctx;
}

/* Destroy compression context */
void compress_context_destroy(CompressContext *ctx) {
    if (!ctx) return;
    
    if (ctx->pair_table) {
        pair_table_destroy(ctx->pair_table);
    }
    
    if (ctx->slab) {
        slab_destroy(ctx->slab);
    }
    
    if (ctx->symbol_stack) {
        free(ctx->symbol_stack);
    }
    
    free(ctx);
}

/* Reset context for reuse */
void compress_context_reset(CompressContext *ctx) {
    if (!ctx || !ctx->is_initialized) return;
    
    pair_table_clear(ctx->pair_table);
    slab_reset(ctx->slab);
    /* symbol_stack doesn't need resetting, just ensure capacity */
}

/* Getters for compression routines */
PairHashTable *compress_get_pair_table(CompressContext *ctx) {
    return ctx ? ctx->pair_table : NULL;
}

SlabAllocator *compress_get_slab(CompressContext *ctx) {
    return ctx ? ctx->slab : NULL;
}

uint32_t *compress_get_symbol_stack(CompressContext *ctx, size_t min_capacity) {
    if (!ctx) return NULL;
    
    if (min_capacity > ctx->stack_capacity) {
        /* Expand stack using size_t arithmetic to avoid uint32_t overflow.
         * The old code used uint32_t doubling; when stack_capacity * 2
         * exceeded UINT32_MAX it wrapped to 0, making new_capacity
         * permanently less than min_capacity and looping forever. */
        size_t new_capacity = (size_t)ctx->stack_capacity * 2;
        while (new_capacity < min_capacity) {
            if (new_capacity > SIZE_MAX / 2) return NULL;  /* overflow guard */
            new_capacity *= 2;
        }

        /* Realloc size must also not overflow the byte count. */
        if (new_capacity > SIZE_MAX / sizeof(uint32_t)) return NULL;

        uint32_t *new_stack = realloc(ctx->symbol_stack, new_capacity * sizeof(uint32_t));
        if (!new_stack) return NULL;
        
        ctx->symbol_stack = new_stack;
        ctx->stack_capacity = new_capacity;
    }
    
    return ctx->symbol_stack;
}
