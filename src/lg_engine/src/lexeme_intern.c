/*
===============================================================================
LEXEME INTERN IMPLEMENTATION — Canonical Surface Form Registry
===============================================================================

Maps identical surface forms to canonical lexeme IDs, enabling aggregated
statistics and proper strongly connected component merging.

Thread safety: Double-checked locking for lookup, synchronized insertion.
Memory: Dynamic array growth, O(lexeme_count) space.
Time: O(lexeme_count) lookup (linear search), O(1) amortized insertion.

Future optimization: Replace linear search with hash table if lexeme_count
becomes a bottleneck (> 10K lexemes).
===============================================================================
*/

#include "lexeme_intern.h"
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include <stdatomic.h>

/* Sentinel value for invalid lexeme ID */
#define LEXEME_INVALID  0xFFFFFFFFU

/* FNV-1a 64-bit hash function (fast, good distribution for words) */
static uint64_t fnv1a_hash(const char *s) {
    uint64_t hash = 0xcbf29ce484222325ULL;  /* FNV offset basis */
    const uint64_t prime = 0x100000001b3ULL; /* FNV prime */
    
    for (; *s; s++) {
        hash ^= (uint64_t)(unsigned char)*s;
        hash *= prime;
    }
    return hash;
}

/**
 * le_intern_lexeme - Canonicalize a surface form to a lexeme ID.
 *
 * Double-checked locking pattern:
 *   1. Fast path: O(lexeme_count) linear search without lock
 *   2. Slow path: Acquire lock, re-check, allocate if needed
 *
 * Lock contention is expected to be minimal because:
 *   - Lexeme lookup hits 99%+ of the time (reused words)
 *   - Insertion only on first occurrence of a new word
 */
uint32_t le_intern_lexeme(EngineContext *ctx, const char *surface_form) {
    if (!ctx || !surface_form || *surface_form == '\0')
        return LEXEME_INVALID;
    
    uint64_t hash = fnv1a_hash(surface_form);
    
    /* ---- FAST PATH: Check existing lexemes with read lock ---- */
    pthread_mutex_lock(&ctx->lexeme_lock);
    for (uint32_t i = 0; i < ctx->lexeme_count; i++) {
        if (ctx->lexemes[i].hash == hash &&
            strcmp(ctx->lexemes[i].surface, surface_form) == 0) {
            /* Found: increment frequency counter */
            atomic_fetch_add_explicit(&ctx->lexemes[i].freq, 1,
                                     memory_order_relaxed);
            pthread_mutex_unlock(&ctx->lexeme_lock);
            return i;
        }
    }
    pthread_mutex_unlock(&ctx->lexeme_lock);
    
    /* ---- SLOW PATH: Allocate new lexeme (synchronized) ---- */
    pthread_mutex_lock(&ctx->lexeme_lock);
    
    /* Double-check: another thread may have inserted while we waited */
    for (uint32_t i = 0; i < ctx->lexeme_count; i++) {
        if (ctx->lexemes[i].hash == hash &&
            strcmp(ctx->lexemes[i].surface, surface_form) == 0) {
            atomic_fetch_add_explicit(&ctx->lexemes[i].freq, 1,
                                     memory_order_relaxed);
            pthread_mutex_unlock(&ctx->lexeme_lock);
            return i;
        }
    }
    
    /* Need to allocate a new entry */
    uint32_t new_id = ctx->lexeme_count;
    
    /* Grow array if at capacity */
    if (new_id >= ctx->lexeme_capacity) {
        uint32_t new_cap = ctx->lexeme_capacity > 0 ?
                          ctx->lexeme_capacity * 2 : 256;
        LexemeEntry *new_lexemes = realloc(ctx->lexemes,
                                          new_cap * sizeof(LexemeEntry));
        if (!new_lexemes) {
            pthread_mutex_unlock(&ctx->lexeme_lock);
            return LEXEME_INVALID;
        }
        ctx->lexemes = new_lexemes;
        ctx->lexeme_capacity = new_cap;
    }
    
    /* Initialize new entry */
    ctx->lexemes[new_id].hash = hash;
    ctx->lexemes[new_id].surface = malloc(strlen(surface_form) + 1);
    if (!ctx->lexemes[new_id].surface) {
        pthread_mutex_unlock(&ctx->lexeme_lock);
        return LEXEME_INVALID;
    }
    strcpy(ctx->lexemes[new_id].surface, surface_form);
    ctx->lexemes[new_id].lexeme_id = new_id;
    atomic_init(&ctx->lexemes[new_id].freq, 1);
    
    ctx->lexeme_count++;
    pthread_mutex_unlock(&ctx->lexeme_lock);
    
    return new_id;
}

/**
 * le_lexeme_frequency - Query total occurrence count of a lexeme.
 */
uint32_t le_lexeme_frequency(EngineContext *ctx, uint32_t lexeme_id) {
    if (!ctx || lexeme_id >= ctx->lexeme_count)
        return 0;
    
    return atomic_load_explicit(&ctx->lexemes[lexeme_id].freq,
                               memory_order_relaxed);
}

/**
 * le_lexeme_surface - Reverse lookup: lexeme_id → surface form.
 */
const char* le_lexeme_surface(EngineContext *ctx, uint32_t lexeme_id) {
    if (!ctx || lexeme_id >= ctx->lexeme_count)
        return NULL;
    
    return ctx->lexemes[lexeme_id].surface;
}
