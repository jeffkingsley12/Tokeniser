/*
===============================================================================
LEXEME INTERN IMPLEMENTATION — Canonical Surface Form Registry
===============================================================================

Maps identical surface forms to canonical lexeme IDs, enabling aggregated
statistics and proper strongly connected component merging.

Thread safety: Double-checked locking for lookup, synchronized insertion.
Memory: Dynamic array growth, O(lexeme_count) space.
Time: O(1) hash table lookup (L-3 FIX), O(1) amortized insertion.

L-3 FIX: Added hash table for O(1) lookup instead of linear search.
===============================================================================
*/

#include "lexeme_intern.h"
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include <stdatomic.h>

/* Sentinel value for invalid lexeme ID */
#define LEXEME_INVALID  0xFFFFFFFFU

/* L-3 FIX: Hash table configuration */
#define HASH_TABLE_SIZE 65536  /* Power of 2 for fast modulo */
#define HASH_TABLE_MASK (HASH_TABLE_SIZE - 1)

/* Hash table entry */
typedef struct {
    uint64_t hash;
    uint32_t lexeme_id;
} HashEntry;

/* L-3 FIX: Static hash table for O(1) lookup */
static HashEntry g_hash_table[HASH_TABLE_SIZE];
static pthread_mutex_t g_hash_table_mutex = PTHREAD_MUTEX_INITIALIZER;

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

/* L-3 FIX: Hash table lookup - returns lexeme_id if found, LEXEME_INVALID otherwise */
static uint32_t hash_table_lookup(uint64_t hash) {
    uint32_t index = (uint32_t)(hash & HASH_TABLE_MASK);
    
    /* Linear probing for collision resolution */
    for (uint32_t i = 0; i < HASH_TABLE_SIZE; i++) {
        uint32_t pos = (index + i) & HASH_TABLE_MASK;
        if (g_hash_table[pos].hash == hash) {
            return g_hash_table[pos].lexeme_id;
        }
        if (g_hash_table[pos].hash == 0) {
            break;  /* Empty slot, not found */
        }
    }
    return LEXEME_INVALID;
}

/* L-3 FIX: Hash table insert */
static void hash_table_insert(uint64_t hash, uint32_t lexeme_id) {
    uint32_t index = (uint32_t)(hash & HASH_TABLE_MASK);
    
    /* Linear probing for collision resolution */
    for (uint32_t i = 0; i < HASH_TABLE_SIZE; i++) {
        uint32_t pos = (index + i) & HASH_TABLE_MASK;
        if (g_hash_table[pos].hash == 0 || g_hash_table[pos].hash == hash) {
            g_hash_table[pos].hash = hash;
            g_hash_table[pos].lexeme_id = lexeme_id;
            return;
        }
    }
    /* Hash table full - this is unlikely with 65536 entries */
}

/**
 * le_intern_lexeme - Canonicalize a surface form to a lexeme ID.
 *
 * L-3 FIX: Uses hash table for O(1) lookup instead of linear search.
 * Double-checked locking pattern:
 *   1. Fast path: O(1) hash table lookup without lock
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
    
    /* L-3 FIX: Fast path - O(1) hash table lookup */
    pthread_mutex_lock(&g_hash_table_mutex);
    uint32_t cached_id = hash_table_lookup(hash);
    if (cached_id != LEXEME_INVALID) {
        /* Verify the cached entry is still valid (string comparison) */
        pthread_rwlock_rdlock(&ctx->lexeme_rwlock);
        if (cached_id < ctx->lexeme_count &&
            ctx->lexemes[cached_id].hash == hash &&
            strcmp(ctx->lexemes[cached_id].surface, surface_form) == 0) {
            atomic_fetch_add_explicit(&ctx->lexemes[cached_id].freq, 1, memory_order_relaxed);
            pthread_rwlock_unlock(&ctx->lexeme_rwlock);
            pthread_mutex_unlock(&g_hash_table_mutex);
            return cached_id;
        }
        pthread_rwlock_unlock(&ctx->lexeme_rwlock);
    }
    pthread_mutex_unlock(&g_hash_table_mutex);
    
    /* 2. Slow Path: Upgrade to Exclusive Write Lock */
    pthread_rwlock_wrlock(&ctx->lexeme_rwlock);
    
    /* Double-check loop: verify no other thread inserted it during lock upgrade */
    for (uint32_t i = 0; i < ctx->lexeme_count; i++) {
        if (ctx->lexemes[i].hash == hash &&
            strcmp(ctx->lexemes[i].surface, surface_form) == 0) {
            atomic_fetch_add_explicit(&ctx->lexemes[i].freq, 1, memory_order_relaxed);
            
            /* L-3 FIX: Update hash table cache */
            pthread_mutex_lock(&g_hash_table_mutex);
            hash_table_insert(hash, i);
            pthread_mutex_unlock(&g_hash_table_mutex);
            
            pthread_rwlock_unlock(&ctx->lexeme_rwlock);
            return i;
        }
    }
    
    uint32_t new_id = ctx->lexeme_count;
    if (new_id >= ctx->lexeme_capacity) {
        uint32_t new_cap = ctx->lexeme_capacity > 0 ? ctx->lexeme_capacity * 2 : 256;
        LexemeEntry *new_lexemes = realloc(ctx->lexemes, new_cap * sizeof(LexemeEntry));
        if (!new_lexemes) {
            pthread_rwlock_unlock(&ctx->lexeme_rwlock);
            return LEXEME_INVALID;
        }
        ctx->lexemes = new_lexemes;
        ctx->lexeme_capacity = new_cap;
    }
    
    ctx->lexemes[new_id].hash = hash;
    ctx->lexemes[new_id].surface = SAFE_STRDUP(surface_form);
    if (!ctx->lexemes[new_id].surface) {
        pthread_rwlock_unlock(&ctx->lexeme_rwlock);
        return LEXEME_INVALID;
    }
    ctx->lexemes[new_id].lexeme_id = new_id;
    atomic_init(&ctx->lexemes[new_id].freq, 1);
    
    /* L-3 FIX: Insert into hash table cache */
    pthread_mutex_lock(&g_hash_table_mutex);
    hash_table_insert(hash, new_id);
    pthread_mutex_unlock(&g_hash_table_mutex);
    
    ctx->lexeme_count++;
    pthread_rwlock_unlock(&ctx->lexeme_rwlock);
    
    return new_id;
}

/**
 * le_lexeme_frequency - Query total occurrence count of a lexeme.
 */
uint32_t le_lexeme_frequency(EngineContext *ctx, uint32_t lexeme_id) {
    if (!ctx || lexeme_id >= ctx->lexeme_count)
        return 0;
    
    pthread_rwlock_rdlock(&ctx->lexeme_rwlock);
    uint32_t freq = atomic_load_explicit(&ctx->lexemes[lexeme_id].freq,
                                         memory_order_relaxed);
    pthread_rwlock_unlock(&ctx->lexeme_rwlock);
    return freq;
}

/**
 * le_lexeme_surface - Reverse lookup: lexeme_id → surface form.
 *
 * WARNING: Returns an internal pointer after releasing the read lock.
 * Prefer le_lexeme_surface_safe() in concurrent contexts.
 */
const char* le_lexeme_surface(EngineContext *ctx, uint32_t lexeme_id) {
    if (!ctx || lexeme_id >= ctx->lexeme_count)
        return NULL;
    
    pthread_rwlock_rdlock(&ctx->lexeme_rwlock);
    const char *surface = ctx->lexemes[lexeme_id].surface;
    pthread_rwlock_unlock(&ctx->lexeme_rwlock);
    return surface;
}

/**
 * le_lexeme_surface_safe - Thread-safe reverse lookup with bounded copy.
 *
 * Copies the surface form into a caller-allocated buffer while the read
 * lock is held, ensuring zero exposure of internal memory addresses
 * outside the lock scope. Immune to internal pointer drift from
 * concurrent compaction, cleanup, or destructive collection sequences.
 *
 * Returns 0 on success, or a negative error code on failure.
 */
int le_lexeme_surface_safe(EngineContext *ctx, uint32_t lexeme_id,
                           char *out_buf, size_t buf_capacity) {
    if (!ctx || !out_buf || buf_capacity == 0) return -1;

    pthread_rwlock_rdlock(&ctx->lexeme_rwlock);

    if (lexeme_id >= ctx->lexeme_count) {
        pthread_rwlock_unlock(&ctx->lexeme_rwlock);
        return -2; /* Out of bounds */
    }

    const char *surface = ctx->lexemes[lexeme_id].surface;
    if (!surface) {
        pthread_rwlock_unlock(&ctx->lexeme_rwlock);
        return -3; /* Null or uninitialized entry */
    }

    size_t len = strlen(surface);
    if (len >= buf_capacity) {
        pthread_rwlock_unlock(&ctx->lexeme_rwlock);
        return -4; /* Buffer too small (truncation risk) */
    }

    memcpy(out_buf, surface, len + 1);

    pthread_rwlock_unlock(&ctx->lexeme_rwlock);
    return 0; /* Successful atomic extraction */
}
