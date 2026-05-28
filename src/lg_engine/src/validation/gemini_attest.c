/*
 * gemini_attest.c — Attestation gating for Gemini SCC promotion
 *
 * Two-tier lookup: Bloom filter (fast reject) + open-addressing hash table
 * (definitive confirm). Both live in a single calloc'd block for cache
 * locality.
 */

#ifndef _GNU_SOURCE
#  define _GNU_SOURCE
#endif
#ifndef _GNU_SOURCE
#  define _GNU_SOURCE
#endif

#include "gemini_attest.h"
#include "libgemini.h"
#include "gemini_internal.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ── Constants ──────────────────────────────────────────────────────────────── */

#define ATTEST_MAX_WORD_LEN  128
#define ATTEST_ALPHA         0.01f   /* Bloom filter FP target */

/* ── Hash functions ─────────────────────────────────────────────────────────── */

static uint32_t hash_djb2(const char *s) {
    uint32_t h = 5381;
    int c;
    while ((c = (unsigned char)*s++)) h = ((h << 5) + h) ^ (uint32_t)c;
    return h;
}

static uint32_t hash_fnv1a(const char *s) {
    uint32_t h = 2166136261u;
    while (*s) { h ^= (unsigned char)*s++; h *= 16777619u; }
    return h;
}

/* ── GeminiAttestDB ─────────────────────────────────────────────────────────── */

struct GeminiAttestDB {
    /* Bloom filter */
    uint64_t *bloom;
    size_t    bloom_words;   /* Number of uint64_t cells */

    /* Open-addressing hash table of strdup'd words */
    char    **ht_words;
    size_t    ht_size;
    size_t    ht_used;

    size_t    word_count;
};

/* ── Bloom helpers ──────────────────────────────────────────────────────────── */

static inline void bloom_set(GeminiAttestDB *db, const char *word) {
    uint32_t h1 = hash_djb2(word);
    uint32_t h2 = hash_fnv1a(word);
    uint32_t h3 = h1 ^ (h2 << 1);
    size_t   N  = db->bloom_words * 64;

    db->bloom[(h1 % N) / 64] |= 1ULL << ((h1 % N) % 64);
    db->bloom[(h2 % N) / 64] |= 1ULL << ((h2 % N) % 64);
    db->bloom[(h3 % N) / 64] |= 1ULL << ((h3 % N) % 64);
}

static inline bool bloom_check(const GeminiAttestDB *db, const char *word) {
    uint32_t h1 = hash_djb2(word);
    uint32_t h2 = hash_fnv1a(word);
    uint32_t h3 = h1 ^ (h2 << 1);
    size_t   N  = db->bloom_words * 64;

    if (!(db->bloom[(h1 % N) / 64] & (1ULL << ((h1 % N) % 64)))) return false;
    if (!(db->bloom[(h2 % N) / 64] & (1ULL << ((h2 % N) % 64)))) return false;
    if (!(db->bloom[(h3 % N) / 64] & (1ULL << ((h3 % N) % 64)))) return false;
    return true;
}

/* ── Hash table helpers ─────────────────────────────────────────────────────── */

static void ht_insert(GeminiAttestDB *db, const char *word) {
    uint32_t h = hash_fnv1a(word) % (uint32_t)db->ht_size;
    for (size_t i = 0; i < db->ht_size; i++) {
        size_t idx = (h + i) % db->ht_size;
        if (!db->ht_words[idx]) {
            db->ht_words[idx] = strdup(word);
            db->ht_used++;
            return;
        }
        if (strcmp(db->ht_words[idx], word) == 0) return; /* Duplicate */
    }
    /* M-7 FIX: Table-full is not silent. Under adversarial hash inputs (all
     * words with the same hash_fnv1a mod ht_size), linear probing fills the
     * table before ht_used == word_count. Words dropped here are still in the
     * Bloom filter (no false negatives from Bloom) but missing from the hash
     * table, so gemini_attest_db_contains returns false — a false negative.
     * Log the collision so operators can tune ht_size or switch to Robin Hood
     * probing in a follow-up iteration. */
    fprintf(stderr,
            "gemini_attest: WARNING — hash table full, dropping word '%s'. "
            "Consider increasing ht_size (currently word_count*2+1=%zu).\n",
            word, db->ht_size);
}

static bool ht_contains(const GeminiAttestDB *db, const char *word) {
    uint32_t h = hash_fnv1a(word) % (uint32_t)db->ht_size;
    for (size_t i = 0; i < db->ht_size; i++) {
        size_t idx = (h + i) % db->ht_size;
        if (!db->ht_words[idx])  return false;
        if (strcmp(db->ht_words[idx], word) == 0) return true;
    }
    return false;
}

/* ── gemini_attest_db_create ────────────────────────────────────────────────── */

GeminiAttestDB *gemini_attest_db_create(const char **words, size_t count) {
    if (!words || count == 0) return NULL;

    GeminiAttestDB *db = calloc(1, sizeof(GeminiAttestDB));
    if (!db) return NULL;

    db->word_count = count;

    /* Bloom: ~10 bits per element, rounded up to 64-bit cells */
    size_t bits    = count * 10;
    db->bloom_words= (bits / 64) + 1;
    db->bloom      = calloc(db->bloom_words, sizeof(uint64_t));

    /* Hash table: 2× capacity, must be prime or power-of-two */
    db->ht_size  = count * 2 + 1;   /* Simple odd number for probe */
    db->ht_words = calloc(db->ht_size, sizeof(char *));

    if (!db->bloom || !db->ht_words) {
        gemini_attest_db_destroy(db);
        return NULL;
    }

    for (size_t i = 0; i < count; i++) {
        if (!words[i]) continue;
        bloom_set(db, words[i]);
        ht_insert(db, words[i]);
    }

    return db;
}

/* ── gemini_attest_db_create_from_file ─────────────────────────────────────── */

GeminiAttestDB *gemini_attest_db_create_from_file(const char *path) {
    if (!path) return NULL;

    FILE *fp = fopen(path, "r");
    if (!fp) {
        fprintf(stderr, "gemini_attest: cannot open '%s'\n", path);
        return NULL;
    }

    /* First pass: count lines */
    size_t count = 0;
    char   line[ATTEST_MAX_WORD_LEN];
    while (fgets(line, sizeof(line), fp)) count++;
    if (count == 0) { fclose(fp); return NULL; }

    /* Allocate temp array */
    char **words = calloc(count, sizeof(char *));
    if (!words) { fclose(fp); return NULL; }

    /* Second pass: load words */
    rewind(fp);
    size_t loaded = 0;
    while (loaded < count && fgets(line, sizeof(line), fp)) {
        /* Strip trailing newline / whitespace */
        size_t len = strlen(line);
        while (len > 0 && (line[len-1] == '\n' || line[len-1] == '\r' ||
                            line[len-1] == ' '  || line[len-1] == '\t'))
            line[--len] = '\0';
        if (len == 0) continue;
        words[loaded++] = strdup(line);
    }
    fclose(fp);

    GeminiAttestDB *db = gemini_attest_db_create((const char **)words, loaded);

    for (size_t i = 0; i < loaded; i++) free(words[i]);
    free(words);

    return db;
}

/* ── Query / size / destroy ─────────────────────────────────────────────────── */

bool gemini_attest_db_contains(const GeminiAttestDB *db, const char *word) {
    if (!db || !word) return false;
    if (!bloom_check(db, word)) return false;   /* Definitive NO */
    return ht_contains(db, word);               /* Definitive YES/NO */
}

size_t gemini_attest_db_size(const GeminiAttestDB *db) {
    return db ? db->word_count : 0;
}

void gemini_attest_db_destroy(GeminiAttestDB *db) {
    if (!db) return;
    if (db->ht_words) {
        for (size_t i = 0; i < db->ht_size; i++) free(db->ht_words[i]);
        free(db->ht_words);
    }
    free(db->bloom);
    free(db);
}

/* ── gemini_attest_promote ──────────────────────────────────────────────────── */

/*
 * Walk the member-node linked list of an SCC and count how many
 * correspond to attested words.
 */
static float scc_attest_ratio(EngineContext        *ctx,
                               Tokenizer            *tok,
                               SccID                 scc_id,
                               const GeminiAttestDB *db) {
    const SccNode *scc = &ctx->scc_nodes[scc_id];
    uint32_t member_count = atomic_load_explicit(&scc->member_count, memory_order_acquire);
    if (member_count == 0) return 0.0f;

    int attested = 0;
    NodeID cur = atomic_load_explicit(&scc->head, memory_order_acquire);
    int    iters = 0;

    /*
     * FIX: was `iters < (int)scc->member_count + 1`.
     * The +1 allowed one extra traversal step past the final valid member,
     * potentially dereferencing ctx->nodes[LE_INVALID] or an unrelated node
     * when the linked list is exactly member_count long. Correct bound is strict.
     */
    while (cur != LE_INVALID && iters < (int)member_count) {
        TokenID tid  = get_node_token(ctx, cur);
        const char *w = tokenizer_get_word(tok, tid);
        if (w && gemini_attest_db_contains(db, w)) attested++;
        cur = atomic_load_explicit(&ctx->transient_nodes[cur].next_in_scc, memory_order_acquire);
        iters++;
    }

    return (float)attested / (float)member_count;
}

int gemini_attest_promote(EngineContext        *ctx,
                          Tokenizer            *tok,
                          const GeminiAttestDB *db,
                          float                 min_attest_ratio,
                          AttestGateResult     *result) {
    if (!ctx || !tok || !db) return -1;

    /*
     * H-5 SYNCHRONIZATION CONTRACT:
     * This function writes scc->is_candidate (a plain bool) to block promotion.
     * le_update_all_scc_candidates() in the engine thread also reads and writes
     * is_candidate. Concurrent execution produces a C11 data race (UB).
     *
     * REQUIREMENT: gemini_attest_promote MUST be called only between epochs,
     * after the engine has quiesced (no concurrent le_process_token /
     * le_begin_epoch / le_update_all_scc_candidates). Callers must ensure this
     * by calling le_quiesce_reader() and waiting for epoch_in_progress == 0
     * before invoking this function.
     *
     * TODO (follow-up): Promote SccNode.is_candidate to _Atomic bool and use
     * atomic_compare_exchange to make this race-free regardless of call timing.
     */
    if (result) memset(result, 0, sizeof(*result));

    int total_promoted = 0;
    /*
     * H-4 FIX: ctx->scc_count is _Atomic uint32_t. Use atomic_load() to
     * avoid a data race if another thread is concurrently allocating SCCs.
     * (Guarded by the epoch-boundary contract above, but atomic_load is
     * still the correct primitive for an _Atomic field.)
     */
    uint32_t scc_count = atomic_load(&ctx->scc_count);

    for (SccID sid = 0; sid < scc_count; sid++) {
        SccNode *scc = &ctx->scc_nodes[sid];
        /* H-4 FIX: Use atomic_load_explicit for _Atomic member_count */
        if (atomic_load_explicit(&scc->member_count, memory_order_acquire) == 0) continue;
        if (atomic_load_explicit(&scc->is_promoted, memory_order_acquire)) continue;        /* Already done */
        if (!atomic_load_explicit(&scc->is_candidate, memory_order_acquire)) continue;      /* Not eligible by Gemini's gate */

        if (result) result->sccs_evaluated++;

        /* Force-promoted SCCs bypass attestation entirely */
        if (atomic_load_explicit(&scc->is_forced, memory_order_relaxed)) {
            if (result) result->sccs_forced++;
            /* Actual promotion handled by le_promote_eligible below */
            continue;
        }

        float ratio = scc_attest_ratio(ctx, tok, sid, db);

        if (ratio < min_attest_ratio) {
            /* Block: clear candidate flag so le_promote_eligible skips it */
            atomic_store_explicit(&scc->is_candidate, false, memory_order_release);
            if (result) result->sccs_rejected++;
#ifdef GEMINI_ATTEST_VERBOSE
            fprintf(stderr,
                    "gemini_attest: blocked SCC %u (ratio=%.2f, members=%u)\n",
                    sid, ratio, atomic_load_explicit(&scc->member_count, memory_order_acquire));
#endif
        } else {
            if (result) result->sccs_attested++;
        }
    }

    /* Now run the standard Gemini promotion pass (respects is_candidate flag) */
    uint32_t promoted = le_promote_eligible(ctx);
    total_promoted = (int)promoted;

    if (result) result->sccs_promoted = total_promoted;
    return total_promoted;
}

/* ── gemini_attest_result_print ─────────────────────────────────────────────── */

void gemini_attest_result_print(const AttestGateResult *r) {
    if (!r) { printf("AttestGateResult: (null)\n"); return; }
    printf("Attestation Gate Results:\n");
    printf("  SCCs evaluated  : %d\n", r->sccs_evaluated);
    printf("  SCCs attested   : %d\n", r->sccs_attested);
    printf("  SCCs rejected   : %d\n", r->sccs_rejected);
    printf("  SCCs promoted   : %d\n", r->sccs_promoted);
    printf("  SCCs forced     : %d\n", r->sccs_forced);
}
