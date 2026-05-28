/*
 * gemini_attest.h — Attestation gating for Gemini SCC promotion
 *
 * Wraps le_promote_eligible() with a pre-filter that checks whether the
 * member nodes of each promotion candidate represent attested Luganda words.
 * SCCs whose attested ratio falls below a threshold are blocked, preventing
 * noise tokens (misspellings, punctuation fragments, garbled input) from
 * polluting the DAWG with meaningless symbols.
 *
 * Implementation uses a two-tier filter:
 *   Tier 1: 3-hash Bloom filter — O(1) fast rejection (no false negatives)
 *   Tier 2: Linear hash table  — O(1) definitive confirmation
 *
 * Memory budget: ~1 MB for 100k words (Bloom) + ~3 MB hash table = ~4 MB total.
 */

#ifndef GEMINI_ATTEST_H
#define GEMINI_ATTEST_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct EngineContext EngineContext;
typedef struct Tokenizer     Tokenizer;

/* ── Attestation database ───────────────────────────────────────────────────── */

typedef struct GeminiAttestDB GeminiAttestDB;

/**
 * Create an attestation database from a word list file (one word per line).
 * Returns NULL on OOM or file error.
 */
GeminiAttestDB *gemini_attest_db_create_from_file(const char *wordlist_path);

/**
 * Create an attestation database from an in-memory word array.
 * words[count] must be a valid NULL-terminated string array of length count.
 */
GeminiAttestDB *gemini_attest_db_create(const char **words, size_t count);

/**
 * Query whether a word is attested.
 * Returns true if the word was in the word list; never has false negatives.
 * May have a ~1% false-positive rate (Bloom filter property).
 */
bool gemini_attest_db_contains(const GeminiAttestDB *db, const char *word);

/**
 * Return the number of words in the database.
 */
size_t gemini_attest_db_size(const GeminiAttestDB *db);

/**
 * Destroy the database and free all memory.
 */
void gemini_attest_db_destroy(GeminiAttestDB *db);

/* ── Gated promotion ────────────────────────────────────────────────────────── */

typedef struct {
    int sccs_evaluated;      /* Total SCC candidates examined */
    int sccs_attested;       /* Passed attestation check */
    int sccs_rejected;       /* Blocked (too many unattested members) */
    int sccs_promoted;       /* Actually promoted (passed all gates) */
    int sccs_forced;         /* Bypassed attestation (is_forced=true) */
} AttestGateResult;

/**
 * Run one gated promotion pass.
 *
 * For each SCC that passes the Gemini coherence/entropy gate:
 *   1. Walk member nodes and check each word against the attestation DB
 *   2. If attested_ratio >= min_attest_ratio (or SCC is_forced), promote
 *   3. Otherwise, block promotion and log the rejected SCC
 *
 * This replaces a bare call to le_promote_eligible().
 *
 * @param ctx              EngineContext
 * @param tok              Tokenizer (for word lookup)
 * @param db               Attestation database
 * @param min_attest_ratio Fraction of SCC members that must be attested [0,1]
 * @param result           Output stats; may be NULL
 * @return                 Number of SCCs promoted, or -1 on error
 */
int gemini_attest_promote(EngineContext       *ctx,
                          Tokenizer           *tok,
                          const GeminiAttestDB *db,
                          float                min_attest_ratio,
                          AttestGateResult    *result);

void gemini_attest_result_print(const AttestGateResult *result);

#ifdef __cplusplus
}
#endif

#endif /* GEMINI_ATTEST_H */
