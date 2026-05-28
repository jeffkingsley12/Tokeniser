/*
 * gemini_phrase_seed.c — Phrase registry seeding for Gemini
 *
 * Injects high-frequency Luganda phrase pairs as pre-weighted edges so
 * that culturally critical patterns ("oli otya", "webale nnyo") receive
 * instant recognition without waiting for corpus epochs.
 */

#include "gemini_phrase_seed.h"
#include "libgemini.h"
#include "gemini_internal.h"

#include <stdio.h>
#include <string.h>
#include <limits.h>      /* for INT_MAX */
#include <stdbool.h>     /* for bool type */

/* Forward declaration for le_set_scc_forced (defined in gemini_accessors.c) */
void le_set_scc_forced(EngineContext *ctx, SccID scc_id, bool forced);

/* ── Built-in phrase table (mirrors luganda_phrase_registry.c) ──────────────── */

static const PhraseSeedEntry BUILTIN_PHRASES[] = {
    /* HONORIFICS — highest cultural weight */
    {"oli",      "otya",    2.0f, PHRASE_CAT_HONORIFICS,  1},
    {"webale",   "nnyo",    2.0f, PHRASE_CAT_HONORIFICS,  1},
    {"osiibye",  "otya",    1.9f, PHRASE_CAT_HONORIFICS,  2},
    {"wasuze",   "otya",    1.9f, PHRASE_CAT_HONORIFICS,  2},
    {"ssebo",    "nnyabo",  1.9f, PHRASE_CAT_HONORIFICS,  3},
    {"kulika",   "yo",      1.8f, PHRASE_CAT_HONORIFICS,  4},
    {"weewa",    "nnyo",    1.8f, PHRASE_CAT_HONORIFICS,  5},
    {"gyendi",   "ko",      1.7f, PHRASE_CAT_HONORIFICS,  6},

    /* CONNECTORS */
    {"ate",      "nno",     1.2f, PHRASE_CAT_CONNECTORS, 15},
    {"era",      "nnyo",    1.2f, PHRASE_CAT_CONNECTORS, 18},
    {"kale",     "nno",     1.3f, PHRASE_CAT_CONNECTORS, 12},
    {"naye",     "nno",     1.3f, PHRASE_CAT_CONNECTORS, 11},
    {"nga",      "bwe",     1.4f, PHRASE_CAT_CONNECTORS,  8},
    {"olw'",     "okuba",   1.5f, PHRASE_CAT_CONNECTORS,  5},
    {"bwe",      "nga",     1.2f, PHRASE_CAT_CONNECTORS, 28},
    {"kino",     "kyo",     1.3f, PHRASE_CAT_CONNECTORS, 24},
    {"kino",     "kiri",    1.3f, PHRASE_CAT_CONNECTORS, 25},
    {"kuva",     "mu",      1.2f, PHRASE_CAT_CONNECTORS, 32},

    /* TEMPORAL */
    {"buli",     "lunaku",  1.6f, PHRASE_CAT_TEMPORAL,    9},
    {"buli",     "muntu",   1.6f, PHRASE_CAT_TEMPORAL,   10},
    {"buli",     "saawa",   1.5f, PHRASE_CAT_TEMPORAL,   14},
    {"olunaku",  "luno",    1.4f, PHRASE_CAT_TEMPORAL,   20},
    {"omwezi",   "guno",    1.4f, PHRASE_CAT_TEMPORAL,   22},
    {"saawa",    "lumu",    1.8f, PHRASE_CAT_TEMPORAL,    2},
    {"saawa",    "bbiri",   1.8f, PHRASE_CAT_TEMPORAL,    3},
    {"saawa",    "ssatu",   1.8f, PHRASE_CAT_TEMPORAL,    4},

    /* AUXILIARIES */
    {"asobola",  "okukola", 1.5f, PHRASE_CAT_AUXILIARIES, 9},
    {"ayagala",  "okujja",  1.5f, PHRASE_CAT_AUXILIARIES,10},
    {"agenda",   "okulya",  1.4f, PHRASE_CAT_AUXILIARIES,12},
    {"ali",      "mubulamu",1.3f, PHRASE_CAT_AUXILIARIES,17},

    /* IDIOMS */
    {"nnyo",     "nnyo",    1.5f, PHRASE_CAT_IDIOMS,      7},
    {"bambi",    "bambi",   1.6f, PHRASE_CAT_IDIOMS,      8},

    /* QUESTION */
    {"lwaki",    "otya",    1.5f, PHRASE_CAT_QUESTION,    9},
    {"ani",      "ayita",   1.4f, PHRASE_CAT_QUESTION,   12},

    /* NEGATION */
    {"tewali",   "kyo",     1.4f, PHRASE_CAT_NEGATION,   11},
    {"te",       "na",      1.2f, PHRASE_CAT_NEGATION,   23},
    {"si",       "mbu",     1.3f, PHRASE_CAT_NEGATION,   15},

    /* Terminator */
    {NULL, NULL, 0.0f, PHRASE_CAT_COUNT, 0}
};

/* ── Internal helpers ───────────────────────────────────────────────────────── */

/*
 * Inject a single phrase pair into the Gemini graph.
 *
 * Creates or retrieves NodeIDs for w1 and w2, then calls le_add_edge
 * `calls` times (each call atomically increments the edge weight if the
 * edge already exists, or creates it).
 *
 * out_n1 and out_n2 receive the NodeIDs for w1 and w2 so callers can
 * operate on them (e.g. mark_scc_forced) without a second lookup.
 *
 * Returns 0 on success, -1 on error (invalid token ID or node allocation
 * failure).
 * NOTE: le_add_edge handles create-vs-increment internally; we cannot
 * distinguish a new edge from a weight boost at this layer.
 */
static int inject_phrase(EngineContext *ctx, Tokenizer *tok,
                         const char *w1, const char *w2,
                         float weight_calls,
                         NodeID *out_n1, NodeID *out_n2) {
    TokenID id1 = tokenizer_get_id(tok, w1, true);
    TokenID id2 = tokenizer_get_id(tok, w2, true);

    /* FIX: The old guard `id == 0` incorrectly rejected token ID 0, which is
     * the valid <unk> slot (TOK_UNK = 0 per config.py). tokenizer_get_id()
     * returns INVALID (0xFFFFFFFF) on allocation failure, not 0. Using 0 as
     * the failure sentinel would silently drop any phrase whose first or second
     * word happened to be assigned token ID 0 (i.e. the <unk> token). */
    if (id1 == INVALID || id2 == INVALID) return -1;

    NodeID n1 = le_get_or_create_node(ctx, id1);
    NodeID n2 = le_get_or_create_node(ctx, id2);
    if (n1 == LE_INVALID || n2 == LE_INVALID) return -1;

    if (out_n1) *out_n1 = n1;
    if (out_n2) *out_n2 = n2;

    int calls = (int)(weight_calls + 0.5f);
    if (calls < 1) calls = 1;

    le_add_edge_bulk(ctx, n1, n2, (float)calls);
    return 0;
}

/*
 * Mark the SCC containing node n as is_forced so it survives epoch gating.
 * We access the SccNode directly since this is a sibling translation unit
 * that includes gemini_internal.h.
 */
static void mark_scc_forced(EngineContext *ctx, NodeID n) {
    if (n == LE_INVALID || n >= MAX_NODES) return;
    /* FIX: transient_nodes[n].scc_id is a plain uint32_t; the sentinel is INVALID
     * (0xFFFFFFFF), not LE_INVALID (which is the NodeID sentinel — same value, but
     * using LE_INVALID here was misleading since this is an SccID). Also added the
     * n >= MAX_NODES guard to prevent OOB access if n is unexpectedly large. */
    SccID sid = ctx->transient_nodes[n].scc_id;
    if (sid == INVALID || sid >= MAX_SCCS) return;
    /* Use atomic accessor instead of plain assignment to avoid data race */
    le_set_scc_forced(ctx, sid, true);
}

/* ── gemini_phrase_seed_custom ──────────────────────────────────────────────── */

int gemini_phrase_seed_custom(EngineContext        *ctx,
                              Tokenizer            *tok,
                              const PhraseSeedEntry *entries,
                              float                 base_weight,
                              int                   force_rank_threshold,
                              PhraseSeedResult     *result) {
    if (!ctx || !tok || !entries) return -1;
    if (base_weight <= 0.0f) base_weight = 5.0f;

    if (result) memset(result, 0, sizeof(*result));

    int seeded = 0, forced = 0, edges = 0, nodes_created = 0;

    for (int i = 0; entries[i].first_word; i++) {
        const PhraseSeedEntry *e = &entries[i];
        float calls = base_weight * e->bonus_multiplier;
        /* Clamp to prevent float overflow to Inf and subsequent undefined behavior
         * when casting to int */
        if (!isfinite(calls) || calls > (float)INT_MAX) calls = (float)INT_MAX;

        /*
         * FIX: inject_phrase now outputs the resolved NodeIDs for w1 and w2.
         * We sample node_count before/after to track newly allocated nodes,
         * and reuse the returned NodeIDs in the force-mark block instead of
         * doing a redundant second lookup via tokenizer_get_id + le_get_or_create_node.
         */
        NodeID n1 = LE_INVALID, n2 = LE_INVALID;
        uint32_t nc_before = atomic_load(&ctx->node_count);
        int rc = inject_phrase(ctx, tok, e->first_word, e->second_word, calls,
                               &n1, &n2);
        uint32_t nc_after = atomic_load(&ctx->node_count);

        if (rc < 0) continue;

        /* FIX: nc_after - nc_before as uint32_t wraps to a huge value if nc_after
         * somehow equals nc_before (no new nodes). Cast to int32_t first and clamp. */
        int32_t delta = (int32_t)(nc_after - nc_before);
        if (delta > 0) nodes_created += delta;
        seeded++;
        edges++;

        /* Force-promote highly ranked phrases using already-resolved NodeIDs */
        if (e->rank <= force_rank_threshold) {
            mark_scc_forced(ctx, n1);
            mark_scc_forced(ctx, n2);
            forced++;
        }
    }

    if (result) {
        result->phrases_seeded    = seeded;
        result->nodes_created     = nodes_created;
        result->edges_created     = edges;
        result->forced_promotions = forced;
    }

    return seeded;
}

/* ── gemini_phrase_seed ─────────────────────────────────────────────────────── */

int gemini_phrase_seed(EngineContext    *ctx,
                       Tokenizer        *tok,
                       float             base_weight,
                       int               force_rank_threshold,
                       PhraseSeedResult *result) {
    return gemini_phrase_seed_custom(ctx, tok, BUILTIN_PHRASES,
                                     base_weight, force_rank_threshold, result);
}

/* ── gemini_phrase_seed_print ───────────────────────────────────────────────── */

void gemini_phrase_seed_print(const PhraseSeedResult *r) {
    if (!r) { printf("PhraseSeedResult: (null)\n"); return; }
    printf("Phrase Seed Results:\n");
    printf("  Phrases seeded    : %d\n", r->phrases_seeded);
    printf("  Edges created     : %d\n", r->edges_created);
    printf("  Forced promotions : %d\n", r->forced_promotions);
}
