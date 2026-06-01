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
#include "lexeme_intern.h"

#include <stdio.h>
#include <string.h>
#include <limits.h>      /* for INT_MAX */
#include <stdbool.h>     /* for bool type */
#include <stdlib.h>      /* for calloc, free */

/* Forward declaration for le_set_scc_forced (defined in gemini_accessors.c) */
void le_set_scc_forced(EngineContext *ctx, SccID scc_id, bool forced);

/* CRITICAL FIX C-5: Move extern declarations to top of file (proper linkage) */
void le_kosaraju_scc(EngineContext *ctx);
void le_recompute_scc_metrics(EngineContext *ctx);

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

    /* FIX A2: Seed the canonical registry so phrase components have
     * non-zero frequency footprints before the first epoch runs. */
    le_intern_lexeme(ctx, w1);
    le_intern_lexeme(ctx, w2);

    NodeID n1 = le_get_or_create_node(ctx, id1);
    NodeID n2 = le_get_or_create_node(ctx, id2);
    if (n1 == LE_INVALID || n2 == LE_INVALID) return -1;

    if (out_n1) *out_n1 = n1;
    if (out_n2) *out_n2 = n2;

    int calls = (int)(weight_calls + 0.5f);
    if (calls < 1) calls = 1;

    /* HYPERGRAPH: Use le_add_edge_bulk for now - relation types will be
     * applied in a future update when the edge creation API supports relations */
    le_add_edge_bulk(ctx, n1, n2, (float)calls);

    /* CRITICAL FIX: Increment SCC frequency to activate seeded phrases
     * Without this, seeded SCCs have freq=0 and never get promoted.
     * Increment by min_freq (5) to immediately satisfy frequency threshold. */
    SccID sid1 = atomic_load_explicit(&ctx->transient_nodes[n1].scc_id, memory_order_acquire);
    SccID sid2 = atomic_load_explicit(&ctx->transient_nodes[n2].scc_id, memory_order_acquire);
    if (sid1 != INVALID && sid1 < MAX_SCCS) {
        atomic_fetch_add_explicit(&ctx->scc_nodes[sid1].freq, 5, memory_order_relaxed);
    }
    if (sid2 != INVALID && sid2 < MAX_SCCS && sid2 != sid1) {
        atomic_fetch_add_explicit(&ctx->scc_nodes[sid2].freq, 5, memory_order_relaxed);
    }

    return 0;
}

/*
 * Mark the SCC containing node n as is_forced so it survives epoch gating.
 * We access the SccNode directly since this is a sibling translation unit
 * that includes gemini_internal.h.
 */
static void mark_scc_forced(EngineContext *ctx, NodeID n) {
    if (n == LE_INVALID || n >= MAX_NODES) return;
    /* FIX: transient_nodes[n].scc_id is _Atomic uint32_t; must use atomic_load_explicit
     * to avoid C11 undefined behavior from non-atomic access to atomic object. */
    SccID sid = atomic_load_explicit(&ctx->transient_nodes[n].scc_id, memory_order_acquire);
    if (sid == INVALID || sid >= MAX_SCCS) return;
    /* Use atomic accessor instead of plain assignment to avoid data race */
    le_set_scc_forced(ctx, sid, true);
}

/*
 * activate_seeded_scc - Bootstrap frequency and stability parameters for seeded SCCs
 *
 * This ensures seeded phrases can climb out of the zero-frequency trap by
 * pre-priming their frequency and stability counters to meet promotion thresholds.
 */
static void activate_seeded_scc(EngineContext *ctx, NodeID n, uint32_t min_freq) {
    if (n == LE_INVALID || n >= MAX_NODES) return;

    SccID sid = atomic_load_explicit(&ctx->transient_nodes[n].scc_id, memory_order_acquire);
    if (sid == INVALID || sid >= MAX_SCCS) return;

    /* 1. Force the frequency past the engine promotion threshold */
    uint32_t current_freq = atomic_load_explicit(&ctx->scc_nodes[sid].freq, memory_order_acquire);
    if (current_freq < min_freq) {
        atomic_store_explicit(&ctx->scc_nodes[sid].freq, min_freq, memory_order_release);
    }

    /* 2. Give the component an explicit historical footprint so stable_epochs won't be zeroed */
    uint32_t members = atomic_load_explicit(&ctx->scc_nodes[sid].member_count, memory_order_acquire);
    atomic_store_explicit(&ctx->scc_nodes[sid].last_member_count, members, memory_order_release);

    /* 3. Give it a baseline stability head-start */
    atomic_store_explicit(&ctx->scc_nodes[sid].stable_epochs, 1, memory_order_release);
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

    /* M-4 FIX: Cache node IDs to avoid redundant lookups in second loop */
    int entry_count = 0;
    for (int i = 0; entries[i].first_word; i++) entry_count++;
    NodeID *cached_n1 = (NodeID *)calloc(entry_count, sizeof(NodeID));
    NodeID *cached_n2 = (NodeID *)calloc(entry_count, sizeof(NodeID));
    if (!cached_n1 || !cached_n2) {
        if (cached_n1) free(cached_n1);
        if (cached_n2) free(cached_n2);
        return -1;
    }

    for (int i = 0; entries[i].first_word; i++) {
        const PhraseSeedEntry *e = &entries[i];
        float calls = base_weight * e->bonus_multiplier;
        /* CRITICAL FIX CR7: Clamp to prevent float overflow to Inf and subsequent undefined behavior
         * when casting to int. Previous code used (float)INT_MAX which rounds up to 2147483648.0f
         * (INT_MAX + 1), and casting that to int is undefined behavior per C11 §6.3.1.4.
         * INT-2 FIX: Use 2147483520.0f (highest precise float under INT_MAX) to avoid UB. */
        if (!isfinite(calls) || calls >= 2147483520.0f) calls = 2147483520.0f;

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

        if (rc < 0) {
            cached_n1[i] = LE_INVALID;
            cached_n2[i] = LE_INVALID;
            continue;
        }

        /* M-4 FIX: Cache node IDs for reuse in second loop */
        cached_n1[i] = n1;
        cached_n2[i] = n2;

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

    /* CRITICAL FIX: Recompute SCCs after all injections to get actual SCC assignments
     * This ensures activate_seeded_scc operates on the real SCCs, not SCC 0 */
    le_kosaraju_scc(ctx);
    le_recompute_scc_metrics(ctx);

    /* Now activate the REAL SCCs containing the seeded nodes */
    for (int i = 0; entries[i].first_word; i++) {
        const PhraseSeedEntry *e = &entries[i];
        if (e->rank <= force_rank_threshold) {
            /* M-4 FIX: Use cached node IDs instead of redundant tokenizer_get_id + le_get_or_create_node lookups */
            NodeID n1 = cached_n1[i];
            NodeID n2 = cached_n2[i];
            if (n1 != LE_INVALID && n2 != LE_INVALID) {
                uint32_t target_min_freq = ctx->min_freq;
                activate_seeded_scc(ctx, n1, target_min_freq);
                activate_seeded_scc(ctx, n2, target_min_freq);
            }
        }
    }

    /* M-4 FIX: Free cached node ID arrays */
    free(cached_n1);
    free(cached_n2);

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
