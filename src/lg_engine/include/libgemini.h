/*
===============================================================================
GEMINI LINGUISTIC ENGINE — Production API
libgemini.h — Public header for mobile/web/desktop frontends

Link:    -lgemini -lm

This header exposes ONLY the opaque handle and public functions.
Internal structs (Node, Edge, SccNode) are hidden.
===============================================================================
*/

#ifndef LIBGEMINI_H
#define LIBGEMINI_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>
#include <stdbool.h>

/* =========================== Opaque Handle ================================ */

typedef struct EngineContext EngineContext;

/* =========================== Integer Types ================================ */

typedef uint32_t NodeID;
typedef uint32_t TokenID;
typedef uint32_t SccID;
typedef uint32_t SymbolID;

#define LE_INVALID  0xFFFFFFFFu   /* Sentinel for "no node / no symbol" */

/* =========================== Beam Search Result =========================== */

typedef struct {
    SymbolID* sequence;     /* Caller must NOT free — use le_free_beam_results */
    uint32_t  length;
    float     log_prob;     /* Cumulative log-probability (base-e) */
    SymbolID  current;      /* Tip of the current beam path (sequence[length-1]) */
} BeamState;

/* =========================== Lifecycle ==================================== */

/** Allocate and initialise a new engine context. Returns NULL on failure. */
EngineContext* le_init(void);

/** Destroy a heap-allocated context (from le_init). */
void le_destroy(EngineContext* ctx);

/* =========================== Persistence (mmap) =========================== */

/** Save entire context to a binary file. Returns true on success. */
bool le_save_mmap(EngineContext* ctx, const char* path);

/** Load context from a binary file (memory-mapped). Returns NULL on failure. */
EngineContext* le_load_mmap(const char* path, bool writable);

/** Unmap a context returned by le_load_mmap. Do NOT call le_destroy on it. */
void le_unload_mmap(EngineContext* ctx);

/* =========================== Token Ingestion =============================== */

/**
 * Process a single token.  Pass LE_INVALID as prev_node for the first token
 * in a sentence / session.
 *
 * This is the primary entry point.  Internally it:
 *   1. Gets-or-creates the node for token_id
 *   2. Adds an edge from prev_node → new node (if applicable)
 *   3. Propagates turbulence upstream (incremental BFS)
 *   4. Triggers epoch transitions at EPOCH_SIZE boundaries
 */
/**
 * DEPRECATED: This function is deprecated. Migrate to gemini_enhanced_process_word
 * or gemini_enhanced_process_text in the enhanced pipeline for full feature support
 * including morphology, phonology, n-gram priors, phrase seeding, attestation,
 * and semantic merging.
 */
NodeID le_process_token(EngineContext* ctx, TokenID token_id, NodeID prev_node)
    __attribute__((deprecated("le_process_token is deprecated. Use gemini_enhanced_process_word or gemini_enhanced_process_text instead.")));

/** Get or create a node for a token ID. */
NodeID le_get_or_create_node(EngineContext* ctx, TokenID token_id);

/** Add a directed edge between two nodes (weight += 1.0). */
void le_add_edge(EngineContext* ctx, NodeID from, NodeID to);

/**
 * Add weight_to_add to the edge from → to, creating it if it does not exist.
 * Use this for bulk prior injection (n-gram seeding, phrase seeding) where
 * fractional or large integer weights are needed in a single call.
 */
void le_add_edge_bulk(EngineContext* ctx, NodeID from, NodeID to,
                      float weight_to_add);

/* =========================== Graph Operations ============================== */

/** Merge two SCCs (used by incremental cycle detection). */
void le_merge_sccs(EngineContext* ctx, SccID target_id, SccID source_id);

/** Split an SCC into multiple SCCs based on entropy/coherence thresholds. */
void le_split_sccs(EngineContext* ctx, SccID scc_id);

/** Demote a symbol back to SCC status (removes stale abstractions). */
void demote_symbol(EngineContext* ctx, SymbolID sym_id);

/** Scan and demote all stale symbols. */
uint32_t le_demote_stale_symbols(EngineContext *ctx);

/** Begin a new epoch — updates stability counters and promotes eligible SCCs. */
void le_begin_epoch(EngineContext* ctx);

/** Manually scan and promote all eligible SCCs. Returns promotion count. */
uint32_t le_promote_eligible(EngineContext* ctx);

/** Manually scan all SCCs and identify candidates for promotion. */
void le_update_all_scc_candidates(EngineContext *ctx);

/* =========================== Prediction ==================================== */

/**
 * Beam search over the DAWG for top-K likely continuations.
 * Caller must free results with le_free_beam_results().
 */
BeamState* le_beam_search(EngineContext* ctx, TokenID start_token,
                          uint32_t beam_width, uint32_t max_depth,
                          uint32_t* result_count);

/** Free beam search results (sequences + container). */
void le_free_beam_results(BeamState* results, uint32_t count);

/** Register this thread as a concurrent reader (QSBR). Call before le_beam_search. */
void le_announce_reader(EngineContext* ctx, uint32_t reader_id);

/** Deregister this thread. Call when done reading. */
void le_quiesce_reader(EngineContext* ctx, uint32_t reader_id);

/* =========================== Turbulence Index ============================== */

/**
 * Get the turbulence t(v) for a node.
 *
 * Interpretation:
 *   t = 0    → absorbing (promoted / terminal)      — "on rails"
 *   t ≈ 1    → one step from absorption              — "nearly committed"
 *   t >> 1   → many steps from any absorbing basin    — "brainstorming"
 *
 * Use for the "Flow State" HUD.
 */
float le_get_turbulence(EngineContext* ctx, NodeID node_id);

/* =========================== Entropy Metrics =============================== */

/**
 * Get the current system-wide weighted entropy H_global.
 * Updated at epoch boundaries and after each promotion.
 */
float le_get_global_entropy(EngineContext* ctx);

/* =========================== Accessors ===================================== */

/** Get the external token ID stored in a node. */
TokenID get_node_token(EngineContext* ctx, NodeID node_id);

/**
 * Retrieve node IDs belonging to a promoted symbol.
 * Writes up to max_nodes IDs into result_nodes.  Returns actual count.
 */
uint32_t le_get_symbol_nodes(EngineContext* ctx, SymbolID symbol_id,
                             NodeID* result_nodes, uint32_t max_nodes);

/* Aggregate counters */
uint32_t get_node_count(EngineContext* ctx);
uint32_t get_edge_count(EngineContext* ctx);
uint32_t get_scc_count(EngineContext* ctx);
uint32_t get_symbol_count(EngineContext* ctx);
uint32_t get_token_count(EngineContext* ctx);
uint32_t get_current_epoch(EngineContext* ctx);
uint64_t get_current_step(EngineContext* ctx);
uint64_t get_total_merges(EngineContext* ctx);
uint64_t get_total_promotions(EngineContext* ctx);

/* Closed-Loop Feedback Ingestion Hooks */
float engine_get_scc_stability(EngineContext* ctx);
float engine_get_entropy_delta(EngineContext* ctx);
uint32_t engine_get_active_region_count(EngineContext* ctx);
void engine_set_ingestion_weight_modifier(EngineContext* ctx, float modifier);

/* =========================== Diagnostics =================================== */

/** Print a human-readable statistics summary to stdout. */
void le_print_stats(EngineContext* ctx);

/* =========================== Tokenizer (TST) =============================== */

typedef struct Tokenizer Tokenizer;

/** Initialize a new TST tokenizer. */
Tokenizer* tokenizer_init(void);

/** Destroy a tokenizer. */
void tokenizer_destroy(Tokenizer* t);

/**
 * Get or create a token ID for a word.
 * If create_if_missing is true, assigns a new ID for unknown words.
 * Thread-safe for read-only access; single-threaded for creation.
 */
uint32_t tokenizer_get_id(Tokenizer* t, const char* word, bool create_if_missing);

/**
 * Reverse lookup: TokenID → string.
 * Returns "<?>" if unknown. fast O(1) array lookup.
 */
const char* tokenizer_get_word(Tokenizer* t, uint32_t id);

/**
 * Process raw text using the tokenizer's streaming buffer.
 * Automatically handles word-breaks (spaces, punctuation) and feeds
 * tokens into the engine.  Returns the final node ID.
 */
NodeID tokenizer_process_text(Tokenizer* t, EngineContext* engine,
                              const char* text, NodeID prev_node);

/** Save tokenizer state to binary file. */
bool tokenizer_save(Tokenizer* t, const char* path);

/** Load tokenizer state from binary file. caller must free with tokenizer_destroy. */
Tokenizer* tokenizer_load(const char* path);

/** Get number of tokens in vocabulary. */
uint32_t tokenizer_vocab_size(Tokenizer* t);

#ifdef __cplusplus
}
#endif

#endif /* LIBGEMINI_H */
