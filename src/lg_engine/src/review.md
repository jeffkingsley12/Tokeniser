# Gemini Engine — Production Code Review

## Overview

Reviewed files: `bridge_engine.c`, `gemini_pool.c`, `gemini_eval.c`, `gemini_internal.h`,
`gemini_accessors.c`, `gemini_attest.c`, `gemini_phrase_seed.c`, `gemini_tokenizer.c`.

Severity levels: **CRITICAL** (undefined behavior / crash), **HIGH** (wrong results / race condition), **MEDIUM** (quality / performance degradation), **LOW** (style / latent hazard).

---

## CRITICAL Issues

### C-1 — `bridge_engine.c` L176: `engine_ingest_text` — invalid pointer from `int`
```c
// BUG
Tokenizer *t = (Tokenizer *)(intptr_t)tok_handle;  // tok_handle is int (32-bit)
```
On any 64-bit platform, `int` is 32 bits. Sign-extending it via `intptr_t` and reinterpreting as a pointer produces an address in the low 4 GiB of virtual memory — which is never a valid heap pointer. This crashes on first dereference. All other bridge functions correctly call `tok_encode(tok_handle, ...)` using the external tokenizer handle. `engine_ingest_text` must be made consistent.

**Fix:** Use `tok_encode` + `le_process_token` (the same pattern as `engine_ingest_text_with_lineage`). See `bridge_engine.c` for the corrected implementation.

---

### C-2 — `bridge_engine.c` `dawg_predict`: wrong dedup index causes silent OOB read
```c
// BUG — p is the prediction-output index (0..pred_count-1),
//        but member_buf holds the *current SCC's* member nodes.
if (tid == get_node_token(ctx, member_buf[p])) { ... }
```
`p` iterates `0..pred_count-1` into `member_buf[]`, but `member_buf[]` was filled with nodes of the *current* beam symbol's SCC (0..mc-1). If `pred_count > mc`, this reads past the initialised portion of `member_buf`. In either case, the NodeIDs at `member_buf[p]` are completely unrelated to the predictions already collected, so the deduplication produces arbitrary false positives (drops valid predictions) or false negatives (emits duplicates).

**Fix:** Track seen token IDs in a dedicated array parallel to `predictions[]` / `probabilities[]`. See corrected `dawg_predict` in `bridge_engine.c`.

---

### C-3 — `bridge_engine.c` `engine_infer_distributed`: OOB edge traversal
```c
EdgeID e = atomic_load(&ctx->nodes[current].first_edge);
while (e != INVALID && next_count < MAX_ACTIVE_STATES) {
    NodeID alt = ctx->edges[e].target;        // e may be >= edge_count
    ...
    e = ctx->edge_nexts[e];                   // e may be >= MAX_EDGES
}
```
There is no bounds check on `e` before indexing into `ctx->edges[]` or `ctx->edge_nexts[]`. A corrupt or partially-initialised edge list (INVALID target written as next link, or a freed edge slot) will read outside the allocated arrays. The hot-path function `engine_step` correctly handles `INVALID` node IDs; this path does not.

**Fix:** Add `e < edge_count` guard; see corrected implementation.

---

### C-4 — `gemini_internal.h` / `gemini_engine.c`: `DawgTransition.next` is non-atomic but concurrently accessed
```c
// gemini_internal.h
typedef struct DawgTransition {
    SymbolID         target;
    _Atomic uint32_t weight;
    uint32_t         next;   // ← plain, but...
} DawgTransition;

// gemini_engine.c add_dawg_transition (concurrent writer):
t->next = atomic_load_explicit(&ctx->dawg_nodes[from].first_transition, ...);
atomic_store_explicit(&ctx->dawg_nodes[from].first_transition, id, memory_order_release);

// bridge_engine.c engine_step (concurrent reader):
t_idx = t->next;   // data race — no atomic load
```
`add_dawg_transition` writes `t->next` with a plain store while `engine_step` / `dawg_predict` read it concurrently. C11 §5.1.2.4: concurrent non-atomic read and write to the same memory location is undefined behaviour. On ARM the acquire on `first_transition` does **not** make `t->next` visible — they are separate memory locations.

**Fix:** Promote `DawgTransition.next` to `_Atomic uint32_t`. All read sites must use `atomic_load_explicit(..., memory_order_acquire)` and all write sites must use `atomic_store_explicit(..., memory_order_release)`. See `gemini_internal.h` and `bridge_engine.c`.

---

## HIGH Issues

### H-1 — `bridge_engine.c` heap sort output: predictions in ascending (worst-first) order
After the partial heapsort extraction loop (`for i = k-1; i > 0; i--`), `next_beam[0]` holds the **minimum** score and `next_beam[k-1]` holds the maximum. `memcpy(beam, next_beam, ...)` copies this ascending array unchanged, so predictions are collected lowest-probability-first. When `max_predictions < beam_size`, the highest-probability candidates are silently dropped. This affects both `dawg_predict` and `dawg_predict_multi_hop`.

**Fix:** Iterate the beam from `beam_size - 1` down to `0` during prediction collection. See corrected implementation.

---

### H-2 — `gemini_pool.c` `le_add_incoming_edge_pooled`: stale `capacity` inside write lock
```c
uint32_t capacity = atomic_load(...);   // snapshot before lock
...
pthread_rwlock_wrlock(...);
uint32_t old_capacity = capacity;       // stale — concurrent realloc may have changed this
```
If two threads both find `count == capacity` and both proceed to the reallocation path, the first one acquires the lock, reallocates, and releases. The second one then acquires the lock with a stale `old_capacity` and `current_count`, copies the wrong number of edges, and allocates redundant pool space. The span ends up with an inconsistent count/capacity.

**Fix:** Re-read `capacity` and `count` atomically after acquiring the write lock, and early-return if another thread already extended the capacity. See corrected `gemini_pool.c`.

---

### H-3 — `gemini_eval.c` `gemini_eval_suite_load_csv`: signed integer overflow in capacity growth
```c
int newcap = suite->capacity * 2;   // overflows when capacity >= 2^30
GeminiTestCase *nb = realloc(suite->cases, (size_t)newcap * sizeof(GeminiTestCase));
```
When `suite->capacity >= INT_MAX / 2 + 1` (rare but reachable with large corpora), `capacity * 2` overflows to a negative value. Cast to `size_t` produces a huge number; `realloc` returns NULL; the loop breaks silently with `suite->cases` unchanged but `suite->capacity` not updated — meaning later `suite->cases[suite->count]` accesses go out of bounds.

**Fix:** Use `size_t` arithmetic with an explicit overflow check. See corrected `gemini_eval.c`.

---

### H-4 — `gemini_accessors.c`: plain reads of `_Atomic` SCC fields outside epoch boundary
`member_count`, `is_candidate`, `is_forced`, `head`, and `turbulence` in `SccNode` are read without `atomic_load_explicit`. While C11 permits implicit seq-cst loads on `_Atomic` fields, the engine writes these fields in `le_merge_sccs`, `le_split_sccs`, and `le_update_all_scc_candidates` — all of which can run concurrently with the accessor path. Implicit seq-cst reads have performance implications on ARM/POWER, and their memory-order semantics are surprising.

**Fix:** Replace all `scc->field` reads in `gemini_accessors.c` with `atomic_load_explicit(&scc->field, memory_order_acquire)`. Particularly critical for `member_count` and `is_candidate` in `le_scc_mean_turbulence`, `get_scc_count`, and `engine_get_active_region_count`.

---

### H-5 — `gemini_attest.c` `gemini_attest_promote`: `is_candidate` written without synchronization
```c
scc->is_candidate = false;   // plain store while engine may be reading is_candidate
```
`le_update_all_scc_candidates` in the engine thread writes `is_candidate` while `gemini_attest_promote` (possibly running in a user thread) reads and writes the same field. This is a C11 data race on a non-atomic object.

**Fix:** Either (a) call `gemini_attest_promote` only between epochs when the engine is idle, and document this requirement explicitly, or (b) promote `is_candidate` to `_Atomic bool` with appropriate acquire/release semantics.

---

## MEDIUM Issues

### M-1 — `bridge_engine.c` `dawg_predict` / `dawg_predict_multi_hop`: missing symbol bounds check
```c
Symbol *sym = &ctx->dawg_nodes[state->current];  // state->current could be stale/OOB
```
`state->current` is a `SymbolID` loaded from a DAWG transition's `.target` field. If a stale transition points to a since-demoted symbol whose ID is ≥ `symbol_count`, this is an OOB access into `ctx->dawg_nodes[]`.

**Fix:** `if (state->current >= atomic_load_explicit(&ctx->symbol_count, memory_order_acquire)) continue;`

---

### M-2 — `bridge_engine.c` `engine_infer_distributed`: missing `memory_order_acquire` on `first_edge`
```c
EdgeID e = atomic_load(&ctx->nodes[current].first_edge);
```
`atomic_load` without an explicit memory order defaults to `memory_order_seq_cst`. This works correctly but carries full barrier cost on every iteration. The correct order is `memory_order_acquire` to synchronise with the release store in `le_add_edge`.

---

### M-3 — `gemini_eval.c` `gemini_eval_run`: `start_token == 0` OOV check is fragile
```c
if (start_token == 0) continue;   // tokenizer returns 0 for "not found"
```
The tokenizer uses `0` as both the TST sentinel (root `mid` pointer) and the "not found" return value. Token ID `0` is never assigned to any word (first assigned ID is `1`). If the convention changes, this check silently misclassifies the first token as OOV. Better: `if (start_token == 0 || start_token == (uint32_t)UINT32_MAX) continue;` to also cover the `UINT32_MAX` error return from `tokenizer_get_id`.

---

### M-4 — `gemini_tokenizer.c` `tokenizer_get_id`: shadowed `len` variable
```c
int len = 0;                          // outer: counts chars in clean_word
...
    size_t len = word_len + 1;        // inner: shadows outer len
```
The shadow compiles cleanly but defeats tools that detect unintentional shadowing. Rename the inner variable to `str_copy_len`.

---

### M-5 — `gemini_tokenizer.c` `tokenizer_get_id`: unbounded TST traversal
No iteration limit on the TST walk loop. A corrupted `node_count` or `mid/left/right` cycle after a bad `tokenizer_load` could loop forever. Add a loop counter capped at `MAX_TRIE_NODES`.

---

### M-6 — `gemini_phrase_seed.c` `inject_phrase`: float-to-int weight conversion can overflow
```c
int calls = (int)(weight_calls + 0.5f);
```
If `base_weight * e->bonus_multiplier` > `INT_MAX` (e.g. a caller passes `base_weight = 1e10`), the cast is UB. Clamp first: `if (weight_calls > (float)INT_MAX) weight_calls = (float)INT_MAX;`

---

### M-7 — `gemini_attest.c` `ht_insert`: silent data loss on full table
```c
/* Table full — should not happen given ht_size = 2 × count */
```
Under adversarial hash inputs (all words with the same `hash_fnv1a` modulo `ht_size`), the table can fill before `ht_used == word_count`. The silent return drops words from the hash table while they are still in the Bloom filter, producing false negatives on `gemini_attest_db_contains` (Bloom says "maybe present"; hash says "absent"; function returns `false` — a false negative). Add a `fprintf(stderr, ...)` and consider Robin Hood probing or chaining.

---

## LOW Issues

### L-1 — `atomic_fetch_add_float`: `backoff_yields` counter semantics
In the fast path, every loop iteration increments `backoff_yields` even though no `sched_yield()` has been called. The comment says "Count initial loop iterations as yields" which is misleading and inflates the threshold. Rename to `iterations` and keep a separate `yields` counter.

### L-2 — `bridge_engine.c` `dawg_predict_multi_hop`: entropy threshold `2.0` is a magic constant
`if (new_entropy < 2.0f || depth < 2)` — the value `2.0` has no symbolic name or documentation. If weights are ever re-normalised to true probabilities the meaning changes. Define as `#define MULTI_HOP_ENTROPY_CAP 2.0f` with a comment.

### L-3 — `gemini_eval.c` `gemini_eval_save_baseline` / `gemini_eval_load_baseline`: raw struct serialization
`fwrite(r, sizeof(GeminiEvalReport), 1, fp)` will silently produce unreadable files if the struct ever gains a pointer member or if the binary is moved to a machine with different endianness/padding. Use explicit field-by-field serialization.

### L-4 — `bridge_engine.h` `dawg_predict` / `dawg_predict_multi_hop`: `predictions[][256]` fixed column width
The 256-byte column is a hidden ABI constraint. If callers allocate smaller columns and pass the raw array, writes will overflow. Document the minimum column size with `#define DAWG_PREDICTION_MAX_LEN 256` and validate `max_predictions == 0` returning early.

---

## Concurrency Contract Summary (for documentation)

| Function | Caller thread | Engine thread concurrent? | Notes |
|---|---|---|---|
| `get_scc_count` / `le_scc_mean_turbulence` | User | No (epoch boundary only) | `member_count` / `head` reads are racy if called mid-epoch |
| `le_set_scc_forced` / `le_set_scc_candidate` | User | No | Plain writes to bool fields |
| `gemini_attest_promote` | User | No | Writes `is_candidate`; must be called between epochs |
| `dawg_predict` / `engine_infer_text` | User | Yes | DAWG read-only; safe with QSBR reader registration |
| `engine_ingest_text` | Engine | — | Single-threaded ingestion path |
