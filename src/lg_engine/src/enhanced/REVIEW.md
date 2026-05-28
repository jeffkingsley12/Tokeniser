# Gemini Engine — Comprehensive Code Review

> Files reviewed: `bridge_engine.c/h`, `gemini_engine.c`, `gemini_pool.c`,
> `gemini_tokenizer.c`, `gemini_accessors.c`, `gemini_phrase_seed.c`,
> `gemini_enhanced.c/h`, `gemini_internal.h`
>
> Severity scale: **CRITICAL** (data corruption / crash) → **HIGH** (race / leak) →
> **MEDIUM** (logic / perf) → **LOW** (style / robustness)

---

## CRITICAL Bugs

### C-1 · Stack buffer overflow in `engine_infer_distributed` — `bridge_engine.c:678`

**Location:** `engine_infer_distributed`, the *resonance* (alternative-path) `!found` branch.

**Problem:**
```c
// First !found block — correctly guarded:
if (!found && next_count < MAX_ACTIVE_STATES) {   // ← guard present
    next_states[next_count++] = ...;
}

// Second !found block — guard MISSING:
if (!found) {                                       // ← NO guard
    next_states[next_count].node = alt_next;        // overflow when next_count == 32
    next_states[next_count].activation = resonance;
    next_count++;
}
```
`next_states[]` is `ActiveState next_states[MAX_ACTIVE_STATES]` (32 elements on the stack).
The outer `while` condition checks `next_count < MAX_ACTIVE_STATES` at the **top** of each
iteration, but by the time execution reaches the inner `!found` branch the count may already
be 32 (filled by the DAWG-transition path earlier in the same iteration). The unguarded write
overflows the stack frame, corrupting the return address and adjacent locals. On a 64-bit
build with 32 active states and 32 resonance candidates, this fires consistently.

**Fix:** Add `next_count < MAX_ACTIVE_STATES` to every unconditional `!found` write.

---

### C-2 · `le_get_or_create_node` leaks EdgeSpan pool capacity on CAS loss — `gemini_engine.c:683–709`

**Problem:**
```c
if (edge_span_init(ctx, id, 4) != 0) { ... }  // bumps pool_ptr by 4

NodeID expected = INVALID;
if (!atomic_compare_exchange_strong(&ctx->node_hash[h], &expected, id)) {
    atomic_fetch_sub(&ctx->node_count, 1);      // node slot reclaimed ✓
    // edge_span pool slots NOT reclaimed       ← LEAK
    return winner_node;
}
```
Under contention (two threads racing to insert the same token), the losing thread decrements
`node_count` but the four pool slots consumed by `edge_span_init` are permanently wasted.
At a moderate ingestion rate of 10 k tokens/s with 1 % collision probability, the pool
exhausts approximately 30 % sooner than expected. Under high concurrency the pool drains
entirely, returning `INVALID` for all subsequent node creations and halting the engine.

**Fix:** Before returning after a lost CAS, roll back `edge_pool.pool_ptr` by the
initial capacity (4) using `atomic_fetch_sub`. This is safe because the losing thread
uniquely owns those slots — no other thread has been given that offset range.

---

### C-3 · `gemini_enhanced_load` leaks mmap'd context on vocab mismatch — `gemini_enhanced.c`

**Problem:**
```c
le_destroy(ge->ctx);             // destroys the freshly-init'd ctx
ge->ctx = le_load_mmap(path, false);  // returns a valid mmap'd EngineContext
if (!ge->ctx) { ... return NULL; }

// vocab mismatch:
ge->ctx = NULL;                  // sets ge->ctx to NULL
gemini_enhanced_destroy(ge);     // calls le_destroy(NULL) → no-op
// The mmap region + fd from le_load_mmap are NEVER released
```
The mmap'd `EngineContext` (potentially gigabytes) and its open file descriptor are leaked
every time a mismatched snapshot is loaded. In server environments that hot-reload models on
rotation, this exhausts virtual address space and file descriptors.

**Fix:** Store the pointer, set `ge->ctx = NULL`, call `le_destroy(stored_ptr)` **before**
calling `gemini_enhanced_destroy`.

---

## HIGH Bugs

### H-1 · `le_add_incoming_edge_pooled` fast path: count published before data written — `gemini_pool.c:278–306`

**Problem:** The CAS atomically sets `span->count = idx+1` (with `acq_rel`), then the datum
is written to `edges[offset+idx]` with a separate `release` store. A concurrent reader that
does an `acquire` load of `count`, sees `idx+1`, then loads `edges[offset+idx]` will
observe **uninitialized** edge data — the acq/rel pair on `count` does **not** imply
visibility of a subsequent store to a different address (`edges[offset+idx]`).

The correct ordering is: **write datum first (release), then publish count (release CAS)**.
Because the datum store is sequenced-before the CAS in the writer's thread, the CAS's
release fence covers the datum, and any reader's acquire-load of count that sees `idx+1`
is guaranteed to also see the datum.

**Fix:** Swap the order: load `idx = count` (acquire), write data with release, CAS count
from `idx` to `idx+1` with `acq_rel`. Retry on CAS failure (another thread stole the slot).

---

### H-2 · Non-atomic `edge_nexts[]` reads in hot traversal paths — `gemini_engine.c`

**Locations:** `apply_lazy_decay:206`, `compactor_sweep:292`, `le_recompute_all_turbulence:2843/2858`,
`le_split_sccs:1374`, `le_update_dawg_transitions:2628,2636`.

**Problem:** `ctx->edge_nexts` is a plain `EdgeID[]` array. `le_free_edge` writes
`edge_nexts[edge_id] = head_tp.ptr` and `le_add_edge_bulk` writes
`ctx->edge_nexts[new_edge] = old_head` without any atomic. Under C11, a concurrent
non-atomic write and non-atomic read on the same memory location is **undefined behavior**
regardless of hardware memory ordering. On x86 this is benign in practice, but on ARM/POWER
torn reads can produce a garbage `edge_nexts` index, causing traversal to jump into
unallocated edges or infinite-loop.

Similarly, `le_update_dawg_transitions` writes `t->next = ...` and reads `t_idx = t->next`
as plain accesses on `_Atomic uint32_t` fields — the assignment bypasses the atomic
machinery and is UB.

**Fix:** Replace all `edge_nexts[e]` reads/writes with `atomic_load_explicit` /
`atomic_store_explicit` calls (`memory_order_acquire` / `memory_order_release`). Declare
`edge_nexts` as `_Atomic EdgeID *`. For `DawgTransition.next`, use
`atomic_store_explicit(&t->next, val, memory_order_release)` everywhere it is written.

---

### H-3 · QSBR grace-period wait inside per-node compaction loop — `gemini_engine.c:344`

**Problem:**
```c
for (uint32_t i = 0; i < node_count; i++) {
    ...
    if (atomic_compare_exchange_strong(...)) {
        while (!is_grace_period_over(ctx, my_epoch)) {
            nanosleep(...);   // up to 10 s per node
        }
        // free old edges
    }
}
```
The QSBR wait blocks for up to 10 seconds **per pruned node**. With `MAX_NODES = 2 000 000`
and a moderate prune rate, a single `compactor_sweep` can block for hours, stalling the
engine thread entirely. All reader threads that arrived after the CAS are waiting for the
compactor to advance, creating a deadlock-like livelock.

**Fix:** Push freed chains onto a deferred-free list (an array of `(old_head, epoch)` pairs).
After the node iteration loop completes, do **one** QSBR wait and then free every chain on
the list. This reduces QSBR overhead from O(pruned_nodes × wait_time) to O(wait_time).

---

### H-4 · `mark_scc_forced` writes plain `bool` without synchronization — `gemini_phrase_seed.c:75`

**Problem:**
```c
ctx->scc_nodes[sid].is_forced = true;   // plain write
```
`SccNode.is_forced` is a plain `bool`. The engine thread reads `is_forced` in
`le_process_epoch_stability` and promotion paths concurrently. Under C11, a concurrent
plain write and plain read on the same object is a data race → undefined behavior.
`is_candidate` and `is_promoted` in `gemini_accessors.c` have the same problem.

**Fix:** Promote `is_forced`, `is_candidate`, and `is_promoted` to `_Atomic bool`, and use
`atomic_store_explicit(..., memory_order_release)` / `atomic_load_explicit(..., memory_order_acquire)`.
As a minimal interim fix, wrap with `__atomic_store_n` / `__atomic_load_n`.

---

### H-5 · `le_add_scc_edge` traverses `SccEdge.next` without atomic access — `gemini_engine.c:67,105`

**Problem:**
```c
e = ctx->scc_edges[e].next;   // plain read of plain uint32_t
```
`SccEdge.next` is a plain `SccEdgeID`. `le_merge_sccs` writes it (free-list push) and
`le_add_scc_edge` itself writes it (new-edge insertion), both possibly concurrent. This is
a data race. While `SccEdgeID` writes are typically pointer-sized and atomic on x86, C11
gives no guarantee and ARM/POWER can produce torn values.

**Fix:** Make `SccEdge.next` `_Atomic SccEdgeID` and use explicit atomic loads in all
traversals.

---

### H-6 · `le_get_or_create_node` post-CAS re-scan can return `INVALID` for a valid token — `gemini_engine.c:700–709`

**Problem:** After losing the CAS race, the code re-scans the hash table starting from
`h2 = token_id % HASH_SIZE`. If the winning thread placed the node at a probed slot (not
`h2`), the scan will find it. However, if the table is near-full and the winning thread
placed the node after many probes, the re-scan may terminate on an empty slot before
reaching the winner's slot (if another thread filled that slot after the CAS but before
re-scan). Under these conditions the function incorrectly returns `INVALID` for a valid
token.

**Fix:** The re-scan must use the same linear-probe logic with a full `HASH_SIZE` step
limit, identical to the primary lookup path. Verify the outer `while` in the post-CAS
branch has its own `steps` counter bounded by `HASH_SIZE`.

---

## MEDIUM Bugs

### M-1 · `le_split_sccs` first-pass DFS has O(N²) node-index lookups — `gemini_engine.c:1342,1361`

**Problem:**
```c
// Inner DFS loop — runs O(E) times total:
for (uint32_t k = 0; k < node_count; k++) {   // O(N) per edge
    if (nodes[k] == u) { u_idx = k; break; }
}
```
Both the u_idx and v_idx lookups are linear scans, making the first-pass DFS O(N·E) instead
of O(N+E). For a 10 000-node SCC with 50 000 edges, this is 500 M iterations per split —
blocking the engine thread for seconds.

**Fix:** Build a `NodeID → local_index` hash map before the DFS. A flat array
`uint32_t node_to_idx[MAX_NODES]` (initialized to `INVALID`) and filled once in O(N)
makes every lookup O(1).

---

### M-2 · `le_split_sccs` allocates `dfs_idx` but never reads it — `gemini_engine.c:1326`

**Problem:**
```c
uint32_t *dfs_idx = calloc(node_count, sizeof(uint32_t));
dfs_idx[i] = 0;   // only write
// never read; freed at line 1385
```
Dead allocation: O(N) calloc per outer DFS call for no purpose.

**Fix:** Remove the allocation and the single write. The variable served no role.

---

### M-3 · `le_update_dawg_transitions` plain writes to `_Atomic` `DawgTransition.next` — `gemini_engine.c:2628,2636`

**Problem:**
```c
t_idx = t->next;                                          // plain read of _Atomic
t->next = (k + 1 < count) ? items[k+1].t_idx : INVALID; // plain write of _Atomic
```
Both are direct reads/writes to `_Atomic uint32_t` fields without the atomic intrinsics.
The header comment for `DawgTransition.next` explicitly requires `atomic_load_explicit` /
`atomic_store_explicit`. These plain accesses defeat the acquire/release ordering that
`engine_step` and `dawg_predict` rely on when they read `t->next` with
`memory_order_acquire`.

**Fix:**
```c
t_idx = atomic_load_explicit(&t->next, memory_order_acquire);
atomic_store_explicit(&t->next, new_val, memory_order_release);
```

---

### M-4 · `le_add_scc_edge` monotonic allocator uses wrong capacity limit — `gemini_engine.c:85`

**Problem:**
```c
new_edge = atomic_fetch_add(&ctx->scc_edge_count, 1);
if (new_edge >= MAX_EDGES) {          // ← checks MAX_EDGES
    atomic_fetch_sub(&ctx->scc_edge_count, 1);
    return;
}
```
`ctx->scc_edges` is allocated as `calloc(MAX_EDGES, sizeof(SccEdge))` — so the guard
`MAX_EDGES` is correct. However, the free-list also uses indices in `[0, MAX_EDGES)`.
The monotonic allocator and the free-list together can cumulatively hand out more than
`MAX_EDGES` unique indices if the free-list is populated beyond initial bounds (e.g., after
`le_merge_sccs` returns slots and the allocator counter was not properly bounded from both
paths). A belt-and-suspenders check: also assert `new_edge < MAX_EDGES` before indexing
`ctx->scc_edges`.

---

### M-5 · `compactor_sweep` may undercount global `edge_count` — `gemini_engine.c:383`

**Problem:**
```c
uint32_t pruned = degree - keep_count;
if (pruned > 0)
    atomic_fetch_sub(&ctx->edge_count, pruned);
```
`degree` is read from `node->edge_count` **before** the CAS swap. Between the snapshot and
the CAS, concurrent `le_add_edge` calls may have inserted new edges and incremented
`edge_count`. After a successful CAS, the new edges are discarded (they are in `old_head`,
not `newest_head`), so `edge_count` is decremented by `degree - keep_count` — but the new
edges that were added and then lost aren't counted in `pruned`. This produces an undercount
in `edge_count` that widens on every sweep under high write concurrency.

**Fix:** Re-read `degree` from `node->edge_count` **after** the successful CAS and compute
`pruned` from that atomic snapshot.

---

### M-6 · `tokenizer_load` uses `malloc` leaving `string_pool` uninitialized beyond `pool_used` — `gemini_tokenizer.c:555`

**Problem:**
```c
Tokenizer* t = (Tokenizer*)malloc(sizeof(Tokenizer));
```
`t->string_pool[pool_used .. MAX_STRING_POOL-1]` is uninitialized. After load, callers
invoke `tokenizer_get_word`, which returns `&string_pool[id_to_offset[id]]`. While valid
IDs always point within `pool_used`, any future code that walks past a short string (e.g.,
a `strlen` that doesn't find a NUL before `pool_used`) could read garbage. Using `calloc`
costs ~6 MB once and eliminates the risk.

**Fix:** Change `malloc` to `calloc(1, sizeof(Tokenizer))`.

---

### M-7 · `le_entropy_guided_fission` reads `scc->avg_entropy` and `scc->coherence` without synchronization — `gemini_engine.c:1593`

**Problem:** Both fields are plain `float`s in `SccNode`. The function reads them without
any atomic. The TODO comment in `gemini_accessors.c` acknowledges this but the fission path
is not guarded by the same epoch-boundary requirement. Any concurrent call to
`le_recalculate_scc_metrics` can race with this read.

**Fix:** Either gate the fission function with the `large_scratch_mutex` (already used by
the compactor), or promote `avg_entropy` and `coherence` to `_Atomic uint32_t` with
bit-cast accessors (like `ingestion_weight_modifier`).

---

## LOW Issues

### L-1 · `gemini_phrase_seed_print` omits `nodes_created` from output

**Problem:** `PhraseSeedResult` tracks `nodes_created` but `gemini_phrase_seed_print` never
prints it, making the diagnostic incomplete.

**Fix:** Add `printf("  Nodes created     : %d\n", r->nodes_created);`.

---

### L-2 · `engine_lineage_print` prints `from_scc` as `%u` when value may be `INVALID` (0xFFFFFFFF)

**Problem:** When `from_scc` is `INVALID`, the format `"SCC %u"` prints `4294967295`,
which looks like a valid large SCC ID in logs. Inspectors may waste time investigating it.

**Fix:** Print `"SCC INVALID"` when `e->from_scc == INVALID`.

---

### L-3 · `engine_infer_distributed` normalization divides by `total_activation` which could be computed from unnormalized values

**Problem:** The normalization loop sums activations that may already contain merged
resonance values from multiple paths. If activation values exceed 1.0 due to accumulation,
the normalization suppresses valid high-confidence states below the 0.01 prune threshold.

**Fix:** Track the maximum activation and normalize to that, or prune by an absolute
threshold relative to the maximum rather than a fixed 0.01.

---

### L-4 · `le_alloc_edge` / `le_free_edge` have no protection against corrupted `edge_nexts` cycle

**Problem:** Both functions traverse `edge_nexts` without a step counter. A corrupted
pointer cycle (e.g., from a torn read on a 32-bit platform) causes an infinite loop that
cannot be interrupted, hanging the engine thread permanently.

**Fix:** Add a `steps` counter bounded by `MAX_EDGES` with a warning log and `return INVALID`
on exhaustion — identical to the fix already applied in `find_node_by_symbol`.

---

## Summary Table

| ID  | Severity | File                  | Description                                          |
|-----|----------|-----------------------|------------------------------------------------------|
| C-1 | CRITICAL | bridge_engine.c       | Stack overflow in resonance `!found` branch          |
| C-2 | CRITICAL | gemini_engine.c       | EdgeSpan pool leak on CAS loss in node creation      |
| C-3 | CRITICAL | gemini_enhanced.c     | mmap context leaked on vocab mismatch in `_load`     |
| H-1 | HIGH     | gemini_pool.c         | Count published before datum in fast-path insert     |
| H-2 | HIGH     | gemini_engine.c       | Non-atomic `edge_nexts[]` reads — UB data race       |
| H-3 | HIGH     | gemini_engine.c       | QSBR wait inside per-node loop — O(N) blocking       |
| H-4 | HIGH     | gemini_phrase_seed.c  | Plain `bool` write to `is_forced` — data race        |
| H-5 | HIGH     | gemini_engine.c       | `SccEdge.next` traversed without atomic load         |
| H-6 | HIGH     | gemini_engine.c       | Post-CAS re-scan can miss winner on near-full table  |
| M-1 | MEDIUM   | gemini_engine.c       | O(N²) node-index lookups in `le_split_sccs`          |
| M-2 | MEDIUM   | gemini_engine.c       | Dead `dfs_idx` allocation in `le_split_sccs`         |
| M-3 | MEDIUM   | gemini_engine.c       | Plain reads/writes to `_Atomic` `DawgTransition.next`|
| M-4 | MEDIUM   | gemini_engine.c       | `le_add_scc_edge` limit check uses `MAX_EDGES`       |
| M-5 | MEDIUM   | gemini_engine.c       | `edge_count` undercount after concurrent sweep       |
| M-6 | MEDIUM   | gemini_tokenizer.c    | `tokenizer_load` uses `malloc`, leaving pool uninit  |
| M-7 | MEDIUM   | gemini_engine.c       | `avg_entropy/coherence` plain reads in fission path  |
| L-1 | LOW      | gemini_phrase_seed.c  | `nodes_created` missing from print output            |
| L-2 | LOW      | bridge_engine.c       | `from_scc == INVALID` printed as large uint          |
| L-3 | LOW      | bridge_engine.c       | Activation normalization may suppress valid states   |
| L-4 | LOW      | gemini_engine.c       | No cycle guard in `le_alloc_edge` / `le_free_edge`   |
