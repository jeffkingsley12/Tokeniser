/*
===============================================================================
GEMINI POOLED ADJACENCY ALLOCATOR
Phase 2A Implementation: Dynamic edge pool management for O(V + E) scaling

Architecture:
  - Global flat array (edge_pool.edges) stores all inbound edges
  - Each node references a span via EdgeSpan descriptor (offset + capacity +
count)
  - Bump allocator for O(1) allocation during ingestion
  - 1.5x growth factor for resizing (amortized O(1) insertion)
  - mmap-relocatable: offsets are indices, not pointers

Status: Production-grade with comprehensive error handling
===============================================================================
*/

#include "gemini_internal.h"
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/**
 * Initialize the flat edge pool with an initial base capacity.
 */
int edge_pool_init(EngineContext *ctx, uint32_t pool_size) {
  if (!ctx || pool_size == 0)
    return -1;

  ctx->edge_pool.edges = (NodeID *)calloc(pool_size, sizeof(NodeID));
  if (!ctx->edge_pool.edges) {
    fprintf(stderr,
            "ERROR: Failed to allocate flat edge pool array of size %u\n",
            pool_size);
    return -1;
  }

  ctx->edge_pool.capacity = pool_size;
  atomic_store_explicit(&ctx->edge_pool.pool_ptr, 0, memory_order_relaxed);

  if (pthread_rwlock_init(&ctx->edge_pool.pool_lock, NULL) != 0) {
    fprintf(stderr, "ERROR: Failed to initialize edge pool rwlock\n");
    free(ctx->edge_pool.edges);
    ctx->edge_pool.edges = NULL;
    return -1;
  }

  return 0;
}

/*
 * edge_pool_expand_writelocked — grow the flat edge array.
 *
 * MUST be called with pool_lock held for WRITING by the caller.
 * Does NOT acquire or release pool_lock itself.
 *
 * Returns 0 on success, -1 on OOM.
 *
 * FIX (Bug 2 — deadlock): The original edge_pool_allocate acquired the write
 * lock internally.  le_add_incoming_edge_pooled's realloc path already held the
 * write lock and called edge_pool_allocate, causing a deadlock.  Separating the
 * "expand" logic into this _writelocked helper lets the realloc path call it
 * directly (without re-locking), while edge_pool_allocate's own slow path can
 * still lock → call → unlock cleanly.
 */
static int edge_pool_expand_writelocked(EngineContext *ctx,
                                        uint32_t min_extra) {
  uint32_t current_ptr =
      atomic_load_explicit(&ctx->edge_pool.pool_ptr, memory_order_relaxed);
  uint32_t needed = current_ptr + min_extra;

  if (needed <= ctx->edge_pool.capacity) {
    return 0; /* Another thread already expanded — nothing to do */
  }

  uint32_t new_capacity = ctx->edge_pool.capacity * 3 / 2;
  if (new_capacity < needed) {
    new_capacity = needed + 1024;
  }

  NodeID *new_edges =
      (NodeID *)realloc(ctx->edge_pool.edges, new_capacity * sizeof(NodeID));
  if (!new_edges) {
    fprintf(stderr,
            "CRITICAL OOM: Unable to expand edge pool from %u to %u items\n",
            ctx->edge_pool.capacity, new_capacity);
    return -1;
  }

  /* Initialize the newly appended region to INVALID (0xFFFFFFFF) instead of 0
   * to prevent treating uninitialized slots as valid edges to node 0 */
  memset(new_edges + ctx->edge_pool.capacity, 0xFF,
         (new_capacity - ctx->edge_pool.capacity) * sizeof(NodeID));

  ctx->edge_pool.edges = new_edges;
  ctx->edge_pool.capacity = new_capacity;
  return 0;
}

/**
 * Allocate a contiguous span in the global flat edge pool.
 * Implements a dual-path strategy:
 * - Fast path: Optimistic atomic bump under a shared read-lock.
 * - Slow path: Double-checked exclusive write-lock to safely realloc the flat
 * array.
 *
 * The read lock is held across the bump allocation AND the capacity check so
 * that no concurrent realloc (write lock) can move the edges pointer between
 * the check and a subsequent write.  Callers that write into the returned span
 * must also hold pool_lock (at least rdlock) for the same reason — see the
 * contract in the header comment on EdgePoolState.
 */
/**
 * Allocate a contiguous span in the global flat edge pool.
 * Implements a dual-path strategy:
 * - Fast path: Lock-free atomic CAS under a shared read-lock.
 * - Slow path: Double-checked exclusive write-lock to safely realloc the flat
 * array.
 */
uint32_t edge_pool_allocate(EngineContext *ctx, uint32_t capacity) {
  if (!ctx || capacity == 0)
    return (uint32_t)-1;

  /* Fast path: CAS loop under shared read-lock */
  pthread_rwlock_rdlock(&ctx->edge_pool.pool_lock);

  uint32_t expected =
      atomic_load_explicit(&ctx->edge_pool.pool_ptr, memory_order_relaxed);

  /* Only attempt the CAS if there is enough space */
  while (expected + capacity <= ctx->edge_pool.capacity) {
    if (atomic_compare_exchange_weak_explicit(
            &ctx->edge_pool.pool_ptr, &expected, expected + capacity,
            memory_order_relaxed, memory_order_relaxed)) {
      /* Success: we safely claimed the slots without exceeding capacity */
      pthread_rwlock_unlock(&ctx->edge_pool.pool_lock);
      return expected;
    }
    /* If CAS failed, 'expected' is automatically updated to the new pool_ptr.
     * Loop retries. */
  }

  /* Capacity filled — fall through to slow path */
  pthread_rwlock_unlock(&ctx->edge_pool.pool_lock);

  /* Slow path: expand under exclusive write-lock, then retry */
  pthread_rwlock_wrlock(&ctx->edge_pool.pool_lock);

  if (edge_pool_expand_writelocked(ctx, capacity) != 0) {
    pthread_rwlock_unlock(&ctx->edge_pool.pool_lock);
    return (uint32_t)-1;
  }

  /* We now hold the exclusive wrlock. No other thread can be in the fast path
   * or expanding the pool. A simple fetch_add is safe here. */
  uint32_t offset = atomic_fetch_add_explicit(&ctx->edge_pool.pool_ptr,
                                              capacity, memory_order_relaxed);

  pthread_rwlock_unlock(&ctx->edge_pool.pool_lock);
  return offset;
}

/**
 * Shut down the flat edge pool and free all associated memory maps/heap
 * buffers.
 */
void edge_pool_destroy(EngineContext *ctx) {
  if (!ctx)
    return;

  pthread_rwlock_wrlock(&ctx->edge_pool.pool_lock);
  if (ctx->edge_pool.edges) {
    free(ctx->edge_pool.edges);
    ctx->edge_pool.edges = NULL;
  }
  ctx->edge_pool.capacity = 0;
  atomic_store_explicit(&ctx->edge_pool.pool_ptr, 0, memory_order_relaxed);
  pthread_rwlock_unlock(&ctx->edge_pool.pool_lock);

  pthread_rwlock_destroy(&ctx->edge_pool.pool_lock);
}

/**
 * Report flat edge pool statistics for tracing and diagnostics.
 */
void edge_pool_stats(EngineContext *ctx) {
  if (!ctx)
    return;

  pthread_rwlock_rdlock(&ctx->edge_pool.pool_lock);
  uint32_t total_capacity = ctx->edge_pool.capacity;
  uint32_t total_used =
      atomic_load_explicit(&ctx->edge_pool.pool_ptr, memory_order_acquire);
  pthread_rwlock_unlock(&ctx->edge_pool.pool_lock);

  float utilization =
      (total_capacity > 0) ? (100.0f * total_used / total_capacity) : 0.0f;

  fprintf(stdout, "=== Gemini Edge Pool Status ===\n");
  fprintf(stdout, "  Layout Strategy : Flat mmap-relocatable array\n");
  fprintf(stdout, "  Total Capacity  : %u entries\n", total_capacity);
  fprintf(stdout, "  Allocated Edges : %u entries\n", total_used);
  fprintf(stdout, "  Pool Utilization: %.2f%%\n", utilization);
  fprintf(stdout, "===============================\n");
}

/**
 * Initialize a node's EdgeSpan with an initial allocation.
 * Called once per node during the ingestion phase to establish its edge
 * storage.
 */
int edge_span_init(EngineContext *ctx, NodeID node_id,
                   uint32_t initial_capacity) {
  if (!ctx || node_id >= MAX_NODES) {
    fprintf(stderr, "ERROR: edge_span_init invalid node_id %u\n", node_id);
    return -1;
  }

  if (initial_capacity == 0) {
    fprintf(stderr, "ERROR: edge_span_init zero capacity for node %u\n",
            node_id);
    return -1;
  }

  NodeTransientState *tn = &ctx->transient_nodes[node_id];

  /* Allocate initial span */
  uint32_t offset = edge_pool_allocate(ctx, initial_capacity);
  if (offset == (uint32_t)-1) {
    fprintf(stderr,
            "ERROR: edge_span_init pool exhausted for node %u (capacity %u)\n",
            node_id, initial_capacity);
    return -12; /* -ENOMEM */
  }

  /* Initialize EdgeSpan descriptor */
  atomic_store_explicit(&tn->in_edges.pool_offset, offset,
                        memory_order_relaxed);
  atomic_store_explicit(&tn->in_edges.capacity, initial_capacity,
                        memory_order_relaxed);
  atomic_store_explicit(&tn->in_edges.count, 0, memory_order_release);

  return 0;
}

/**
 * Add an inbound edge to a node's edge set, reallocating if necessary.
 * Implements vector-like amortized O(1) insertion with 1.5x growth.
 *
 * Thread safety — three bugs fixed:
 *
 * FIX Bug 1 (write ordering): The original fast path incremented span->count
 * via CAS and then wrote the edge datum.  A concurrent reader that acquired
 * count (load-acquire) before the write committed would read an uninitialised
 * slot.  Now we write the datum first (store-release into the pool) and
 * then publish the count increment with a store-release so readers see the
 * data before they see the higher count.
 *
 * FIX Bug 3 (use-after-free): The fast path wrote to ctx->edge_pool.edges[]
 * without holding any lock.  A concurrent slow-path thread could call
 * realloc (under write lock), move the backing array, and free the old
 * pointer while the fast-path write was still in flight.  Now the fast path
 * holds pool_lock rdlock from CAS through the store, matching the contract
 * that writers hold pool_lock and the write lock is exclusive for realloc.
 *
 * FIX Bug 2 (deadlock): The realloc path held pool_lock wrlock and then called
 * edge_pool_allocate(), which internally tried to acquire pool_lock rdlock —
 * deadlock.  Now the realloc path calls edge_pool_expand_writelocked()
 * directly (which assumes the write lock is already held) and then does its
 * own pool_ptr bump atomically.
 */
int le_add_incoming_edge_pooled(EngineContext *ctx, NodeID target,
                                NodeID source) {
  if (!ctx || target >= MAX_NODES) {
    fprintf(stderr, "ERROR: le_add_incoming_edge_pooled invalid target %u\n",
            target);
    return -1;
  }

  NodeTransientState *tn = &ctx->transient_nodes[target];
  EdgeSpan *span = &tn->in_edges;

  uint32_t capacity =
      atomic_load_explicit(&span->capacity, memory_order_relaxed);
  if (capacity == 0) {
    fprintf(stderr,
            "ERROR: Edge span not initialized for node %u (capacity=0)\n",
            target);
    return -1;
  }

  /* ── CASE 1: Fast path — space available ─────────────────────────────────
   *
   * FIX H-1: The previous implementation used CAS to set span->count = idx+1
   * BEFORE writing the datum to edges[offset+idx]. A concurrent reader that
   * acquire-loaded count and saw idx+1 could then load edges[offset+idx]
   * before the release store of the datum, observing uninitialized data.
   *
   * Corrected ordering (producer side):
   * 1. Load idx = span->count with acquire to establish a snapshot.
   * 2. Write datum to edges[offset+idx] with RELEASE, creating a
   * happens-before edge with any subsequent acquire of count.
   * 3. CAS count from idx to idx+1 with acq_rel.  Because the datum store
   * is sequenced-before the CAS in this thread, the CAS's release fence
   * covers the datum store.  Any reader who acquire-loads count and sees
   * idx+1 is guaranteed to see the datum.
   * 4. On CAS failure another thread claimed slot idx; zero the datum
   * (restore invariant) and retry from step 1.
   *
   * Hold pool_lock (rdlock) across the whole operation so that a concurrent
   * slow-path realloc (which requires wrlock) cannot move
   * ctx->edge_pool.edges between our datum write and our CAS.
   */
  pthread_rwlock_rdlock(&ctx->edge_pool.pool_lock);

  /* Re-read capacity while holding the lock */
  capacity = atomic_load_explicit(&span->capacity, memory_order_acquire);
  uint32_t current_count =
      atomic_load_explicit(&span->count, memory_order_acquire);

  if (current_count < capacity) {
    uint32_t idx;
    bool claimed = false;

    /* FIX (Issue #5): Use CAS-first ordering — claim the slot index atomically
     * BEFORE writing the datum. The original write-then-CAS pattern had two
     * threads writing different source values to the same slot before one CAS
     * succeeded. The winning thread's slot could then be overwritten by the
     * losing thread's store issued after the losing CAS. The INVALID-restore
     * "fix" made this worse: it overwrote the winner's published datum with
     * INVALID after the winner had already incremented count.
     *
     * With CAS-first, each thread atomically claims an exclusive slot index
     * before writing to it, so no two threads ever write to the same slot.
     *
     * Reader visibility gap: between the CAS (count increment) and the datum
     * store, a concurrent reader could load count=idx+1 and read slot[idx]=INVALID
     * (since slots are 0xFF-initialised). All reader loops must skip INVALID
     * entries or spin briefly. This is safe because INVALID (0xFFFFFFFF) is
     * never a valid NodeID (valid IDs are < MAX_NODES < 0xFFFFFFFF). */
    do {
      idx = atomic_load_explicit(&span->count, memory_order_acquire);
      if (idx >= capacity)
        break;

      uint32_t expected_idx = idx;
      claimed = atomic_compare_exchange_weak_explicit(
          &span->count, &expected_idx, idx + 1,
          memory_order_acq_rel, memory_order_acquire);
      /* On CAS failure expected_idx is updated to the current count; retry. */
    } while (!claimed);

    if (claimed) {
      /* We exclusively own slot idx. Write datum with release so any reader
       * that acquire-loads count > idx is guaranteed to see the datum. */
      uint32_t offset =
          atomic_load_explicit(&span->pool_offset, memory_order_relaxed);
      atomic_store_explicit(
          (_Atomic NodeID *)&ctx->edge_pool.edges[offset + idx], source,
          memory_order_release);
      pthread_rwlock_unlock(&ctx->edge_pool.pool_lock);
      return 0;
    }
    /* capacity filled by competing thread — fall through to slow path */
  }

  pthread_rwlock_unlock(&ctx->edge_pool.pool_lock);

  /* ── CASE 2: Slow path — realloc under exclusive write lock ──────────────
   *
   * No need to call edge_pool_allocate() here; that function acquires
   * pool_lock internally, which would deadlock since we are about to hold
   * the write lock ourselves (Bug 2).  Instead we bump pool_ptr directly
   * after calling edge_pool_expand_writelocked().
   */
  pthread_rwlock_wrlock(&ctx->edge_pool.pool_lock);

  /* Re-read under lock — another thread may have grown the span already */
  uint32_t old_capacity =
      atomic_load_explicit(&span->capacity, memory_order_acquire);
  current_count = atomic_load_explicit(&span->count, memory_order_acquire);

  if (current_count < old_capacity) {
    /* Span was extended by a competing thread while we waited for wrlock */
    uint32_t old_offset =
        atomic_load_explicit(&span->pool_offset, memory_order_acquire);
    ctx->edge_pool.edges[old_offset + current_count] = source;
    atomic_store_explicit(&span->count, current_count + 1,
                          memory_order_release);
    pthread_rwlock_unlock(&ctx->edge_pool.pool_lock);
    return 0;
  }

  /* Compute new capacity with 1.5x growth, guarding against overflow */
  uint32_t new_capacity;
  if (old_capacity > UINT32_MAX / 3 * 2) {
    new_capacity = old_capacity + 1024;
  } else {
    new_capacity = (old_capacity * 3) / 2;
  }
  if (new_capacity <= old_capacity)
    new_capacity = old_capacity + 1024;

  /* FIX: Remove the legacy MAX_IN_EDGES slab cap. The only theoretical bound
   * for a single node's dynamic in-degree is the global MAX_EDGES. */
  if (new_capacity > MAX_EDGES) {
    fprintf(stderr, "ERROR: Node %u edge count exceeds global MAX_EDGES (%u)\n",
            target, MAX_EDGES);
    pthread_rwlock_unlock(&ctx->edge_pool.pool_lock);
    return -12;
  }

  /* Ensure the flat pool has enough room for the new span */
  if (edge_pool_expand_writelocked(ctx, new_capacity) != 0) {
    fprintf(stderr, "ERROR: Pool expansion failed during realloc for node %u\n",
            target);
    pthread_rwlock_unlock(&ctx->edge_pool.pool_lock);
    return -12;
  }

  /* Bump pool_ptr directly (we hold wrlock; no other allocator is running) */
  uint32_t new_offset = atomic_fetch_add_explicit(
      &ctx->edge_pool.pool_ptr, new_capacity, memory_order_relaxed);

  /* Copy valid existing edges into the new region, compacting out INVALID slots
   */
  uint32_t valid_count = 0;
  if (current_count > 0) {
    uint32_t old_offset =
        atomic_load_explicit(&span->pool_offset, memory_order_acquire);
    for (uint32_t i = 0; i < current_count; i++) {
      NodeID edge = ctx->edge_pool.edges[old_offset + i];
      if (edge != INVALID) {
        ctx->edge_pool.edges[new_offset + valid_count++] = edge;
      }
    }
  }

  /* Append new edge and publish the updated descriptor atomically */
  ctx->edge_pool.edges[new_offset + valid_count] = source;
  current_count = valid_count + 1;

  atomic_store_explicit(&span->pool_offset, new_offset, memory_order_release);
  atomic_store_explicit(&span->capacity, new_capacity, memory_order_release);
  atomic_store_explicit(&span->count, current_count, memory_order_release);

  pthread_rwlock_unlock(&ctx->edge_pool.pool_lock);
  return 0;
}