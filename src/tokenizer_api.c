/*
 * tokenizer_api.c
 *
 * Flat C API for FFI consumers (Python ctypes/cffi, Rust, Go, Java JNI).
 * All four public functions are safe to call from multiple threads.
 *
 * Fix C-3: added pthread_mutex_t around the g_handles[] table.
 * Note: tok_encode / tok_decode are NOT locked after the handle is
 * validated — the underlying tokenizer_encode / tokenizer_decode are
 * read-only with respect to the Tokenizer struct and are themselves
 * thread-safe.  Only the table bookkeeping (tok_load / tok_free) needs
 * the lock.
 */

#include "tokenizer_api.h"
#include "tokenizer.h"

#include <pthread.h>
#include <stdio.h>
#include <sched.h>

#include <stdatomic.h>

/* ── Handle table ─────────────────────────────────────────────────────────── */

#define MAX_HANDLES 16

/* C-4: Atomic pointers prevent torn reads on non-x86 platforms.
 * M-4: Generation counter detects stale handles after recycling.
 *       Public handle = (generation << 8) | slot.  Only slot bits are used
 *       for the array index; generation is validated on every access.
 * C-5: Active reader counter prevents use-after-free during concurrent tok_free. */
typedef struct {
    _Atomic(MmapTokenizer *) ptr;
    atomic_uint              generation;   /* incremented on free + on alloc */
    atomic_int               active_readers; /* reader lock for safe teardown */
} HandleSlot;

static HandleSlot       g_slots[MAX_HANDLES];
static pthread_mutex_t  g_lock = PTHREAD_MUTEX_INITIALIZER;

/* Encode slot + generation into a single int handle. */
static inline int make_handle(int slot, unsigned gen) {
    return (int)(((unsigned)gen << 8) | (unsigned)slot);
}
static inline int handle_slot(int handle) { return handle & 0xFF; }
static inline unsigned handle_gen(int handle)  { return (unsigned)handle >> 8; }

/* ── tok_load ─────────────────────────────────────────────────────────────── */

int tok_load(const char *path) {
    if (!path) return -1;

    /*
     * Load the tokenizer BEFORE taking the lock so that slow I/O
     * (mmap, CRC, trie rebuild) does not block other threads from calling
     * tok_encode / tok_decode on existing handles.
     */
    MmapTokenizer *mt = tokenizer_load_mmap(path);
    if (!mt) return -1;

    pthread_mutex_lock(&g_lock);
    int slot = -1;
    for (int i = 0; i < MAX_HANDLES; i++) {
        if (!atomic_load_explicit(&g_slots[i].ptr, memory_order_relaxed)) {
            unsigned gen = atomic_fetch_add_explicit(
                               &g_slots[i].generation, 1u, memory_order_relaxed) + 1u;
            atomic_store_explicit(&g_slots[i].ptr, mt, memory_order_release);
            slot = make_handle(i, gen);
            break;
        }
    }
    pthread_mutex_unlock(&g_lock);

    if (slot < 0) {
        /* No free slot — destroy the tokenizer we just loaded. */
        tokenizer_destroy_mmap(mt);
        fprintf(stderr,
                "[tok_api] Maximum handle count (%d) reached; "
                "free an existing handle with tok_free() first.\n",
                MAX_HANDLES);
    }

    return slot;
}

/* ── tok_encode ───────────────────────────────────────────────────────────── */

int tok_encode(int handle, const char *text, uint32_t *out, int cap) {
    /* Validate inputs before touching the handle table. */
    if (handle < 0) return -1;
    if (!text || !out || cap <= 0)            return -1;

    int slot = handle_slot(handle);
    if (slot < 0 || slot >= MAX_HANDLES) return -1;

    atomic_fetch_add_explicit(&g_slots[slot].active_readers, 1, memory_order_acquire);

    /* C-4: atomic acquire load — safe against concurrent tok_free.
     * M-4: validate generation to catch stale handles after recycling. */
    unsigned expected_gen = handle_gen(handle);
    unsigned current_gen  = atomic_load_explicit(
                                &g_slots[slot].generation, memory_order_acquire);
    if (current_gen != expected_gen) {
        atomic_fetch_sub_explicit(&g_slots[slot].active_readers, 1, memory_order_release);
        return -1;   /* stale handle */
    }

    MmapTokenizer *mt = atomic_load_explicit(&g_slots[slot].ptr, memory_order_acquire);
    if (!mt) {
        atomic_fetch_sub_explicit(&g_slots[slot].active_readers, 1, memory_order_release);
        return -1;
    }

    int n = tokenizer_encode(&mt->base, text, out, (uint32_t)cap);
    if (n > 0) {
        atomic_fetch_add_explicit(&mt->total_tokens_encoded, (uint64_t)n, memory_order_relaxed);
    }

    atomic_fetch_sub_explicit(&g_slots[slot].active_readers, 1, memory_order_release);
    return n;
}

/* ── tok_decode ───────────────────────────────────────────────────────────── */

const char *tok_decode(int handle, uint32_t id) {
    if (handle < 0) return NULL;
    int slot = handle_slot(handle);
    if (slot < 0 || slot >= MAX_HANDLES) return NULL;

    atomic_fetch_add_explicit(&g_slots[slot].active_readers, 1, memory_order_acquire);

    unsigned expected_gen = handle_gen(handle);
    if (atomic_load_explicit(&g_slots[slot].generation, memory_order_acquire)
            != expected_gen) {
        atomic_fetch_sub_explicit(&g_slots[slot].active_readers, 1, memory_order_release);
        return NULL;
    }

    MmapTokenizer *mt = atomic_load_explicit(&g_slots[slot].ptr, memory_order_acquire);
    if (!mt) {
        atomic_fetch_sub_explicit(&g_slots[slot].active_readers, 1, memory_order_release);
        return NULL;
    }

    const char *result = tokenizer_decode(&mt->base, id);
    atomic_fetch_sub_explicit(&g_slots[slot].active_readers, 1, memory_order_release);
    return result;
}

/* ── tok_free ─────────────────────────────────────────────────────────────── */

void tok_free(int handle) {
    if (handle < 0) return;
    int slot = handle_slot(handle);
    if (slot < 0 || slot >= MAX_HANDLES) return;

    pthread_mutex_lock(&g_lock);
    /* Validate generation so double-free is a no-op. */
    unsigned expected_gen = handle_gen(handle);
    unsigned current_gen  = atomic_load_explicit(
                                &g_slots[slot].generation, memory_order_relaxed);
    if (current_gen != expected_gen) {
        pthread_mutex_unlock(&g_lock);
        return;  /* already freed or never allocated */
    }
    MmapTokenizer *mt = atomic_load_explicit(&g_slots[slot].ptr, memory_order_relaxed);
    /* Bump generation BEFORE releasing pointer so new readers see an invalid gen. */
    atomic_fetch_add_explicit(&g_slots[slot].generation, 1u, memory_order_release);
    atomic_store_explicit(&g_slots[slot].ptr, NULL, memory_order_release);
    pthread_mutex_unlock(&g_lock);

    /* Drain active readers safely before tearing down memory */
    while (atomic_load_explicit(&g_slots[slot].active_readers, memory_order_acquire) > 0) {
        sched_yield();
    }

    if (mt) tokenizer_destroy_mmap(mt);
}

/* ── Metrics Accessors ────────────────────────────────────────────────────── */

uint32_t tok_vocab_size(int handle) {
    if (handle < 0) return 0;
    int slot = handle_slot(handle);
    if (slot < 0 || slot >= MAX_HANDLES) return 0;
    if (atomic_load_explicit(&g_slots[slot].generation, memory_order_acquire)
            != handle_gen(handle)) return 0;
    atomic_fetch_add_explicit(&g_slots[slot].active_readers, 1, memory_order_acquire);
    MmapTokenizer *mt = atomic_load_explicit(&g_slots[slot].ptr, memory_order_acquire);
    uint32_t result = mt ? mt->base.vocab_size : 0;
    atomic_fetch_sub_explicit(&g_slots[slot].active_readers, 1, memory_order_release);
    return result;
}

uint64_t tok_tokens_encoded(int handle) {
    if (handle < 0) return 0;
    int slot = handle_slot(handle);
    if (slot < 0 || slot >= MAX_HANDLES) return 0;
    if (atomic_load_explicit(&g_slots[slot].generation, memory_order_acquire)
            != handle_gen(handle)) return 0;
    atomic_fetch_add_explicit(&g_slots[slot].active_readers, 1, memory_order_acquire);
    MmapTokenizer *mt = atomic_load_explicit(&g_slots[slot].ptr, memory_order_acquire);
    uint64_t result = mt ? atomic_load_explicit(&mt->total_tokens_encoded, memory_order_relaxed) : 0;
    atomic_fetch_sub_explicit(&g_slots[slot].active_readers, 1, memory_order_release);
    return result;
}
