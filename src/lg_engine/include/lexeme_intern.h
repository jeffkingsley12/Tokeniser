/*
===============================================================================
LEXEME CANONICALIZATION — Interning Identical Surface Forms
===============================================================================

Problem:
  Same word "yingira" appearing in different training contexts gets different
  token IDs, creating separate graph nodes. SCC can't merge non-cyclically-
  connected duplicates, resulting in fragmented statistics.

Solution:
  Map surface forms to canonical lexeme IDs BEFORE graph node creation.
  All "yingira" occurrences → single lexeme_id → single aggregated node.

Architecture:
  surface_form → FNV-1a hash → lexeme_id → node creation
                 (intern check)

Expected Gain:
  - 11 weak nodes (2–5 transitions) → 1 strong node (50–500 transitions)
  - Aggregated statistics improve SCC coherence
  - Symbol promotion acts on consolidated structure
  - Autocomplete quality: 3–5× improvement

Thread Safety:
  le_intern_lexeme() uses double-checked locking:
    1. Lock-free hash table lookup (common case)
    2. Fast path if found
    3. Synchronized insertion for new lexeme (rare case)

===============================================================================
*/

#ifndef LEXEME_INTERN_H
#define LEXEME_INTERN_H

#include <stdint.h>
#include <stdbool.h>
#include "gemini_internal.h"

/* Forward decl */
typedef struct EngineContext EngineContext;

/**
 * le_intern_lexeme - Canonicalize a surface form to a lexeme ID.
 *
 * All identical surface forms map to the same lexeme_id, enabling
 * aggregated statistics and proper SCC merging.
 *
 * @ctx: Engine context with lexeme registry
 * @surface_form: UTF-8 string of the token (NUL-terminated)
 *
 * Returns: Canonical lexeme ID (uint32_t, 0 = invalid)
 *
 * Thread-safe: Yes (double-checked locking)
 * Blocking: Yes (acquires lexeme_lock on insertion)
 */
uint32_t le_intern_lexeme(EngineContext *ctx, const char *surface_form);

/**
 * le_lexeme_frequency - Query total occurrence count of a lexeme.
 *
 * @ctx: Engine context
 * @lexeme_id: Canonical ID from le_intern_lexeme()
 *
 * Returns: Total count of this lexeme across all contexts
 */
uint32_t le_lexeme_frequency(EngineContext *ctx, uint32_t lexeme_id);

/**
 * le_lexeme_surface - Reverse lookup: lexeme_id → surface form.
 *
 * WARNING: Returns an internal pointer that may become dangling if the
 * lexeme registry is modified concurrently. Prefer le_lexeme_surface_safe()
 * in threaded contexts.
 *
 * @ctx: Engine context
 * @lexeme_id: Canonical ID
 *
 * Returns: NUL-terminated UTF-8 string (owned by ctx), or NULL if invalid
 */
const char* le_lexeme_surface(EngineContext *ctx, uint32_t lexeme_id);

/**
 * le_lexeme_surface_safe - Thread-safe reverse lookup with bounded copy.
 *
 * Copies the surface form into a caller-allocated buffer while the read
 * lock is held, eliminating the TOCTOU vulnerability in le_lexeme_surface().
 *
 * @ctx: Engine context
 * @lexeme_id: Canonical ID from le_intern_lexeme()
 * @out_buf: Caller-allocated output buffer
 * @buf_capacity: Size of out_buf in bytes (must include space for NUL)
 *
 * Returns:  0 on success
 *          -1 invalid arguments
 *          -2 lexeme_id out of bounds
 *          -3 null/uninitialized surface entry
 *          -4 buffer too small (would truncate)
 */
int le_lexeme_surface_safe(EngineContext *ctx, uint32_t lexeme_id,
                           char *out_buf, size_t buf_capacity);

#endif /* LEXEME_INTERN_H */
