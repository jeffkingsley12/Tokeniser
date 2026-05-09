/*
 * tokenizer_config.h
 *
 * Central configuration for the Luganda tokenizer.
 * Tune here to switch between safety/maximum speed or to enable
 * experimental features.
 *
 * ALL compile-time limits live here.  No other header or source file
 * should define these macros independently — always #include this file
 * first, then #include tokenizer.h.
 */

#ifndef TOKENIZER_CONFIG_H
#define TOKENIZER_CONFIG_H

/* ──────────────────────────────────────────
 *   P0: Production Stability (Safe Defaults)
 * ────────────────────────────────────────── */

/* Fused syllabify+tokenize pass — experimental, currently slower on most
 * workloads than the two-pass path.  Disable for production builds. */
#ifndef USE_FUSED
#define USE_FUSED 0
#endif

/* Atomic stats collection.  Each hit costs ~20-30 cycles on x86.
 * Leave off in production unless you need per-call telemetry. */
#ifndef ENABLE_STATS
#define ENABLE_STATS 0
#endif

/* Grapheme cluster mode: ensures the syllabifier never sees partial
 * graphemes (e.g. emoji + skin-tone modifier split across a call boundary).
 * Disable only if you are certain the input is pure Luganda ASCII. */
#ifndef USE_GRAPHEME_CLUSTERS
#define USE_GRAPHEME_CLUSTERS 1
#endif

/* In non-grapheme mode, skip multi-byte UTF-8 sequences atomically to
 * avoid yielding incomplete code-unit sequences to the syllabifier. */
#ifndef BYTE_MODE_UTF8_SKIP
#define BYTE_MODE_UTF8_SKIP 1
#endif

/* ──────────────────────────────────────────
 *   Compile-time Hard Limits
 * ────────────────────────────────────────── */

/* Version stamp — increment when the config ABI changes. */
#define TOKENIZER_CONFIG_VERSION 2

/* Catch stale tokenizer.h that was compiled with an older config. */
#if defined(TOKENIZER_H) && TOKENIZER_CONFIG_VERSION < 2
#error "tokenizer_config.h version mismatch — rebuild tokenizer.h"
#endif

/* Maximum number of syllable/token IDs per encode call.
 * 4096 is sufficient for the longest realistic Luganda sentence. */
#ifndef MAX_SYLLABLES
#define MAX_SYLLABLES       4096u
#endif

/* Syllable IDs occupy [0, BASE_SYMBOL_OFFSET).
 * Grammar rule IDs start at BASE_SYMBOL_OFFSET.
 * Must equal MAX_SYLLABLES for the LOUDS fast-path arrays to be sized
 * correctly (fast_token[BASE_SYMBOL_OFFSET], can_extend[BASE_SYMBOL_OFFSET]). */
#ifndef BASE_SYMBOL_OFFSET
#define BASE_SYMBOL_OFFSET  4096u
#endif

#if BASE_SYMBOL_OFFSET != MAX_SYLLABLES
#error "BASE_SYMBOL_OFFSET must equal MAX_SYLLABLES"
#endif

/* Maximum number of Re-Pair grammar rules. */
#ifndef MAX_RULES
#define MAX_RULES           32768u
#endif

/* Re-Pair pair-frequency pruning threshold.
 * Only pairs that appear more than MIN_PAIR_FREQ times are merged. */
#ifndef MIN_PAIR_FREQ
#define MIN_PAIR_FREQ       3u
#endif

/* Maximum UTF-8 bytes in a single token's surface form (including NUL). */
#ifndef MAX_TOKEN_CHARS
#define MAX_TOKEN_CHARS     64u
#endif

/* =========================================================
 *  P1: Cache Optimization (Advanced)
 * ========================================================= */

/* Replace the 512 KB sparse cv_to_token / space_cv_to_token arrays with a
 * 256-entry dense hash table, cutting L1/L2 pressure significantly. */
#ifndef USE_DENSE_HASH
#define USE_DENSE_HASH 1
#endif

/* Emit __builtin_prefetch hints inside the LOUDS trie traversal loop. */
#ifndef USE_TRIE_PREFETCH
#define USE_TRIE_PREFETCH 1
#endif

/* =========================================================
 *  P2: SIMD & Platform Acceleration
 * ========================================================= */

/* Truth layer: morphological verification trie.
 * Set to 0 to omit the truth-layer code entirely from the binary. */
#ifndef ENABLE_TRUTH_LAYER
#define ENABLE_TRUTH_LAYER 1
#endif

/* SSE4.1 syllabification (x86_64 only).
 * Enabled automatically when the compiler defines __SSE4_1__. */
#ifndef USE_SSE4_1
#  ifdef __SSE4_1__
#    define USE_SSE4_1 1
#  else
#    define USE_SSE4_1 0
#  endif
#endif

/* NEON syllabification (AArch64 / ARM only).
 * Enabled automatically when the compiler defines __ARM_NEON. */
#ifndef USE_NEON
#  ifdef __ARM_NEON
#    define USE_NEON 1
#  else
#    define USE_NEON 0
#  endif
#endif

/* =========================================================
 *  P3: Runtime Telemetry (Stats)
 * ========================================================= */

#if ENABLE_STATS
  /* Use GCC/Clang built-in atomics — C11 <stdatomic.h> wrappers are in
   * tokenizer.h so the GCC builtins are the lowest-common-denominator here. */
  #define INC_ATOMIC(x)    __atomic_add_fetch(&(x), 1,  __ATOMIC_RELAXED)
  #define ADD_ATOMIC(x, v) __atomic_add_fetch(&(x), (v), __ATOMIC_RELAXED)
  #define STAT_INC(x)      INC_ATOMIC(x)
  #define STAT_ADD(x, v)   ADD_ATOMIC(x, v)
  /* Zero-initialiser for TokenizerStats (all fields, including atomics) */
  #define STATS_INITIALIZER {0}
#else
  #define INC_ATOMIC(x)    ((void)0)
  #define ADD_ATOMIC(x, v) ((void)0)
  #define STAT_INC(x)      ((void)0)
  #define STAT_ADD(x, v)   ((void)0)
  #define STATS_INITIALIZER {0}
#endif

/* =========================================================
 *  Sanity checks
 * ========================================================= */

#if USE_FUSED && !USE_GRAPHEME_CLUSTERS
#  warning "USE_FUSED without USE_GRAPHEME_CLUSTERS may split emoji incorrectly"
#endif

#if ENABLE_STATS && !defined(__ATOMIC_RELAXED)
#  error "ENABLE_STATS requires a compiler that supports __atomic builtins (GCC ≥ 4.7 or Clang ≥ 3.1)"
#endif

#if MAX_RULES > 65535u
#  error "MAX_RULES exceeds uint16_t range — Rule.depth and related fields must be widened"
#endif

#endif /* TOKENIZER_CONFIG_H */
