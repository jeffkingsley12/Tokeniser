#ifndef TOKENIZER_UTILS_H
#define TOKENIZER_UTILS_H

#include <stdint.h>

/*
 * next_pow2 — return the smallest power of 2 that is >= n.
 *
 * Edge cases:
 *   n == 0           → returns 1  (not 0; a valid minimum capacity)
 *   n is already a   → returns n  (power-of-2 inputs are preserved)
 *   power of 2
 *   n > (1u << 31)   → returns 0  (overflow sentinel; caller must check)
 *
 * Implementation note: the bit-twiddling idiom "n--; n |= n >> k; n++"
 * works for all 32-bit values.  The special cases for 0 and overflow are
 * checked before the main body to avoid undefined behaviour on shift by 32.
 *
 * The original code returned 0 for n==0, which caused
 * pair_table_create() to allocate a zero-element bucket array and then
 * set mask = 0-1 = UINT32_MAX, producing unbounded hash index writes.
 */
static inline uint32_t next_pow2(uint32_t n) {
    if (n == 0) return 1u;              /* degenerate: return minimum */
    if (n > (1u << 31)) return 0u;     /* would overflow uint32_t    */
    n--;
    n |= n >>  1;
    n |= n >>  2;
    n |= n >>  4;
    n |= n >>  8;
    n |= n >> 16;
    n++;
    return n;
}

#endif /* TOKENIZER_UTILS_H */
