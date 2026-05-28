#ifndef TOKENIZER_API_H
#define TOKENIZER_API_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Simple opaque handle based API for FFI (Foreign Function Interface).
 * This provides a flat C API suitable for consumption by Python (ctypes/cffi),
 * Rust, Go, and other languages.
 */

/**
 * Load a tokenizer model from a binary file using mmap.
 * Returns a handle (>= 0) on success, or -1 on failure.
 */
int         tok_load(const char *path);

/**
 * Encode text into a sequence of token IDs.
 * Returns the number of tokens written to 'out', or -1 on failure.
 */
int         tok_encode(int handle, const char *text, uint32_t *out, int cap);

/**
 * Decode a single token ID back into its string representation.
 * Returns a pointer to the internal string (do not free), or NULL on failure.
 */
const char *tok_decode(int handle, uint32_t id);

/**
 * Release the tokenizer and its associated memory-mapped region.
 */
void        tok_free(int handle);

/**
 * Get the total vocabulary size of the loaded tokenizer model.
 * Returns the vocabulary count, or 0 on invalid handle.
 */
uint32_t    tok_vocab_size(int handle);

/**
 * Get the cumulative count of tokens encoded by this handle.
 * Returns total tokens encoded, or 0 on invalid handle.
 */
uint64_t    tok_tokens_encoded(int handle);

#ifdef __cplusplus
}
#endif

#endif
