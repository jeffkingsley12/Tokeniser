#ifndef CORPUS_UTILS_H
#define CORPUS_UTILS_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

/* Unified Corpus structure used across benchmark and main demo tools */
typedef struct {
    char **docs;
    uint32_t n_docs;
    size_t total_bytes;
    void  *mmap_ptr;   /* Pointer to mapped file region */
    size_t mmap_len;   /* Total size of the mapped region */
} Corpus;

/**
 * Load a corpus from a text file (one document per line).
 * 
 * @param path Path to the corpus file.
 * @param max_bytes Maximum total bytes to load (0 for unlimited).
 * @param max_lines Maximum number of lines to load (0 for unlimited).
 * @param line_buf_size Maximum bytes per line (lines longer than this are truncated).
 * @return Pointer to allocated Corpus, or NULL on error.
 */
Corpus *corpus_load(const char *path, size_t max_bytes, uint32_t max_lines, size_t line_buf_size);

/**
 * Free a corpus allocated by corpus_load.
 * @param c Corpus to free (NULL-safe).
 */
void corpus_free(Corpus *c);

#endif /* CORPUS_UTILS_H */
