#include "corpus_utils.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <stdint.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

Corpus *corpus_load(const char *path, size_t max_bytes, uint32_t max_lines,
                    size_t line_buf_size) {
    (void)line_buf_size; /* Unused in mmap mode */
    if (!path) return NULL;

    /* O_RDONLY is sufficient: MAP_PRIVATE directs all writes (the '\0'
     * null-termination pass) to anonymous private pages, never back to
     * the file.  O_RDWR would fail on read-only mounts and 444 files. */
    int fd = open(path, O_RDONLY);
    if (fd < 0) {
        perror("corpus_load: open");
        return NULL;
    }

    struct stat st;
    if (fstat(fd, &st) < 0) {
        perror("corpus_load: fstat");
        close(fd);
        return NULL;
    }

    /* Guard against files too large to address with size_t (32-bit platforms) */
    if ((uint64_t)st.st_size > (uint64_t)SIZE_MAX) {
        fprintf(stderr, "corpus_load: file too large for this platform\n");
        close(fd);
        return NULL;
    }
    size_t file_len = (size_t)st.st_size;
    if (file_len == 0) {
        close(fd);
        return calloc(1, sizeof(Corpus));
    }

    /* MAP_PRIVATE allows us to write '\0' over newlines without modifying the source file.
     * The OS handles loading pages from disk into memory only as we touch them. */
    void *ptr = mmap(NULL, file_len, PROT_READ | PROT_WRITE, MAP_PRIVATE, fd, 0);
    close(fd);
    if (ptr == MAP_FAILED) {
        perror("corpus_load: mmap");
        return NULL;
    }

    Corpus *c = calloc(1, sizeof(Corpus));
    if (!c) { munmap(ptr, file_len); return NULL; }

    c->mmap_ptr = ptr;
    c->mmap_len = file_len;

    uint32_t cap = 1024;
    c->docs = malloc(cap * sizeof(char *));
    if (!c->docs) { munmap(ptr, file_len); free(c); return NULL; }

    char *p = (char *)ptr;
    char *end = p + file_len;

    /* Phase 1: Replace all line endings with null terminators */
    for (char *tmp = p; tmp < end; tmp++) {
        if (*tmp == '\n' || *tmp == '\r') {
            *tmp = '\0';
        }
    }

    /* Phase 2: Scan for non-empty document strings */
    while (p < end) {
        /* Skip leading nulls (empty lines) */
        while (p < end && *p == '\0') p++;
        if (p == end) break;

        /* Found start of a document */
        if (c->n_docs >= cap) {
            if (cap > (uint32_t)(UINT32_MAX / 2)) break;
            cap *= 2;
            char **tmp = realloc(c->docs, cap * sizeof(char *));
            if (!tmp) break;
            c->docs = tmp;
        }

        c->docs[c->n_docs++] = p;
        size_t len = strlen(p);
        c->total_bytes += len;
        p += len + 1; /* +1 to step over the null terminator written in Phase 1 */

        if (max_bytes > 0 && c->total_bytes >= max_bytes) break;
        if (max_lines > 0 && c->n_docs >= max_lines) break;
    }

    return c;
}

void corpus_free(Corpus *c) {
    if (!c) return;
    if (c->mmap_ptr) {
        munmap(c->mmap_ptr, c->mmap_len);
    } else {
        for (uint32_t i = 0; i < c->n_docs; i++) {
            free(c->docs[i]);
        }
    }
    free(c->docs);
    free(c);
}
