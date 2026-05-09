#include "tokenizer.h"
#include "truth_layer.h"
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <stdio.h>
#include <inttypes.h>

/* Alignment helper */
#define ALIGN8(x) (((x) + 7ULL) & ~7ULL)

/* Convert offset to pointer within mapping.
 * Declared as returning (char *) to make arithmetic explicit and avoid
 * accidental pointer arithmetic through void *. */
#define OFF2PTR(base, off) ((void *)((char *)(base) + (off)))

/* Compute truth seed fingerprint for reproducibility */
extern uint64_t compute_truth_seed_hash(void);

/* -------------------------------------------------------------------------
 * Helper: validate that an offset + span lies entirely within the mapping.
 * Returns 0 on success, -1 if the range is out of bounds.
 * ------------------------------------------------------------------------- */
static int check_range(size_t file_size, uint64_t offset, uint64_t span) {
    if (offset > file_size) return -1;
    if (span   > file_size - offset) return -1;
    return 0;
}

int tokenizer_save_mmap(const Tokenizer *t, const char *path) {
    /* NOT IMPLEMENTED: The current system focuses on regular tokenizer_save
     * for model building. Mmap is primarily used for read-only loading in
     * production.  Stub retained to satisfy link-time references. */
    (void)t; (void)path;
    return -1;
}

static int verify_mmap_header(const MmapHeader *hdr, size_t file_size) {
    if (file_size < sizeof(MmapHeader)) return -1;
    if (memcmp(hdr->magic, "LGMMAPv1", 8) != 0) return -1;
    if (hdr->version < 14 || hdr->version > 15) return -1;   /* expect v14 (CSR) or v15 (Truth) */
    if (hdr->data_size != (uint64_t)file_size) return -1;

    /* v15: Reproducibility check (non-fatal: warn but do not abort) */
    if (hdr->version >= 15) {
        uint64_t current_hash = compute_truth_seed_hash();
        if (hdr->truth_seed_hash != current_hash) {
            fprintf(stderr,
                "[load_mmap] WARNING: truth seed hash mismatch "
                "(file=%" PRIx64 " current=%" PRIx64 ") — model may be stale\n",
                hdr->truth_seed_hash, current_hash);
        }
    }

    return 0;
}

MmapTokenizer *tokenizer_load_mmap(const char *path) {
    if (!path) return NULL;

    int fd = open(path, O_RDONLY);
    if (fd < 0) {
        perror("tokenizer_load_mmap: open");
        return NULL;
    }

    struct stat st;
    if (fstat(fd, &st) < 0) {
        perror("tokenizer_load_mmap: fstat");
        close(fd);
        return NULL;
    }

    /* Reject empty files and files too large to mmap on this platform */
    if (st.st_size <= 0) {
        fprintf(stderr, "[load_mmap] file is empty: %s\n", path);
        close(fd);
        return NULL;
    }
    if ((uint64_t)st.st_size > (uint64_t)SIZE_MAX) {
        fprintf(stderr, "[load_mmap] file too large for 32-bit size_t: %s\n", path);
        close(fd);
        return NULL;
    }
    size_t size = (size_t)st.st_size;

    if (size < sizeof(MmapHeader)) {
        fprintf(stderr, "[load_mmap] file too small to contain a header: %s\n", path);
        close(fd);
        return NULL;
    }

    void *base = mmap(NULL, size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (base == MAP_FAILED) {
        perror("tokenizer_load_mmap: mmap");
        close(fd);
        return NULL;
    }

    const MmapHeader *hdr = (const MmapHeader *)base;
    if (verify_mmap_header(hdr, size) != 0) {
        fprintf(stderr, "[load_mmap] header verification failed: %s\n", path);
        munmap(base, size);
        close(fd);
        return NULL;
    }

    MmapTokenizer *mt = calloc(1, sizeof(MmapTokenizer));
    if (!mt) {
        munmap(base, size);
        close(fd);
        return NULL;
    }

    mt->mmap_base  = base;
    mt->mmap_size  = size;
    mt->fd         = fd;
    mt->owns_buffer = true;

    Tokenizer *t = &mt->base;
    t->vocab_size = hdr->vocab_size;

    /* --- Wire up SyllableTable ---
     * Validate the offset before dereferencing. */
    if (check_range(size, hdr->stbl_offset, sizeof(SyllableTable)) != 0) {
        fprintf(stderr, "[load_mmap] stbl_offset out of range\n");
        goto fail;
    }
    t->stbl = (SyllableTable *)OFF2PTR(base, hdr->stbl_offset);

    /* --- RePairState ---
     * Only metadata is needed at inference time; rules are not traversed
     * via t->rs->rules, so we leave rules pointer NULL and store the count. */
    t->rs = calloc(1, sizeof(RePairState));
    if (!t->rs) goto fail;
    t->rs->rule_count = hdr->rule_count;
    /* rules pointer intentionally left NULL — inference does not walk rules */

    /* --- LOUDS CSR Trie ---
     * Validate the LOUDS section offset and then wire the internal
     * sub-array pointers as offsets from the start of the LOUDS block.
     *
     * Layout (immediately after the LOUDS header struct):
     *   [row_ptr:  csr_node_count  × 4 bytes, ALIGN8]
     *   [edges:    csr_edge_count  × 2 bytes, ALIGN8]
     *   [next_node:csr_edge_count  × 4 bytes, ALIGN8]
     *   [terminals:csr_edge_count  × 4 bytes, ALIGN8]
     */
    if (check_range(size, hdr->louds_offset, sizeof(LOUDS)) != 0) {
        fprintf(stderr, "[load_mmap] louds_offset out of range\n");
        goto fail;
    }
    t->louds = (LOUDS *)OFF2PTR(base, hdr->louds_offset);
    {
        LOUDS   *l          = t->louds;
        uint8_t *louds_base = (uint8_t *)l;
        uint64_t l_size     = (uint64_t)sizeof(LOUDS);

        /* Compute sub-section byte offsets in uint64_t to prevent truncation
         * on 32-bit platforms when node_count or edge_count is large.
         * All arithmetic stays in uint64_t until check_range has validated
         * that the values fit within the mapping — only then are they cast
         * to size_t for pointer arithmetic. */
        uint64_t off_row_ptr = l_size;
        uint64_t off_edges   = off_row_ptr + ALIGN8((uint64_t)l->node_count * 4);
        uint64_t off_next    = off_edges   + ALIGN8((uint64_t)l->edge_count * 2);
        uint64_t off_term    = off_next    + ALIGN8((uint64_t)l->edge_count * 4);
        uint64_t louds_total = off_term    + ALIGN8((uint64_t)l->edge_count * 4);

        /* Validate the full LOUDS block fits inside the mapping */
        if (check_range(size, hdr->louds_offset, louds_total) != 0) {
            fprintf(stderr, "[load_mmap] LOUDS sub-arrays extend past file end\n");
            goto fail;
        }

        /* Safe to cast: check_range guarantees all offsets < file_size <= SIZE_MAX */
        l->row_ptr   = (uint32_t *)(louds_base + (size_t)off_row_ptr);
        l->labels    = (uint16_t *)(louds_base + (size_t)off_edges);
        l->next_node = (uint32_t *)(louds_base + (size_t)off_next);
        l->terminals = (uint32_t *)(louds_base + (size_t)off_term);
    }

    /* --- String table (expansion strings for grammar rules) --- */
    if (hdr->strings_size > 0 && hdr->rule_count > 0) {
        /* Validate that rule_count × MAX_TOKEN_CHARS fits in strings_size */
        uint64_t expected = (uint64_t)hdr->rule_count * MAX_TOKEN_CHARS;
        if (expected > hdr->strings_size) {
            fprintf(stderr,
                "[load_mmap] strings_size (%" PRIu64 ") < rule_count × MAX_TOKEN_CHARS "
                "(%" PRIu64 ")\n",
                (uint64_t)hdr->strings_size, expected);
            goto fail;
        }
        if (check_range(size, hdr->strings_offset, hdr->strings_size) != 0) {
            fprintf(stderr, "[load_mmap] strings section out of range\n");
            goto fail;
        }

        t->gp_expansions = calloc(hdr->rule_count, sizeof(char *));
        if (!t->gp_expansions) goto fail;

        char *str_base = (char *)OFF2PTR(base, hdr->strings_offset);
        for (uint32_t i = 0; i < hdr->rule_count; i++) {
            t->gp_expansions[i] = str_base + ((size_t)i * MAX_TOKEN_CHARS);
        }
    }

    return mt;

fail:
    /* Partial init — clean up what we allocated before returning NULL */
    if (t->rs)            { free(t->rs);            t->rs = NULL; }
    if (t->gp_expansions) { free(t->gp_expansions); t->gp_expansions = NULL; }
    munmap(base, size);
    close(fd);
    free(mt);
    return NULL;
}

void tokenizer_destroy_mmap(MmapTokenizer *mt) {
    if (!mt) return;

    /* Free heap-allocated wrappers first (they do not own the mmap memory) */
    if (mt->base.rs)            free(mt->base.rs);
    if (mt->base.gp_expansions) free(mt->base.gp_expansions);

    /* Unmap before closing fd — fd is no longer needed once mapped */
    if (mt->owns_buffer && mt->mmap_base) {
        munmap(mt->mmap_base, mt->mmap_size);
        mt->mmap_base = NULL;
    }

    if (mt->fd >= 0) {
        close(mt->fd);
        mt->fd = -1;
    }

    free(mt);
}
