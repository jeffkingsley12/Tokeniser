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
    fprintf(stderr, "[tokenizer_save_mmap] not implemented — use tokenizer_save()\n");
    return -1;
}

static int verify_mmap_header(const MmapHeader *hdr, size_t file_size) {
    if (file_size < sizeof(MmapHeader)) return -1;
    if (memcmp(hdr->magic, "LGMMAPv1", 8) != 0) return -1;
    if (hdr->version < 14 || hdr->version > 15) return -1;   /* expect v14 (CSR) or v15 (Truth) */
    
    /* Security Fix: data_size covers the header + payload, but not the 4-byte CRC at the end */
    if (file_size < 4 || hdr->data_size != (uint64_t)(file_size - 4)) return -1;

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

static MmapTokenizer *tokenizer_load_mmap_internal(void *base, size_t size) {
    const MmapHeader *hdr = (const MmapHeader *)base;
    if (verify_mmap_header(hdr, size) != 0) {
        return NULL;
    }

    /* -------------------------------------------------------------------------
     * Security Fix: Verify CRC32 Integrity before processing any offsets
     * ------------------------------------------------------------------------- */
    uint32_t expected_crc = *(uint32_t *)((char *)base + size - 4);
    uint32_t actual_crc = crc32_update(0, (const uint8_t *)base, size - 4);
    
    if (actual_crc != expected_crc) {
        fprintf(stderr, "[load_mmap] CRC mismatch: expected %08x, got %08x\n", expected_crc, actual_crc);
        return NULL;
    }

    MmapTokenizer *mt = calloc(1, sizeof(MmapTokenizer));
    if (!mt) return NULL;

    mt->mmap_base = base;
    mt->mmap_size = size;
    mt->fd        = -1;

    Tokenizer *t = &mt->base;
    t->vocab_size = hdr->vocab_size;

    /* --- SyllableTable ---
     * We allocate a HEAP copy of the SyllableTable because we need to store
     * the transient 'trie' pointer in it, and the mmap is read-only. */
    if (check_range(size, hdr->stbl_offset, sizeof(SyllableTable)) != 0) {
        fprintf(stderr, "[load_mmap] stbl_offset out of range\n");
        goto fail;
    }
    t->stbl = malloc(sizeof(SyllableTable));
    if (!t->stbl) goto fail;
    memcpy(t->stbl, OFF2PTR(base, hdr->stbl_offset), sizeof(SyllableTable));
    t->stbl->trie = NULL; /* will be rebuilt below */

    /* --- Syllabifier --- */
    t->syl = syllabifier_create(t->stbl);
    if (!t->syl) goto fail;

    /* --- RePairState --- */
    t->rs = calloc(1, sizeof(RePairState));
    if (!t->rs) goto fail;
    t->rs->rule_count = hdr->rule_count;

    /* --- LOUDS CSR Trie --- */
    if (hdr->louds_offset > 0) {
        if (check_range(size, hdr->louds_offset, sizeof(LOUDS)) != 0) goto fail;
        t->louds = (LOUDS *)OFF2PTR(base, hdr->louds_offset);
        LOUDS *l = t->louds;
        uint8_t *louds_base = (uint8_t *)l;
        uint64_t l_size = (uint64_t)sizeof(LOUDS);

        uint64_t off_row_ptr = ALIGN8(l_size);
        uint64_t off_edges   = ALIGN8(off_row_ptr + ((uint64_t)l->node_count + 1) * 4);
        uint64_t off_next    = ALIGN8(off_edges   + (uint64_t)l->edge_count * 2);
        uint64_t off_term    = ALIGN8(off_next    + (uint64_t)l->edge_count * 4);
        uint64_t louds_total = off_term    + (uint64_t)l->edge_count * 4;

        if (check_range(size, hdr->louds_offset, louds_total) != 0) goto fail;

        l->row_ptr   = (uint32_t *)(louds_base + (size_t)off_row_ptr);
        l->labels    = (uint16_t *)(louds_base + (size_t)off_edges);
        l->next_node = (uint32_t *)(louds_base + (size_t)off_next);
        l->terminals = (uint32_t *)(louds_base + (size_t)off_term);
    }

    /* --- String table expansions --- */
    if (hdr->strings_size > 0 && hdr->rule_count > 0) {
        if (check_range(size, hdr->strings_offset, hdr->strings_size) != 0) goto fail;
        t->gp_expansions = calloc(hdr->rule_count, sizeof(char *));
        if (!t->gp_expansions) goto fail;
        char *str_base = (char *)OFF2PTR(base, hdr->strings_offset);
        for (uint32_t i = 0; i < hdr->rule_count; i++) {
            t->gp_expansions[i] = str_base + ((size_t)i * MAX_TOKEN_CHARS);
        }
    }

    /* --- Rebuild transient tables (H-01) --- */
    extern void tokenizer_rebuild_id_to_str(Tokenizer *t);
    extern void tokenizer_rebuild_fast_paths(Tokenizer *t);
    
    t->stbl->trie = syltrie_build(t->stbl);
    tokenizer_rebuild_id_to_str(t);
    tokenizer_rebuild_fast_paths(t);

    return mt;

fail:
    if (t->stbl) free(t->stbl);
    if (t->syl) syllabifier_destroy(t->syl);
    if (t->rs) free(t->rs);
    if (t->gp_expansions) free(t->gp_expansions);
    free(mt);
    return NULL;
}

MmapTokenizer *tokenizer_load_mmap(const char *path) {
    if (!path) return NULL;
    int fd = open(path, O_RDONLY);
    if (fd < 0) return NULL;

    struct stat st;
    if (fstat(fd, &st) < 0) { close(fd); return NULL; }
    size_t size = (size_t)st.st_size;

    void *base = mmap(NULL, size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (base == MAP_FAILED) { close(fd); return NULL; }

    MmapTokenizer *mt = tokenizer_load_mmap_internal(base, size);
    if (!mt) { munmap(base, size); close(fd); return NULL; }

    mt->fd          = fd;
    mt->owns_buffer = true;
    return mt;
}

MmapTokenizer *tokenizer_load_mmap_from_buffer(const void *buf, size_t size) {
    if (!buf || size < sizeof(MmapHeader)) return NULL;
    MmapTokenizer *mt = tokenizer_load_mmap_internal((void *)buf, size);
    if (mt) {
        mt->owns_buffer = false; /* caller owns the buffer */
    }
    return mt;
}

void tokenizer_destroy_mmap(MmapTokenizer *mt) {
    if (!mt) return;

    Tokenizer *t = &mt->base;

    /* Free transient tables */
    if (t->stbl) {
        if (t->stbl->trie) syltrie_destroy(t->stbl->trie);
        free(t->stbl);
    }
    if (t->syl) syllabifier_destroy(t->syl);
    if (t->id_to_str) free(t->id_to_str);
    if (t->rs) free(t->rs);
    if (t->gp_expansions) free(t->gp_expansions);

    /* Unmap before closing fd */
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
