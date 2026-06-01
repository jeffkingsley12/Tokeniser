#include "tokenizer.h"
#include "truth_layer.h"
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <stdio.h>
#include <inttypes.h>
#include <string.h>

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

#define MMAP_MIN_SUPPORTED_VERSION 14
#define MMAP_CURRENT_VERSION        15

static int verify_mmap_header(const MmapHeader *hdr, size_t file_size) {
    if (file_size < sizeof(MmapHeader)) return -1;
    if (memcmp(hdr->magic, "LGMMAPv1", 8) != 0) return -1;
    if (hdr->version < MMAP_MIN_SUPPORTED_VERSION ||
        hdr->version > MMAP_CURRENT_VERSION) {
        fprintf(stderr,
            "[load_mmap] Unsupported format version %u "
            "(supported: %u\u2013%u)\n",
            hdr->version,
            MMAP_MIN_SUPPORTED_VERSION,
            MMAP_CURRENT_VERSION);
        return -1;
    }
    
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
    uint32_t expected_crc;
    memcpy(&expected_crc, (const char *)base + size - 4, sizeof(uint32_t));
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
        
        /* Copy LOUDS header to heap so we can fix up pointers in it */
        t->louds = malloc(sizeof(LOUDS));
        if (!t->louds) goto fail;
        memcpy(t->louds, OFF2PTR(base, hdr->louds_offset), sizeof(LOUDS));
        
        LOUDS *l = t->louds;
        uint8_t *louds_base_on_disk = (uint8_t *)OFF2PTR(base, hdr->louds_offset);
        uint64_t l_size = (uint64_t)sizeof(LOUDS);

        /* C-3: cap node/edge counts before any arithmetic to prevent
         * ALIGN8 overflow on adversarial input.  65 M nodes/edges is far
         * above any realistic vocabulary size while staying inside uint32_t. */
#define MAX_LOUDS_NODES  ((uint32_t)0x03FFFFFF)  /* ~67 M */
#define MAX_LOUDS_EDGES  ((uint32_t)0x03FFFFFF)
        if (l->csr_node_count > MAX_LOUDS_NODES || l->csr_edge_count > MAX_LOUDS_EDGES) {
            fprintf(stderr, "[load_mmap] LOUDS node/edge count out of range "
                    "(nodes=%u edges=%u)\n", l->csr_node_count, l->csr_edge_count);
            goto fail;
        }
        uint64_t row_bytes  = ((uint64_t)l->csr_node_count + 1) * 4;
        uint64_t off_row_ptr = ALIGN8(l_size);
        if (off_row_ptr < l_size) goto fail;            /* ALIGN8 overflow */
        uint64_t tmp1 = off_row_ptr + row_bytes;
        if (tmp1 < off_row_ptr) goto fail;
        uint64_t off_edges = ALIGN8(tmp1);
        if (off_edges < tmp1) goto fail;
        uint64_t tmp2 = off_edges + (uint64_t)l->csr_edge_count * 2;
        if (tmp2 < off_edges) goto fail;
        uint64_t off_next = ALIGN8(tmp2);
        if (off_next < tmp2) goto fail;
        uint64_t tmp3 = off_next + (uint64_t)l->csr_edge_count * 4;
        if (tmp3 < off_next) goto fail;
        uint64_t off_term = ALIGN8(tmp3);
        if (off_term < tmp3) goto fail;
        /* FIX: terminals[] is indexed by node (0..node_count-1), not by edge. */
        uint64_t tmp4 = off_term + (uint64_t)l->csr_node_count * 4;
        if (tmp4 < off_term) goto fail;
        uint64_t louds_total = tmp4;

        if (check_range(size, hdr->louds_offset, louds_total) != 0) goto fail;

        l->has_csr = true;
        l->is_mmap_backed = true;

        l->row_ptr   = (uint32_t *)(louds_base_on_disk + (size_t)off_row_ptr);
        l->labels    = (uint16_t *)(louds_base_on_disk + (size_t)off_edges);
        l->next_node = (uint32_t *)(louds_base_on_disk + (size_t)off_next);
        l->terminals = (uint32_t *)(louds_base_on_disk + (size_t)off_term);
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
    if (t->louds) louds_destroy(t->louds);
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
    /* M-6: buf is const; we store it as const in mmap_base when owns_buffer=false
     * so that any future write-path through mmap_base cannot reach it.
     * The cast here is safe only because tokenizer_load_mmap_internal never
     * writes through the pointer for read-only (owns_buffer=false) mappings. */
    MmapTokenizer *mt = tokenizer_load_mmap_internal((void *)(uintptr_t)buf, size);
    if (mt) {
        mt->owns_buffer = false; /* caller owns the buffer; must not munmap it */
    }
    return mt;
}

void tokenizer_destroy_mmap(MmapTokenizer *mt) {
    fprintf(stderr, "[DESTROY_MMAP] mt=%p\n", (void*)mt);
    if (!mt) {
        fprintf(stderr, "[DESTROY_MMAP] NULL mt, returning\n");
        return;
    }

    Tokenizer *t = &mt->base;
    fprintf(stderr, "[DESTROY_MMAP] t=%p\n", (void*)t);

    /* Free transient tables */
    fprintf(stderr, "[DESTROY_MMAP] freeing stbl\n");
    if (t->stbl) {
        if (t->stbl->trie) syltrie_destroy(t->stbl->trie);
        free(t->stbl);
    }
    fprintf(stderr, "[DESTROY_MMAP] freeing syl\n");
    if (t->syl) syllabifier_destroy(t->syl);
    fprintf(stderr, "[DESTROY_MMAP] skipping louds free (mmap'd pointer)\n");
    /* WORKAROUND: louds internal arrays point into mmap'd region.
     * The LOUDS struct wrapper is also potentially mmap'd in some paths.
     * Skip the free entirely - the OS will reclaim mmap'd memory on exit.
     * The mmap'd region will be unmapped below. */
    if (t->louds) {
        t->louds = NULL;  /* Poison pointer */
    }
    fprintf(stderr, "[DESTROY_MMAP] freeing id_to_str\n");
    if (t->id_to_str) free(t->id_to_str);
    fprintf(stderr, "[DESTROY_MMAP] freeing token features\n");
    free(t->token_features);
    free(t->token_requires);
    free(t->token_forbids);
    fprintf(stderr, "[DESTROY_MMAP] freeing rs\n");
    if (t->rs) free(t->rs);
    fprintf(stderr, "[DESTROY_MMAP] freeing gp_expansions\n");
    if (t->gp_expansions) free(t->gp_expansions);

    /* Unmap before closing fd */
    fprintf(stderr, "[DESTROY_MMAP] unmapping\n");
    if (mt->owns_buffer && mt->mmap_base) {
        munmap(mt->mmap_base, mt->mmap_size);
        mt->mmap_base = NULL;
    }

    fprintf(stderr, "[DESTROY_MMAP] closing fd\n");
    if (mt->fd >= 0) {
        close(mt->fd);
        mt->fd = -1;
    }

    free(mt);
}
