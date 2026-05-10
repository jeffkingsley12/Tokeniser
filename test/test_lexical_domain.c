/*
 * test_lexical_domain.c
 *
 * Diagnostic: expose every syllable the scanner can emit that is NOT
 * present as an edge label in the LOUDS trie.  This makes the split-
 * lexical-authority bug fully visible before any architectural fix.
 *
 * Build:
 *   make test/test_lexical_domain
 * Run:
 *   ./test/test_lexical_domain [model.bin]
 */

#include "tokenizer.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>

/* -----------------------------------------------------------------------
 * Collect every LOUDS edge label (syllable ID) that appears in the trie.
 * ----------------------------------------------------------------------- */
static bool *collect_louds_domain(const LOUDS *l, uint32_t domain_size) {
    bool *seen = calloc(domain_size, sizeof *seen);
    if (!seen) return NULL;

    for (uint32_t i = 0; i < l->edge_count; i++) {
        uint16_t lbl = l->labels[i];
        if (lbl < domain_size)
            seen[lbl] = true;
    }

    /* Also mark single-syllable tokens that appear in terminals[] */
    for (uint32_t n = 0; n < l->node_count; n++) {
        if (l->terminals[n] != 0) {
            /* terminal at a depth-1 node → single-syllable token */
            (void)n;
        }
    }

    /* Mark single-syllable entries via the flat table */
    for (uint32_t i = 0; i < l->n_entries; i++) {
        if (l->klens[i] == 1) {
            uint32_t off = l->koffs[i];
            uint16_t syl = l->labels_legacy[off];
            if (syl < domain_size)
                seen[syl] = true;
        }
    }

    return seen;
}

/* -----------------------------------------------------------------------
 * Walk every stbl entry through consume_syllable() and record which
 * syllable IDs it emits so we can compare with the LOUDS domain.
 * ----------------------------------------------------------------------- */
static void run_diagnostic(const Tokenizer *t) {
    const SyllableTable *stbl = t->stbl;
    const LOUDS        *l    = t->louds;

    if (!stbl || !l) {
        fprintf(stderr, "[diag] tokenizer missing stbl or LOUDS\n");
        return;
    }

    uint32_t domain_size = (uint32_t)stbl->count + 1;
    bool *louds_domain = collect_louds_domain(l, domain_size);
    if (!louds_domain) {
        fprintf(stderr, "[diag] OOM\n");
        return;
    }

    printf("=== Lexical Domain Diagnostic ===\n");
    printf("stbl->count = %u\n", stbl->count);
    printf("LOUDS: %u nodes, %u edges, %u entries\n",
           l->node_count, l->edge_count, l->n_entries);

    /* ------------------------------------------------------------------
     * Part 1: Every syllable ID in stbl that is NOT a LOUDS edge label.
     * These are orphan atoms that the scanner can emit but the trie will
     * never match.
     * ------------------------------------------------------------------ */
    uint32_t orphan_count = 0;
    printf("\n--- Part 1: stbl entries absent from LOUDS edge labels ---\n");
    for (uint16_t si = 0; si < stbl->count; si++) {
        if (!louds_domain[si]) {
            const char *text = stbl->entries[si].text;
            printf("  stbl[%u] = \"%s\"  NOT in LOUDS edge labels\n", si, text);
            orphan_count++;
        }
    }
    if (orphan_count == 0)
        printf("  (none — stbl is a subset of LOUDS labels)\n");
    else
        printf("  TOTAL: %u orphan stbl entries\n", orphan_count);

    /* ------------------------------------------------------------------
     * Part 2: Check the specific failing tokens.  Syllabify each, then
     * check whether every emitted ID is a LOUDS edge label.
     * ------------------------------------------------------------------ */
    const char *probes[] = {
        "kanywa-musenke",
        "kafumita-bagenge",
        "kalumanywere",
        /* extra probes to test separator & nasal orphan paths */
        "n", "y", "w", "-", "nnakitaddabusa",
        NULL
    };

    printf("\n--- Part 2: Per-token syllabification vs LOUDS domain ---\n");
    uint16_t seq[MAX_SEQ_LEN];
    for (int pi = 0; probes[pi]; pi++) {
        const char *tok = probes[pi];
        int n = syllabify(t->syl, tok, seq, MAX_SEQ_LEN);
        printf("\n  \"%s\"  (%d syllables)\n", tok, n);
        bool tok_ok = true;
        for (int i = 0; i < n; i++) {
            uint16_t id = seq[i];
            const char *name = (id < stbl->count) ? stbl->entries[id].text : "???";
            bool in_louds = (id < domain_size) && louds_domain[id];
            printf("    [%d] id=%-4u text=%-10s  %s\n",
                   i, id, name, in_louds ? "OK" : "** MISSING FROM LOUDS **");
            if (!in_louds) tok_ok = false;
        }
        if (tok_ok && n > 0) printf("    => all syllables in LOUDS domain\n");
    }

    /* ------------------------------------------------------------------
     * Part 3: Check the consumer emit ID vs stbl vs LOUDS.
     *   Walks raw ASCII chars 0x20-0x7E through consume_syllable() and
     *   reports any byte whose emitted syllable text is not in the LOUDS
     *   domain.  This catches runtime fallback paths.
     * ------------------------------------------------------------------ */
    printf("\n--- Part 3: Raw-byte scan through syllabify() ---\n");
    uint32_t raw_orphan = 0;
    for (int c = 0x20; c <= 0x7E; c++) {
        char one[2] = { (char)c, '\0' };
        uint16_t sid[2];
        int ns = syllabify(t->syl, one, sid, 2);
        if (ns <= 0) continue; /* silently dropped */
        uint16_t id = sid[0];
        bool in_louds = (id < domain_size) && louds_domain[id];
        if (!in_louds) {
            const char *name = (id < stbl->count) ? stbl->entries[id].text : "???";
            printf("  byte 0x%02X ('%c') -> stbl_id=%-4u (\"%s\")  ** MISSING FROM LOUDS **\n",
                   c, (char)c, (unsigned)id, name);
            raw_orphan++;
        }
    }
    if (raw_orphan == 0)
        printf("  (none — every ASCII byte maps to a LOUDS-domain syllable)\n");
    else
        printf("  TOTAL: %u raw-byte gaps\n", raw_orphan);

    free(louds_domain);
    printf("\n=== End Lexical Domain Diagnostic ===\n");
}

int main(int argc, char **argv) {
    const char *model_path = (argc > 1) ? argv[1] : "tokenizer_model.bin";

    printf("Loading model: %s\n", model_path);
    Tokenizer *t = tokenizer_load(model_path);
    if (!t) {
        fprintf(stderr, "Failed to load model (have you run tokenizer_build first?)\n");
        return 1;
    }

    run_diagnostic(t);
    tokenizer_destroy(t);
    return 0;
}
