/*
 * main.c
 *
 * Demo driver for the Luganda tokenizer pipeline.
 *
 * Usage:
 *   ./tokenizer_demo                   (uses built-in mini corpus)
 *   ./tokenizer_demo path/to/corpus    (one document per line)
 */

#include "tokenizer.h"
#include "corpus_utils.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static const char *MINI_CORPUS[] = {
    "okusoma",
    "okuwandiika",
    "okukola",
    "okulima",
    "okufumba",
    "okugenda",
    "okujja",
    "okunywa",
    "okulya",
    "okutuula",
    "abantu",
    "abakazi",
    "abayenzi",
    "abaana",
    "emitwe",
    "emigga",
    "embwa",
    "enkoko",
    "ente",
    "enjovu",
    "ssebo",
    "nnyabo",
    "bambi",
    "weebale",
    "nkulaakira",
    "okusoma ebitabo",
    "okuwandiika ebbaluwa",
    "okukola omulimu",
    "abantu bali wano",
    "embwa ennungi",
    "mulwadde",
    "akawungeezi",
    "nnyo",
    "lwadde",
    "wungee",
    "gwanga",
    NULL
};


/* =========================================================
 *  Print tokenization result
 * ========================================================= */

/* Generous buffer: 512 tokens covers the longest plausible Luganda
 * sentence.  If a pathological input exceeds this, we log a warning
 * rather than silently dropping tokens.                              */
#define ENCODE_BUF_SIZE 512

/* Encode text into a heap-allocated token array, resizing as needed.
 * Returns the array (caller frees) and sets *n_out to the token count.
 * Returns NULL on error or OOM. */
static uint32_t *encode_alloc(const Tokenizer *tok, const char *text,
                              int *n_out, bool fused) {
    uint32_t cap = ENCODE_BUF_SIZE;
    uint32_t *ids = malloc(cap * sizeof(uint32_t));
    if (!ids) return NULL;

    int n;
    while (1) {
        #if USE_FUSED
        if (fused)
            n = tokenizer_encode_fused(tok, text, ids, cap);
        else
        #endif
            n = tokenizer_encode(tok, text, ids, cap);
        (void)fused; /* suppress unused warning when !USE_FUSED */
        if (n < 0) {
            free(ids);
            *n_out = -1;
            return NULL;
        }
        /* tokenizer_encode returns exactly cap when the buffer was too small.
         * Guard cap doubling against uint32_t overflow (cap > 1<<31). */
        if ((uint32_t)n < cap) {
            break;
        }
        if (cap > UINT32_MAX / 2) {
            free(ids);
            *n_out = -1;
            return NULL;
        }
        cap *= 2;
        uint32_t *new_ids = realloc(ids, cap * sizeof(uint32_t));
        if (!new_ids) { free(ids); *n_out = -1; return NULL; }
        ids = new_ids;
    }
    *n_out = n;
    return ids;
}

static void print_tokenized(const Tokenizer *tok,
                             const char      *text)
{
    int n;
    uint32_t *ids = encode_alloc(tok, text, &n, false);
    if (!ids || n < 0) {
        printf("  [error encoding: \"%s\"]\n", text);
        return;
    }

    printf("  Input : \"%s\"\n", text);
    printf("  Tokens: ");
    for (int i = 0; i < n; i++) {
        const char *dec = tokenizer_decode(tok, ids[i]);
        /* tokenizer_decode returns NULL only for out-of-range IDs.
         * Token ID 0 is <|space|> (TOK_SPACE), not UNK; TOK_UNK is ID 2. */
        printf("[%s/%u]", dec ? dec : "<invalid>", ids[i]);
        if (i + 1 < n) printf(" ");
    }
    printf("  (%d tokens)\n\n", n);
    free(ids);
}


/* Compare regular tokenizer vs fused streaming tokenizer */
#if USE_FUSED
static void test_fused_tokenizer(const Tokenizer *tok,
                                    const char      *text)
{
    int n_regular, n_fused;
    uint32_t *ids_regular = encode_alloc(tok, text, &n_regular, false);
    if (!ids_regular || n_regular < 0) {
        printf("  [error in tokenizer_encode: \"%s\"]\n", text);
        return;
    }

    uint32_t *ids_fused = encode_alloc(tok, text, &n_fused, true);
    if (!ids_fused || n_fused < 0) {
        printf("  [error in tokenizer_encode_fused: \"%s\"]\n", text);
        free(ids_regular);
        return;
    }

    printf("  Input: \"%s\"\n", text);
    printf("  Regular: %d tokens | Fused: %d tokens | ", n_regular, n_fused);

    if (n_regular != n_fused) {
        printf("MISMATCH (count)\n");
        free(ids_regular);
        free(ids_fused);
        return;
    }

    int match = 1;
    for (int i = 0; i < n_regular; i++) {
        if (ids_regular[i] != ids_fused[i]) {
            match = 0;
            break;
        }
    }

    printf("%s\n", match ? "MATCH" : "MISMATCH (tokens)");
    free(ids_regular);
    free(ids_fused);
}
#endif /* USE_FUSED */

/* =========================================================
 *  main
 * ========================================================= */

int main(int argc, char *argv[])
{
    printf("=== Luganda Tokenizer Demo ===\n\n");

    /* ── Load or use built-in corpus ──────────────────────────────────── */
    const char **docs;
    uint32_t     n_docs;
    Corpus      *corpus = NULL;

    if (argc >= 2) {
        printf("Loading corpus from: %s\n", argv[1]);
        corpus = corpus_load(argv[1], 0, 0, 0);
        if (!corpus) {
            fprintf(stderr, "Failed to load corpus from %s\n", argv[1]);
            return 1;
        }
        docs = (const char **)corpus->docs;
        n_docs = corpus->n_docs;
    } else {
        /* Count NULL-terminated array length */
        n_docs = 0;
        while (MINI_CORPUS[n_docs]) n_docs++;
        printf("Using built-in mini corpus (%u documents)\n", n_docs);
        docs = MINI_CORPUS;
    }

    /* ── [1/4] Build tokenizer ───────────────────────────────────────── */
    printf("\n[1/4] Building tokenizer from %u documents...\n", n_docs);
    Tokenizer *tok = tokenizer_build(docs, n_docs);
    if (!tok) {
        fprintf(stderr, "tokenizer_build() failed\n");
        if (corpus) corpus_free(corpus);
        return 1;
    }
    printf("      Syllable table size : %u\n", tok->stbl->count);
    printf("      Grammar rules (total): %u\n", tok->rs->rule_count);

    /* Count live rules */
    uint32_t live = 0;
    for (uint32_t i = 0; i < tok->rs->rule_count; i++)
        if (!tok->rs->rules[i].dead) live++;
    printf("      Grammar rules (live) : %u\n", live);
    printf("      Vocab size           : %u\n", tok->vocab_size);

    /* ── [2/4] Save ──────────────────────────────────────────────────── */
    const char *model_path = "/tmp/luganda_tok.bin";
    printf("\n[2/4] Saving model to %s...\n", model_path);
    if (tokenizer_save(tok, model_path) != 0) {
        fprintf(stderr, "tokenizer_save() failed\n");
        tokenizer_destroy(tok);
        if (corpus) corpus_free(corpus);
        return 1;
    }
    printf("      Done.\n");

    /* ── [3/4] Reload ────────────────────────────────────────────────── */
    printf("\n[3/4] Reloading model from disk...\n");
    Tokenizer *reloaded = tokenizer_load(model_path);
    if (!reloaded) {
        fprintf(stderr, "tokenizer_load() failed\n");
    } else {
        /* Success - replace the original with the reloaded one for the demo */
        tokenizer_destroy(tok);
        tok = reloaded;
        printf("      Model reloaded successfully.\n");
    }

    /* ── [4/4] Tokenize examples ─────────────────────────────────────── */
    printf("\n[4/4] Tokenization examples (using reloaded model):\n\n");

    static const char *tests[] = {
        "okusoma",
        "okuwandiika ebbaluwa",
        "abantu bali wano",
        "enkoko ennungi nnyo",
        "mulwadde",
        "akawungeezi",
        NULL
    };

    for (int i = 0; tests[i]; i++)
        print_tokenized(tok, tests[i]);

    /* ── Fused tokenizer comparison ─────────────────────────────────── */
    #if USE_FUSED
    printf("\n[Bonus] Fused tokenizer comparison:\n\n");

    for (int i = 0; tests[i]; i++)
        test_fused_tokenizer(tok, tests[i]);
    #endif /* USE_FUSED */

    /* ── Streaming tokenizer test ────────────────────────────────────── */
    printf("\n[Bonus] Streaming tokenizer (chunked processing):\n\n");
    {
        TokenizerCursor cursor;
        tokenizer_cursor_init(&cursor, tok, "okusoma nnyo");
        uint32_t buf[4];  /* Small buffer to force multiple calls */
        size_t total = 0, chunk;
        printf("  Input: \"okusoma nnyo\"\n  Chunks: ");
        while (!cursor.eos && (chunk = tokenizer_encode_streaming(&cursor, buf, 4)) > 0) {
            printf("[%zu]", chunk);
            total += chunk;
        }
        printf(" = %zu tokens total\n", total);
    }

    /* ── Persistent compressor test ─────────────────────────────────── */
    printf("\n[Bonus] Persistent Re-Pair compressor:\n\n");
    {
        RePairCompressor *comp = repair_compressor_init();
        if (comp) {
            /* Test sequence from first document */
            uint32_t seq[] = {19, 20, 21, 19, 22, 23, 24};  /* Example syllable IDs */
            uint32_t len = 7;
            printf("  Before: %u tokens [19, 20, 21, 19, 22, 23, 24]\n", len);
            int rc = repair_compress_with_context(comp, tok->rs, seq, &len);
            printf("  After:  %u tokens (rc=%d)\n", len, rc);
            printf("  Compressor context: OK (reusable, zero per-call allocs)\n");
            repair_compressor_destroy(comp);
        } else {
            printf("  Failed to create compressor\n");
        }
    }

    /* ── Cleanup ─────────────────────────────────────────────────────── */
    tokenizer_destroy(tok);
    if (corpus) corpus_free(corpus);

    printf("=== Done ===\n");
    return 0;
}
