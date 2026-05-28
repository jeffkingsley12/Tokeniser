#ifndef _GNU_SOURCE
#  define _GNU_SOURCE
#endif

#include "libgemini.h"
#include "tokenizer_api.h"
#include "gemini_internal.h"
#include "bridge_engine.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <ctype.h>
#include <inttypes.h>  /* PRIu64, PRIu32 — required for portable uint64_t printf */
#include <signal.h>    /* Required for sigaction and sig_atomic_t */
#include <math.h>      /* Required for isfinite */
#include <sys/select.h> /* Required for select() for non-blocking I/O */
#include <errno.h>      /* Required for errno and EINTR */

#define EPOCH_INTERVAL 100
#define SNAPSHOT_INTERVAL 5 /* Save every 5 epochs */
#define MAX_LINE_LEN 4096

#define HASH_SIZE_WORDS 262144

typedef struct WordEntry {
    char *word;
    uint32_t freq;
    struct WordEntry *next;
} WordEntry;

static WordEntry *word_buckets[HASH_SIZE_WORDS] = {0};
static uint32_t unique_words_count = 0;

/* Volatile flag for safe signal handling across execution loops */
static volatile sig_atomic_t g_keep_running = 1;

static void handle_shutdown_signal(int sig) {
    (void)sig;
    g_keep_running = 0;
}

static void add_word_to_freq(const char *w) {
    while (*w && !isalnum((unsigned char)*w)) {
        w++;
    }
    if (*w == '\0') return;

    char cleaned[256];
    size_t len = 0;
    while (w[len] && len < 255) {
        cleaned[len] = w[len];
        len++;
    }
    cleaned[len] = '\0';
    while (len > 0 && !isalnum((unsigned char)cleaned[len - 1])) {
        cleaned[--len] = '\0';
    }
    if (len == 0) return;

    for (size_t i = 0; i < len; i++) {
        cleaned[i] = tolower((unsigned char)cleaned[i]);
    }

    unsigned long hash = 5381;
    for (size_t i = 0; i < len; i++) {
        hash = ((hash << 5) + hash) + (unsigned char)cleaned[i];
    }
    uint32_t bucket = hash % HASH_SIZE_WORDS;

    WordEntry *curr = word_buckets[bucket];
    while (curr) {
        if (strcmp(curr->word, cleaned) == 0) {
            curr->freq++;
            return;
        }
        curr = curr->next;
    }

    WordEntry *entry = malloc(sizeof(WordEntry));
    if (!entry) return;
    entry->word = strdup(cleaned);
    if (!entry->word) { free(entry); return; }  /* FIX: strdup returns NULL on OOM */
    entry->freq = 1;
    entry->next = word_buckets[bucket];
    word_buckets[bucket] = entry;
    unique_words_count++;
}

static int compare_word_entries(const void *a, const void *b) {
    const WordEntry *const *wa = a;
    const WordEntry *const *wb = b;
    if ((*wa)->freq > (*wb)->freq) return -1;
    if ((*wa)->freq < (*wb)->freq) return 1;
    return strcmp((*wa)->word, (*wb)->word);
}

static void save_word_frequencies(const char *filepath) {
    if (unique_words_count == 0) return;

    WordEntry **arr = malloc(unique_words_count * sizeof(WordEntry *));
    if (!arr) return;

    uint32_t idx = 0;
    for (uint32_t i = 0; i < HASH_SIZE_WORDS; i++) {
        WordEntry *curr = word_buckets[i];
        while (curr) {
            arr[idx++] = curr;
            curr = curr->next;
        }
    }

    qsort(arr, unique_words_count, sizeof(WordEntry *), compare_word_entries);

    FILE *f = fopen(filepath, "w");
    if (f) {
        uint32_t limit = unique_words_count < 100000 ? unique_words_count : 100000;
        uint32_t words_written = 0;
        for (uint32_t i = 0; i < limit; i++) {
            if (arr[i]->freq >= 2) {
                fprintf(f, "%s %u\n", arr[i]->word, arr[i]->freq);
                words_written++;
            }
        }
        fclose(f);
        /* FIX: Report words_written (entries satisfying freq >= 2 that were
         * actually flushed to disk), not `limit` which is the number of
         * candidate entries examined.  Using `limit` overstated the count
         * whenever many low-frequency (freq < 2) words were discarded. */
        printf("Word frequency vocabulary saved to %s (%u unique words saved).\n", filepath, words_written);
    } else {
        perror("Failed to open word_vocab.txt for writing");
    }

    for (uint32_t i = 0; i < unique_words_count; i++) {
        free(arr[i]->word);
        free(arr[i]);
    }
    free(arr);
    memset(word_buckets, 0, sizeof(word_buckets));
    unique_words_count = 0;
}

static double get_time_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec / 1e9;
}

int main(int argc, char **argv) {
    if (argc < 4) {
        fprintf(stderr, "Usage: %s <model.bin> <corpus.txt> <snapshot_to_save.bin> [baseline_to_load.bin]\n", argv[0]);
        return 1;
    }

    const char *model_path = argv[1];
    const char *corpus_path = argv[2];
    const char *snapshot_path = argv[3];
    const char *baseline_path = (argc >= 5) ? argv[4] : NULL;

    printf("=== Gemini Continuous Self-Training Daemon ===\n");
    printf("Model: %s\n", model_path);
    printf("Corpus: %s\n", corpus_path);
    printf("Output Snapshot: %s\n", snapshot_path);
    if (baseline_path) {
        printf("Baseline Snapshot: %s\n", baseline_path);
    }

    /* Intercept SIGINT (^C) and SIGTERM (kill) cleanly */
    struct sigaction sa;
    sa.sa_handler = handle_shutdown_signal;
    sigemptyset(&sa.sa_mask);
    sa.sa_flags = 0; /* Do NOT use SA_RESTART; allow blocking stream I/O to drop out instantly */
    sigaction(SIGINT, &sa, NULL);
    sigaction(SIGTERM, &sa, NULL);

    int tok_handle = tok_load(model_path);
    if (tok_handle < 0) {
        fprintf(stderr, "Error: Failed to load tokenizer model.\n");
        return 1;
    }
    printf("Tokenizer loaded successfully.\n");

    EngineContext *ctx = NULL;
    if (baseline_path) {
        ctx = le_load_mmap(baseline_path, false);
        if (!ctx) {
            fprintf(stderr, "Error: Failed to load baseline snapshot from %s\n", baseline_path);
            tok_free(tok_handle);
            return 1;
        }
        printf("Gemini Linguistic Engine loaded from baseline snapshot.\n");
    } else {
        ctx = le_init();
        if (!ctx) {
            fprintf(stderr, "Error: Failed to initialize Gemini Engine.\n");
            tok_free(tok_handle);
            return 1;
        }
        printf("Gemini Linguistic Engine initialized (fresh state).\n");
    }
    ctx->lambda = 1.0f; /* Disable decay to ensure long-term memory persistence across the entire corpus */
    ctx->promotion_epochs = 1;
    ctx->rho_min = 0.82f; /* Standard quality gate threshold for semantic coherence */
    ctx->min_freq = 5;   /* Reduce frequency threshold for faster discovery in sub-corpora */

    /* CRITICAL FIX: Disable internal epoch triggering inside le_process_token.
     *
     * le_process_token fires le_begin_epoch automatically every ctx->epoch_size
     * TOKENS. The default value is DEFAULT_EPOCH_SIZE=10,000 tokens. With a
     * Luganda corpus averaging ~8 tokens per sentence, this fires every ~1,250
     * sentences — completely independently of online_learning.c's 100-sentence
     * boundary (EPOCH_INTERVAL).
     *
     * This double-firing has three harmful effects:
     *   1. le_begin_epoch runs twice per external epoch boundary: once internally
     *      at the token threshold, and once explicitly in this loop. The second
     *      call runs le_demote_stale_symbols on symbols the first call just
     *      promoted, creating a promote→demote loop within a single reporting window.
     *   2. syms_before and syms_after both capture the state AFTER the internal
     *      epoch has already promoted and potentially demoted symbols. The delta
     *      appears as 0 even though promotions are happening.
     *   3. le_begin_epoch is not reentrant-safe; double-firing increments
     *      current_epoch twice per intended epoch, corrupting stable_epochs math.
     *
     * Fix: Set epoch_size = UINT32_MAX to prevent (prev_tc + 1) % epoch_size
     * from ever reaching 0 within the lifetime of any plausible corpus.
     * Online_learning.c manages epochs explicitly via le_begin_epoch() at each
     * EPOCH_INTERVAL sentence boundary.
     *
     * Also set base_freq to match min_freq so calculate_dynamic_min_freq returns
     * a consistent threshold — base_freq defaults to 10.0f in le_init(), making
     * the effective_min_freq=10 for update_quality_flags() while our singleton
     * path correctly uses ctx->min_freq=5. Setting base_freq=5 makes both paths
     * use the same threshold, preventing rare words from being blocked by the
     * higher dynamic threshold. */
    ctx->epoch_size = UINT32_MAX;  /* Disable automatic internal epochs */
    ctx->base_freq  = (float)ctx->min_freq;  /* Align dynamic threshold with min_freq */

    FILE *f = fopen(corpus_path, "r");
    if (!f) {
        perror("Failed to open corpus");
        le_destroy(ctx);
        tok_free(tok_handle);
        return 1;
    }

    /* Pre-flight symbol capacity check. */
    {
        uint32_t existing_syms = get_symbol_count(ctx);
        uint32_t sym_cap = (uint32_t)MAX_SYMBOLS;
        if (existing_syms >= sym_cap) {
            fprintf(stderr,
                "ERROR: Symbol table already at capacity (%" PRIu32 "/%" PRIu32 "). "
                "All promotion attempts will be rejected. "
                "Recompile with a larger MAX_SYMBOLS before training.\n",
                existing_syms, sym_cap);
            fclose(f);
            le_destroy(ctx);
            tok_free(tok_handle);
            return 1;
        }
        if (existing_syms >= (uint32_t)(sym_cap * 0.9f)) {
            fprintf(stderr,
                "WARNING: Symbol table at %.0f%% capacity (%" PRIu32 "/%" PRIu32 "). "
                "Training will stall when the limit is reached. "
                "Consider recompiling with a larger MAX_SYMBOLS.\n",
                100.0f * existing_syms / sym_cap, existing_syms, sym_cap);
        }
    }

    char line[MAX_LINE_LEN];
    uint64_t sentences_processed = 0;
    uint32_t epoch_count = 0;
    double start_time = get_time_sec();

    printf("\nStarting continuous data ingestion stream...\n");

    /* Loop breaks cleanly if signal changes g_keep_running or file reaches EOF
     * Use select() with timeout to periodically check signal flag instead of
     * blocking indefinitely on fgets() which ignores signal interruption. */
    while (g_keep_running) {
        /* Check if data is available to read with 100ms timeout */
        fd_set read_fds;
        struct timeval tv;
        FD_ZERO(&read_fds);
        FD_SET(fileno(f), &read_fds);
        tv.tv_sec = 0;
        tv.tv_usec = 100000; /* 100ms timeout */

        int select_result = select(fileno(f) + 1, &read_fds, NULL, NULL, &tv);
        
        if (select_result < 0) {
            if (errno == EINTR) {
                /* Interrupted by signal - check g_keep_running and continue */
                continue;
            }
            perror("select() failed");
            break;
        }
        
        if (select_result == 0) {
            /* Timeout - check signal flag and loop back */
            continue;
        }
        
        /* Data available - read it */
        if (!fgets(line, sizeof(line), f)) {
            /* EOF or error */
            break;
        }

        size_t len = strlen(line);
        if (len > 0 && line[len - 1] == '\n') {
            line[len - 1] = '\0';
        }

        if (line[0] == '\0') continue;

        /* Ingest into the cognitive graph */
        engine_ingest_text(tok_handle, ctx, line, INVALID);

        /* Tokenize and insert into word frequency index */
        {
            char line_copy[MAX_LINE_LEN];
            strncpy(line_copy, line, MAX_LINE_LEN - 1);
            line_copy[MAX_LINE_LEN - 1] = '\0';
            char *tok = strtok(line_copy, " \t\r\n.,;:!?\"'()[]{}<>");
            while (tok) {
                add_word_to_freq(tok);
                tok = strtok(NULL, " \t\r\n.,;:!?\"'()[]{}<>");
            }
        }

        sentences_processed++;

        /* Trigger cognitive epoch */
        if (sentences_processed % EPOCH_INTERVAL == 0) {
            epoch_count++;
            printf("\n--- Epoch %u Boundary (%" PRIu64 " sentences ingested) ---\n", epoch_count, sentences_processed);
            
            uint32_t sccs_before = get_scc_count(ctx);
            double epoch_start = get_time_sec();

            /* CRITICAL FIX: Measure promotions via total_promotions delta, not symbol_count delta.
             *
             * symbol_count is a high-water mark (it never decreases when symbols are demoted —
             * demoted slots are not reclaimed). total_promotions is incremented in promote_scc and
             * decremented in demote_symbol, so it accurately reflects the NET number of active
             * promoted symbols. Its delta across le_begin_epoch gives the true count of NEW
             * net promotions this epoch (promotions minus demotions). */
            uint64_t prom_before = get_total_promotions(ctx);
            uint32_t syms_before = get_symbol_count(ctx);
            le_begin_epoch(ctx);
            uint64_t prom_after  = get_total_promotions(ctx);
            uint32_t syms_after  = get_symbol_count(ctx);

            /* Net promotions this epoch: positive = new symbols, negative = more demotions than promotions */
            int64_t net_promoted = (int64_t)prom_after - (int64_t)prom_before;
            uint32_t gross_promoted = syms_after - syms_before; /* High-water mark delta */

            if (net_promoted == 0 && syms_after >= MAX_SYMBOLS) {
                fprintf(stderr,
                    "WARNING [epoch %u]: Symbol table full (%" PRIu32 "/%" PRIu32 "). "
                    "Recompile with a larger MAX_SYMBOLS to continue promotion.\n",
                    epoch_count, syms_after, (uint32_t)MAX_SYMBOLS);
            }

            double epoch_end = get_time_sec();
            uint32_t sccs_after = get_scc_count(ctx);
            uint32_t active_symbols = (uint32_t)prom_after; /* total_promotions = net active symbols */

            printf("Epoch completed in %.3f seconds.\n", epoch_end - epoch_start);
            printf("SCCs: %u -> %u\n", sccs_before, sccs_after);
            printf("Symbols: net promoted this epoch: %" PRId64
                   ", new slots: %u, active total: %u\n",
                   net_promoted, gross_promoted, active_symbols);

            if (epoch_count % SNAPSHOT_INTERVAL == 0) {
                printf("Saving cognitive snapshot to %s...\n", snapshot_path);
                if (le_save_mmap(ctx, snapshot_path)) {
                    printf("Snapshot saved successfully.\n");
                } else {
                    fprintf(stderr, "WARNING: Failed to save snapshot.\n");
                }
                
                char vocab_path[256];
                snprintf(vocab_path, sizeof(vocab_path), "vocab_epoch_%u.txt", epoch_count);
                printf("Flushing word frequency index to %s (%u unique words)...\n", vocab_path, unique_words_count);
                save_word_frequencies(vocab_path);
            }
        }
    }

    fclose(f);
    double total_time = get_time_sec() - start_time;

    if (!g_keep_running) {
        printf("\n[DAEMON] Shutdown signal caught. Running final data consolidation...\n");
    }

    printf("\n=== Continuous Training Complete ===\n");
    printf("Total sentences: %" PRIu64 "\n", sentences_processed);
    printf("Total epochs: %u\n", epoch_count);
    double rate = (total_time > 1e-9) ? (double)sentences_processed / total_time : 0.0;
    printf("Total time: %.2f seconds (%.2f sentences/sec)\n", total_time, rate);
    
    printf("\nFinal cognitive state before forced epoch:\n");
    printf("  Nodes: %u\n", get_node_count(ctx));
    printf("  Edges: %u\n", get_edge_count(ctx));
    printf("  SCCs:  %u\n", get_scc_count(ctx));
    printf("  DAWG Symbols: %u\n", get_symbol_count(ctx));

    printf("\n=== Top 10 SCCs and Quality Metrics ===\n");

    uint32_t raw_scc_limit = atomic_load(&ctx->scc_count);
    
    if (!ctx->scc_nodes) {
        printf("ERROR: scc_nodes array is NULL - cannot display SCC metrics\n");
    } else if (raw_scc_limit == 0) {
        printf("No SCCs to display\n");
    } else if (raw_scc_limit > MAX_SCCS) {
        printf("ERROR: scc_count %u exceeds MAX_SCCS %u - likely corrupted\n",
               raw_scc_limit, MAX_SCCS);
        raw_scc_limit = MAX_SCCS;
    } else {
        uint32_t printed_scc = 0;
        for (uint32_t i = 0; i < raw_scc_limit && printed_scc < 10; i++) {
            SccNode *s = &ctx->scc_nodes[i];
            
            if (atomic_load_explicit(&s->member_count, memory_order_acquire) == 0) continue;
            
            float coherence = scc_load_coherence(s);
            float avg_entropy = scc_load_avg_entropy(s);
            if (!isfinite(coherence) || !isfinite(avg_entropy)) {
                printf("SCC %u: members: %u (invalid float values - skipping)\n",
                       i, atomic_load_explicit(&s->member_count, memory_order_acquire));
                continue;
            }
            
            uint64_t freq_val = atomic_load(&s->freq);
            printf("SCC %u: members: %u, coherence: %f, avg_entropy: %f, "
                   "freq: %" PRIu64 ", stable_epochs: %u, is_candidate: %d, is_weak: %d\n",
                   i, atomic_load_explicit(&s->member_count, memory_order_acquire),
                   coherence, avg_entropy,
                   freq_val, atomic_load_explicit(&s->stable_epochs, memory_order_relaxed),
                   atomic_load_explicit(&s->is_candidate, memory_order_relaxed),
                   atomic_load_explicit(&s->is_weak, memory_order_relaxed));
            printed_scc++;
        }
    }

    printf("\nForcing final epoch consolidation (3 iterations to advance stability counters)...\n");
    uint32_t syms_at_start = get_symbol_count(ctx);
    for (int i = 0; i < 3; i++) {
        le_begin_epoch(ctx);
    }
    uint32_t final_promoted = get_symbol_count(ctx) - syms_at_start;
    printf("Final consolidation promoted %u symbols.\n", final_promoted);

    printf("\nFinal cognitive state after epoch:\n");
    printf("  Nodes: %u\n", get_node_count(ctx));
    printf("  Edges: %u\n", get_edge_count(ctx));
    printf("  SCCs:  %u\n", get_scc_count(ctx));
    printf("  DAWG Symbols: %u\n", get_symbol_count(ctx));

    /* Final snapshot save */
    printf("\nSaving final cognitive snapshot to %s...\n", snapshot_path);
    if (le_save_mmap(ctx, snapshot_path)) {
        printf("Final snapshot saved successfully.\n");
    } else {
        fprintf(stderr, "WARNING: Failed to save final snapshot.\n");
    }

    char vocab_path[256];
    {
        const char *base = strrchr(snapshot_path, '/');
        base = base ? base + 1 : snapshot_path;
        char stem[200] = {0};
        strncpy(stem, base, sizeof(stem) - 1);
        char *dot = strrchr(stem, '.');
        if (dot) *dot = '\0';
        snprintf(vocab_path, sizeof(vocab_path), "vocab_%s.txt", stem);
    }
    save_word_frequencies(vocab_path);

    le_destroy(ctx);
    tok_free(tok_handle);

    return 0;
}