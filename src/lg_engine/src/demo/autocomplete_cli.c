#ifndef _GNU_SOURCE
#  define _GNU_SOURCE
#endif

#include "libgemini.h"
#include "tokenizer_api.h"
#include "../include/gemini_internal.h"
#include "../include/bridge_engine.h"
#include <pthread.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <termios.h>
#include <unistd.h>
#include <ctype.h>
#include <math.h>

/* Terminal handling */
static struct termios orig_termios;

typedef struct {
    char word[256];
    uint32_t freq;
} WordVocabEntry;

static WordVocabEntry *word_vocab = NULL;
static uint32_t word_vocab_size = 0;

static void load_word_vocab(const char *filepath) {
    FILE *f = fopen(filepath, "r");
    if (!f) {
        printf("No word frequency fallback index found at %s. Falling back to raw tokenizer vocabulary.\n", filepath);
        return;
    }

    uint32_t capacity = 1000;
    word_vocab = malloc(capacity * sizeof(WordVocabEntry));
    if (!word_vocab) {
        fclose(f);
        return;
    }

    char w[256];
    uint32_t freq;
    while (fscanf(f, "%255s %u", w, &freq) == 2) {
        if (word_vocab_size >= capacity) {
            capacity *= 2;
            WordVocabEntry *temp = realloc(word_vocab, capacity * sizeof(WordVocabEntry));
            if (!temp) {
                break;
            }
            word_vocab = temp;
        }
        strncpy(word_vocab[word_vocab_size].word, w, 255);
        word_vocab[word_vocab_size].word[255] = '\0';
        word_vocab[word_vocab_size].freq = freq;
        word_vocab_size++;
    }
    fclose(f);
    printf("Loaded %u words into word-frequency autocomplete fallback list.\n", word_vocab_size);
}

static void free_word_vocab(void) {
    if (word_vocab) {
        free(word_vocab);
        word_vocab = NULL;
    }
    word_vocab_size = 0;
}

void disable_raw_mode(void) {
    tcsetattr(STDIN_FILENO, TCSAFLUSH, &orig_termios);
}

/*
 * Signal handler for SIGINT / SIGTERM: restore the terminal before exit so
 * the shell is not left in raw mode.  These signals are delivered synchronously
 * in response to user action, making tcsetattr practically safe here.
 */
static void sig_handler(int sig) {
    /* Restore terminal synchronously — tcsetattr is async-signal-safe on Linux */
    tcsetattr(STDIN_FILENO, TCSAFLUSH, &orig_termios);
    /* Re-raise the signal with the default handler so the shell sees the
     * correct exit status (e.g. 130 for SIGINT). */
    signal(sig, SIG_DFL);
    raise(sig);
}

/*
 * F-3 FIX: Async-signal-safe crash handler for SIGSEGV, SIGABRT, SIGBUS, SIGFPE.
 *
 * The previous implementation called tcsetattr() inside crash handlers, which
 * triggers undefined behavior: tcsetattr requires standard library locks that
 * may be held by the faulting thread, potentially deadlocking the process and
 * preventing core dump generation.
 *
 * This handler uses only write() and _exit(), both guaranteed async-signal-safe
 * by POSIX. Terminal restoration is handled by atexit(disable_raw_mode) which
 * runs on normal exit paths.
 */
static void crash_signal_handler(int sig) {
    const char *msg = "\n[FATAL] Gemini Engine: unrecoverable fault. Terminating.\n";
    /* H-5 FIX: Correct byte count - string is 58 bytes (1 newline + 56 text + 1 newline) */
    (void)write(STDERR_FILENO, msg, 58);
    (void)sig;
    _exit(EXIT_FAILURE);
}

void enable_raw_mode(void) {
    /* HIGH FIX: Guard against non-terminal stdin (piped input, test harnesses).
     * tcgetattr on a non-tty returns ENOTTY and leaves orig_termios
     * uninitialised; the subsequent tcsetattr in disable_raw_mode would
     * corrupt the terminal of the parent process. */
    if (!isatty(STDIN_FILENO)) {
        fprintf(stderr, "WARNING: stdin is not a terminal — raw mode disabled.\n");
        return;
    }
    if (tcgetattr(STDIN_FILENO, &orig_termios) != 0) {
        perror("tcgetattr");
        return;
    }
    atexit(disable_raw_mode);

    /* Install signal handlers so Ctrl+C / kill restores the terminal */
    signal(SIGINT,  sig_handler);
    signal(SIGTERM, sig_handler);

    /* F-3 FIX: Use sigaction with SA_RESETHAND for crash signals.
     * SA_RESETHAND prevents handler loops: after the first invocation,
     * the signal disposition resets to SIG_DFL, allowing core dump generation
     * on re-raise. Only async-signal-safe calls are used in the handler. */
    struct sigaction sa;
    memset(&sa, 0, sizeof(sa));
    sa.sa_handler = crash_signal_handler;
    sa.sa_flags = SA_RESETHAND;

    sigaction(SIGSEGV, &sa, NULL);
    sigaction(SIGABRT, &sa, NULL);
    sigaction(SIGBUS,  &sa, NULL);
    sigaction(SIGFPE,  &sa, NULL);

    struct termios raw = orig_termios;
    raw.c_lflag &= ~(ECHO | ICANON);
    tcsetattr(STDIN_FILENO, TCSAFLUSH, &raw);
}

#define MAX_BUFFER 2048
#define MAX_PREDICTIONS 5

typedef struct {
    char word[256];
    float prob;
} Prediction;

void get_context_and_prefix(const char *buffer, char *context_word, char *prefix) {
    int len = (int)strlen(buffer);
    int last_space = -1;
    for (int i = len - 1; i >= 0; i--) {
        if (buffer[i] == ' ') {
            last_space = i;
            break;
        }
    }
    
    if (last_space == -1) {
        context_word[0] = '\0';
        /* HIGH FIX: was strcpy(prefix, buffer) — unchecked copy into a 256-byte
         * output buffer. buffer can be up to MAX_BUFFER-1 (2047) bytes, overflowing
         * prefix[256] and corrupting the stack frame. Use snprintf instead. */
        snprintf(prefix, 256, "%s", buffer);
    } else {
        /* Extract the word exactly before the last space */
        int prev_space = -1;
        for (int i = last_space - 1; i >= 0; i--) {
            if (buffer[i] == ' ') {
                prev_space = i;
                break;
            }
        }
        int word_start = prev_space + 1;
        int word_len = last_space - word_start;
        /* CRITICAL FIX: Check for negative word_len (multiple consecutive spaces)
         * If word_len <= 0, casting to size_t makes it a huge positive integer,
         * causing strncpy to segfault by reading past buffer bounds. */
        if (word_len <= 0) {
            context_word[0] = '\0';
        } else {
            if (word_len >= 255) word_len = 255;
            strncpy(context_word, buffer + word_start, (size_t)word_len);
            context_word[word_len] = '\0';
        }
        
        /* HIGH FIX: same overflow risk as above */
        snprintf(prefix, 256, "%s", buffer + last_space + 1);
    }
}

int main(int argc, char **argv) {
    if (argc < 3) {
        printf("Usage: %s <engine_model.bin> <tokenizer_model.bin> [vocab_path.txt]\n", argv[0]);
        return 1;
    }

    const char *model_path = argv[1];
    const char *tok_path = argv[2];
    const char *vocab_path = (argc >= 4) ? argv[3] : "vocab/word_vocab.txt";

    printf("Loading Gemini Engine...\n");
    EngineContext *ctx = le_load_mmap(model_path, false);
    if (!ctx) {
        fprintf(stderr, "Failed to load engine model %s\n", model_path);
        return 1;
    }
    le_print_stats(ctx);

    printf("Loading Tokenizer...\n");
    int tok_handle = tok_load(tok_path);
    if (tok_handle < 0) {
        fprintf(stderr, "Failed to load tokenizer model %s\n", tok_path);
        le_unload_mmap(ctx);
        return 1;
    }
    /* L-4 FIX: Allow custom vocab path via command-line argument */
    load_word_vocab(vocab_path);

    printf("\n=== Promoted DAWG Symbols ===\n");
    uint32_t sym_cnt = get_symbol_count(ctx);
    printf("Total promoted symbols: %u\n", sym_cnt);
    
    /* CRITICAL FIX: Validate symbol_count is reasonable */
    if (sym_cnt == 0) {
        printf("No promoted symbols to display\n");
    } else if (sym_cnt > MAX_SYMBOLS) {
        printf("ERROR: symbol_count %u exceeds MAX_SYMBOLS %u - likely corrupted\n", sym_cnt, MAX_SYMBOLS);
        sym_cnt = MAX_SYMBOLS;
    }
    
    /* CRITICAL FIX: Validate dawg_nodes array is allocated */
    if (!ctx->dawg_nodes) {
        printf("ERROR: dawg_nodes array is NULL - cannot iterate symbols\n");
    } else if (!ctx->scc_nodes) {
        printf("ERROR: scc_nodes array is NULL - cannot iterate symbols\n");
    } else {
    
    for (uint32_t i = 0; i < sym_cnt; i++) {
        /* CRITICAL FIX: Validate symbol index before accessing dawg_nodes array */
        if (i >= MAX_SYMBOLS) {
            printf("Symbol index %u exceeds MAX_SYMBOLS - breaking loop\n", i);
            break;
        }

        Symbol *sym = &ctx->dawg_nodes[i];

        /* CRITICAL FIX: Validate canonical node before accessing fields */
        NodeID canonical_node = sym->canonical_node;
        if (canonical_node == INVALID || canonical_node >= MAX_NODES) {
            printf("Symbol %u has invalid canonical_node (0x%x) - skipping\n", i, canonical_node);
            continue;
        }

        /* Get current SCC from canonical node */
        SccID current_scc = atomic_load_explicit(&ctx->transient_nodes[canonical_node].scc_id, memory_order_acquire);
        if (current_scc == INVALID || current_scc >= MAX_SCCS) {
            printf("Symbol %u (canonical node: %u has invalid SCC - skipping)\n", i, canonical_node);
            continue;
        }

        /* CRITICAL FIX: Also check against actual SCC count to prevent accessing unallocated SCCs */
        uint32_t scc_cnt = atomic_load(&ctx->scc_count);
        if (current_scc >= scc_cnt) {
            printf("Symbol %u (current SCC: %u >= scc_count %u - skipping)\n", i, current_scc, scc_cnt);
            continue;
        }

        NodeID member_buf[128];
        uint32_t mc = le_get_symbol_nodes(ctx, i, member_buf, 128);

        /* CRITICAL FIX: le_get_symbol_nodes returns 0 for invalid SCC - skip access */
        if (mc == 0) {
            printf("Symbol %u (current SCC: %u - le_get_symbol_nodes returned 0, skipping)\n", i, current_scc);
            continue;
        }

        SccNode *scc = &ctx->scc_nodes[current_scc];

        /* CRITICAL FIX: Validate SCC structure before accessing fields */
        if (atomic_load_explicit(&scc->head, memory_order_acquire) == INVALID && atomic_load_explicit(&scc->member_count, memory_order_acquire) == 0) {
            printf("Symbol %u (current SCC: %u - SCC appears uninitialized, skipping)\n", i, current_scc);
            continue;
        }

        /* CRITICAL FIX: Validate SCC fields are reasonable before printing */
        float coherence = scc_load_coherence(scc);
        float avg_entropy = scc_load_avg_entropy(scc);
        if (!isfinite(coherence) || !isfinite(avg_entropy)) {
            printf("Symbol %u (current SCC: %u - SCC has invalid float values, skipping)\n", i, current_scc);
            continue;
        }

        printf("Symbol %u (canonical node: %u, current SCC: %u, members: %u, coherence: %f, entropy: %f, stable_epochs: %u, is_forced: %d): ",
               i, canonical_node, current_scc,
               atomic_load_explicit(&scc->member_count, memory_order_acquire),
               coherence, avg_entropy, atomic_load_explicit(&scc->stable_epochs, memory_order_relaxed), atomic_load_explicit(&scc->is_forced, memory_order_relaxed));
        for (uint32_t m = 0; m < mc; m++) {
            TokenID m_tid = get_node_token(ctx, member_buf[m]);
            const char *decoded = tok_decode(tok_handle, m_tid);
            
            /* FIX: Skip corrupted decoded strings - tokenizer string pool may be corrupted */
            if (!decoded || strcmp(decoded, "<?>") == 0 || strlen(decoded) < 1) {
                printf("'%s' (ID %u - CORRUPTED, skipping)\n", decoded ? decoded : "???", m_tid);
                continue;
            }
            
            printf("'%s' (ID %u, hex: ", decoded, m_tid);
            for (int h = 0; decoded[h]; h++) {
                printf("%02x ", (unsigned char)decoded[h]);
            }
            printf(") ");
        }
        printf("\n");
        
        /* Diagnostic: verify the integrity of promoted units */
        if (mc > 1) {
            printf("  [DIAGNOSTIC: SCC successfully condensed - %u nodes merged into single symbol]\n", mc);
        } else if (mc == 1) {
            printf("  [DIAGNOSTIC: Single-node symbol - no merging occurred]\n");
        } else {
            printf("  [DIAGNOSTIC: WARNING - Empty symbol with %u members]\n", mc);
        }
        
        /* C-4 FIX: DawgTransition.next is _Atomic uint32_t. All reads must use 
         * atomic_load_explicit with acquire ordering. */
        uint32_t t_idx = atomic_load_explicit(&sym->first_transition, memory_order_acquire);
        if (t_idx != INVALID) {
            printf("  Transitions:\n");
            while (t_idx != INVALID) {
                DawgTransition *t = &ctx->dawg_transitions[t_idx];
                float w = atomic_load_float(&t->weight);
                uint32_t raw_bits;
                memcpy(&raw_bits, &w, 4);
                printf("    -> Symbol %u (weight: %f, raw: 0x%08x)\n", t->target, w, raw_bits);
                t_idx = atomic_load_explicit(&t->next, memory_order_acquire);
            }
        } else {
            printf("  Transitions: None\n");
        }
    }
    }

    printf("\n=== Luganda Predictive Autocomplete CLI ===\n");
    printf("Type sentences to see DAWG predictions.\n");
    printf("Press [TAB] to autocomplete.\n");
    printf("Press [ESC] or Ctrl+C to exit.\n\n");

    enable_raw_mode();

    char buffer[MAX_BUFFER] = {0};
    int buf_len = 0;
    
    Prediction preds[MAX_PREDICTIONS];
    int pred_count = 0;
    uint32_t last_epoch = get_current_epoch(ctx);

    while (1) {
        /* MEDIUM FIX: Read the epoch counter directly via get_current_epoch() */
        uint32_t curr_epoch = get_current_epoch(ctx);
        if (curr_epoch != last_epoch) {
            last_epoch = curr_epoch;
            pred_count = 0;
            /* Clear screen below prompt to clean up previous predictions */
            printf("\r\033[K> %s\n\033[K  \033[36m[Epoch advanced to %u. Refreshed transition pointers.]\033[0m", buffer, curr_epoch);
            /* Move cursor back up to the prompt line */
            printf("\033[1A\r\033[%dC", buf_len + 2);
            fflush(stdout);
            continue;
        }

        /* Clear screen below prompt and render */
        printf("\r\033[K> %s", buffer);
        
        /* Render predictions below */
        printf("\n\033[K"); /* clear line */
        if (pred_count > 0) {
            printf("  \033[90m"); /* gray color for predictions */
            for (int i = 0; i < pred_count; i++) {
                printf("[%d] %s  ", i+1, preds[i].word);
            }
            printf("\033[0m");
        } else {
            char debug_context[256] = {0};
            char debug_prefix[256] = {0};
            get_context_and_prefix(buffer, debug_context, debug_prefix);
            if (strlen(debug_context) > 0) {
                uint32_t debug_tokens[32];
                int c_toks = tok_encode(tok_handle, debug_context, debug_tokens, 32);
                if (c_toks > 0) {
                    printf("  \033[90m[Context: '%s' -> Last Token ID: %u]\033[0m", debug_context, debug_tokens[c_toks-1]);
                }
            }
        }
        
        /* Move cursor back up to the prompt line */
        printf("\033[1A\r\033[%dC", buf_len + 2);
        fflush(stdout);

        char c;
        if (read(STDIN_FILENO, &c, 1) != 1) break;

        if (c == 27) { /* ESC */
            break;
        } else if (c == 127 || c == '\b') { /* Backspace */
            if (buf_len > 0) {
                buffer[--buf_len] = '\0';
            }
        } else if (c == '\t') { /* Tab */
            if (pred_count > 0) {
                /* Find the prefix length we need to overwrite */
                char context[256], prefix[256];
                get_context_and_prefix(buffer, context, prefix);
                int prefix_len = (int)strlen(prefix);
                
                /* HIGH FIX: preds[].word may store the raw BPE token which can include
                 * a leading space (e.g. " abc"). The prefix comparison was performed on
                 * cmp_w (with the leading space stripped), so the correct completion
                 * suffix starts at cmp_w+prefix_len, not preds[0].word+prefix_len. */
                const char *base = preds[0].word;
                if (base[0] == ' ') base++;   /* strip BPE leading space */
                const char *to_add = base + prefix_len;
                int add_len = (int)strlen(to_add);
                if (buf_len + add_len + 1 < MAX_BUFFER) {
                    strcat(buffer, to_add);
                    strcat(buffer, " ");
                    buf_len += add_len + 1;
                }
            }
        } else if (c == '\n' || c == '\r') { /* Enter */
            printf("\n");
            buffer[0] = '\0';
            buf_len = 0;
            pred_count = 0;
            continue;
        } else if (isprint(c)) {
            if (buf_len < MAX_BUFFER - 1) {
                buffer[buf_len++] = c;
                buffer[buf_len] = '\0';
            }
        } else {
            continue;
        }

        /* Update predictions based on current buffer */
        pred_count = 0;
        
        char context[256] = {0};
        char prefix[256] = {0};
        get_context_and_prefix(buffer, context, prefix);
        
        if (ctx->node_hash && ctx->nodes && strlen(context) > 0) {
            uint32_t context_tokens[32];
            int c_toks = tok_encode(tok_handle, context, context_tokens, 32);
            if (c_toks > 0) {
                TokenID tid = context_tokens[c_toks - 1]; /* last token of context */
                
                /* Look up the current NodeID for tid.
                 * CRITICAL FIX: Cap open-addressing scan at HASH_SIZE steps */
                NodeID current_node = INVALID;
                {
                    /* CRITICAL FIX C-5: Use Fibonacci hash to match le_get_or_create_node
                 * Simple modulo hash causes lookup to start at different bucket than
                 * where node was inserted, making context predictions fail. */
                uint32_t h = (tid * 0x9E3779B1u) % HASH_SIZE;
                    uint32_t steps = 0;
                    NodeID slot;
                    while (steps < HASH_SIZE &&
                           (slot = atomic_load_explicit(&ctx->node_hash[h],
                                                        memory_order_acquire)) != INVALID) {
                        if (ctx->nodes[slot].token_id == tid) {
                            current_node = slot;
                            break;
                        }
                        h = (h + 1) % HASH_SIZE;
                        steps++;
                    }
                }

                if (current_node != INVALID) {
                    char predict_words[MAX_PREDICTIONS][256] = {{0}};
                    float predict_probs[MAX_PREDICTIONS] = {0.0f};
                    int count = dawg_predict(tok_handle, ctx, current_node, 0.0f, MAX_PREDICTIONS, predict_words, predict_probs);

                    // If immediate predictions are thin or weak, look deeper down the graph branch
                    if (count == 0 || predict_probs[0] < 0.15f) {
                        char multi_preds[MAX_PREDICTIONS][256];
                        float multi_probs[MAX_PREDICTIONS];
                        
                        int multi_count = dawg_predict_multi_hop(tok_handle, ctx, current_node, 
                                                                0.0f, MAX_PREDICTIONS, 3, // Look ahead up to 3 tokens deep
                                                                multi_preds, multi_probs);
                        if (multi_count > count) {
                            // Merge or replace low-confidence immediate tokens with long-horizon paths
                            for (int m = 0; m < multi_count && count < MAX_PREDICTIONS; m++) {
                                strncpy(predict_words[count], multi_preds[m], 255);
                                predict_probs[count] = multi_probs[m];
                                count++;
                            }
                        }
                    }

                    for (int i = 0; i < count && pred_count < MAX_PREDICTIONS; i++) {
                        char w[256];
                        strncpy(w, predict_words[i], 255);
                        w[255] = '\0';
                        
                        const char *cmp_w = w;
                        if (cmp_w[0] == ' ') cmp_w++;
                        
                        /* Check prefix */
                        if (strlen(prefix) == 0 || strncmp(cmp_w, prefix, strlen(prefix)) == 0) {
                            // Avoid duplicates
                            bool dup = false;
                            for (int p = 0; p < pred_count; p++) {
                                if (strcmp(preds[p].word, w) == 0) {
                                    dup = true;
                                    break;
                                }
                            }
                            if (!dup) {
                                
                                snprintf(preds[pred_count].word, sizeof(preds[pred_count].word), "%s", w);
                                preds[pred_count].word[sizeof(preds[pred_count].word) - 1] = '\0';
                                preds[pred_count].prob = predict_probs[i];
                                pred_count++;
                            }
                        }
                    }
                }
            }
        } else if (ctx->dawg_nodes && ctx->scc_nodes && ctx->nodes && strlen(prefix) > 0) {
            /* Fix: If there's no context, look up DAWG symbols matching the prefix directly */
            uint32_t sym_cnt = get_symbol_count(ctx);
            for (uint32_t i = 0; i < sym_cnt && pred_count < MAX_PREDICTIONS; i++) {
                NodeID member_buf[32];
                uint32_t mc = le_get_symbol_nodes(ctx, i, member_buf, 32);
                
                for (uint32_t m = 0; m < mc; m++) {
                    TokenID m_tid = get_node_token(ctx, member_buf[m]);
                    char w[256] = {0};
                    const char *decoded = tok_decode(tok_handle, m_tid);
                    /* FIX: Skip corrupted decoded strings - tokenizer string pool may be corrupted */
                    if (!decoded || strcmp(decoded, "<?>") == 0 || strlen(decoded) < 1) continue;
                    strncpy(w, decoded, sizeof(w) - 1);
                    
                    const char *cmp_w = w;
                    if (cmp_w[0] == ' ') cmp_w++;
                    
                    if (strncmp(cmp_w, prefix, strlen(prefix)) == 0) {
                        /* Avoid duplicates */
                        bool dup = false;
                        for (int p = 0; p < pred_count; p++) {
                            if (strcmp(preds[p].word, w) == 0) {
                                dup = true;
                                break;
                            }
                        }
                        if (!dup) {
                            snprintf(preds[pred_count].word, sizeof(preds[pred_count].word), "%s", w);
                            preds[pred_count].word[sizeof(preds[pred_count].word) - 1] = '\0';
                            preds[pred_count].prob = 1.0f;
                            pred_count++;
                            /* M-5 FIX: Remove inner break to allow multiple nodes per symbol */
                        }
                    }
                }
            }
        }
        
        /* Fallback: check word-frequency fallback index, then base tokenizer vocab */
        if (strlen(prefix) > 0 && pred_count < MAX_PREDICTIONS) {
            /* OPTIMIZED FALLBACK: Prioritize the sorted frequency index over raw vocabulary IDs */
            if (word_vocab && word_vocab_size > 0) {
                for (uint32_t i = 0; i < word_vocab_size && pred_count < MAX_PREDICTIONS; i++) {
                    const char *cmp_w = word_vocab[i].word;
                    if (cmp_w[0] == ' ') cmp_w++; // Strip tokenized space prefix
                    
                    if (strncmp(cmp_w, prefix, strlen(prefix)) == 0) {
                        bool dup = false;
                        for (int p = 0; p < pred_count; p++) {
                            if (strcmp(preds[p].word, word_vocab[i].word) == 0) {
                                dup = true;
                                break;
                            }
                        }
                        if (!dup) {
                            snprintf(preds[pred_count].word, sizeof(preds[pred_count].word), "%s", word_vocab[i].word);
                            // Dynamically scale probability based on corpus frequency weights
                            preds[pred_count].prob = 0.01f + ((float)word_vocab[i].freq / 100000.0f);
                            pred_count++;
                        }
                    }
                }
            }
            
         // Hard fallback to token array sequence if we still have available prediction slots
         if (pred_count < MAX_PREDICTIONS) {
                uint32_t vocab_limit = tok_vocab_size(tok_handle);
                for (uint32_t i = 0; i < vocab_limit && pred_count < MAX_PREDICTIONS; i++) {
                    const char *decoded = tok_decode(tok_handle, i);
                    /* CRITICAL FIX: Sparse vocabularies contain NULL slots for
                     * duplicate strings omitted during build. A break here aborted
                     * the entire prefix search on the very first gap. Use continue. */
                    if (!decoded) continue;
                    
                    /* FIX: Skip corrupted decoded strings - tokenizer string pool may be corrupted */
                    if (strcmp(decoded, "<?>") == 0 || strlen(decoded) < 1) continue;
                    
                    if (decoded[0] == '\0' || decoded[0] == '[') continue;
                    
                    const char *cmp_w = decoded;
                    if (cmp_w[0] == ' ') cmp_w++;
                    
                    if (strncmp(cmp_w, prefix, strlen(prefix)) == 0) {
                        bool dup = false;
                        for (int p = 0; p < pred_count; p++) {
                            if (strcmp(preds[p].word, decoded) == 0) {
                                dup = true;
                                break;
                            }
                        }
                        if (!dup) {
                            snprintf(preds[pred_count].word, sizeof(preds[pred_count].word), "%s", decoded);
                            preds[pred_count].prob = 0.01f;
                            pred_count++;
                        }
                    }
                }
            }
        }
    }

    printf("\n\033[KExiting...\n");
    free_word_vocab();
    tok_free(tok_handle);
    le_unload_mmap(ctx);
    return 0;
}