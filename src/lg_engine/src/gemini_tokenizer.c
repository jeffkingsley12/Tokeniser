/*
===============================================================================
GEMINI FAST-TRIE TOKENIZER
Compact Array-Based Ternary Search Tree (TST) + Streaming Word Buffer

Architecture:
  - Array indices (not pointers) → mmap-compatible, zero-copy persistence
  - O(K) lookup where K = string length
  - Streaming character buffer with word-break detection
  - Flat ID→string reverse table for Ghost Text output

Compiled as part of libgemini.so via Makefile
===============================================================================
*/

#ifndef _GNU_SOURCE
#  define _GNU_SOURCE
#endif

#include <string.h>
#include <stdio.h>
#include <stdint.h>
#include <stdbool.h>
#include <ctype.h>

/* ============================= CONFIGURATION =============================== */

#include "gemini_internal.h"
#include "libgemini.h"
#include "lexeme_intern.h"
#include <stdlib.h>

#define MAX_TRIE_NODES   1000000
#define MAX_VOCAB        200000
#define MAX_WORD_LEN     128
#define MAX_STRING_POOL  (MAX_VOCAB * 32)  /* ~6.4 MB for word storage */
#define TST_NIL          0

/* ============================= DATA STRUCTURES ============================= */

/**
 * TSTNode - a single node in the Ternary Search Tree.
 * Stored in a flat array for cache locality and mmap persistence.
 */
typedef struct {
    char     c;             /* Character at this node */
    uint32_t left;          /* Index: chars < c */
    uint32_t mid;           /* Index: chars == c (next char in word) */
    uint32_t right;         /* Index: chars > c */
    uint32_t token_id;      /* 0 if no word ends here; else the assigned ID */
} TSTNode;

/**
 * Tokenizer - complete tokenizer state, fully mmap-serialisable.
 *
 * Thread safety (Issue 9 fix):
 *   All public entry-points (tokenizer_get_id, tokenizer_get_word) must hold
 *   `lock` for the duration of the call.  The struct is intentionally NOT
 *   mmap'd as a flat blob after this change — save/load use the fwrite/fread
 *   path and skip the mutex field.
 */
struct Tokenizer {
    /* Header / validation */
    uint32_t magic;            /* 0x54535420 "TST " */
    uint32_t version;

    /* TST core */
    TSTNode  nodes[MAX_TRIE_NODES];
    uint32_t node_count;       /* Next free node slot (0 is root sentinel) */
    uint32_t token_sequence;   /* Monotonically increasing ID counter */

    /* Reverse lookup: id → string */
    char     string_pool[MAX_STRING_POOL];
    uint32_t pool_used;        /* Bytes consumed in string_pool */
    uint32_t id_to_offset[MAX_VOCAB];  /* string_pool offset for each ID */

    /* Streaming word buffer */
    char     word_buf[MAX_WORD_LEN];
    uint32_t word_len;         /* Current chars in buffer */

    /* Issue 9: per-tokenizer mutex for concurrent callers */
    pthread_mutex_t lock;
};


/* ============================= LIFECYCLE =================================== */

/**
 * Initialise a new tokenizer.  Returns NULL on allocation failure or mutex init failure.
 */
Tokenizer* tokenizer_init(void) {
    Tokenizer* t = (Tokenizer*)calloc(1, sizeof(Tokenizer));
    if (!t) return NULL;

    t->magic   = 0x54535420;  /* "TST " */
    t->version = 1;
    t->node_count = 1;        /* Slot 0 is the root sentinel (unused data) */
    t->token_sequence = 0;
    t->pool_used = 0;
    t->word_len  = 0;
    
    /* HIGH FIX (Issue #21): Check pthread_mutex_init return value */
    if (pthread_mutex_init(&t->lock, NULL) != 0) {
        fprintf(stderr, "ERROR: Failed to initialize tokenizer mutex\n");
        free(t);
        return NULL;
    }
    return t;
}

void tokenizer_destroy(Tokenizer* t) {
    if (!t) return;
    pthread_mutex_destroy(&t->lock);
    free(t);
}

/* ============================= TST CORE ==================================== */

/**
 * Get or create a token ID for `word`.
 *
 * If create_if_missing is true and the word is new, it is inserted into
 * the TST, assigned a new monotonic ID, and its string is copied into
 * the string pool for reverse lookup.
 *
 * Returns 0 if the word is not found and creation is not requested.
 */
uint32_t tokenizer_get_id(Tokenizer* t, const char* word, bool create_if_missing) {
    if (!t || !word) return 0;
    
    /* Phase 1: Skip leading whitespace */
    while (*word && isspace((unsigned char)*word)) {
        word++;
    }
    if (*word == '\0') return 0;

    /* Phase 2: Copy characters until the first embedded whitespace */
    char clean_word[MAX_WORD_LEN];
    int len = 0;
    while (word[len] && len < MAX_WORD_LEN - 1) {
        if (isspace((unsigned char)word[len])) {
            break;
        }
        clean_word[len] = word[len];
        len++;
    }
    clean_word[len] = '\0';
    word = clean_word;

    pthread_mutex_lock(&t->lock);

    /* The root sentinel node's `mid` is the entry point */
    uint32_t* curr = &t->nodes[0].mid;
    const char* p = word;
    size_t word_len = 0;  /* Track length during traversal to avoid second strlen (Issue #22) */
    /* M-5 FIX: Bound TST traversal to prevent infinite loop on corrupted node
     * array (e.g. after a truncated tokenizer_load). Each iteration consumes
     * one character or follows one lateral branch; MAX_TRIE_NODES iterations is
     * a safe upper bound for any valid word of length <= MAX_WORD_LEN. */
    uint32_t tst_steps = 0;

    while (*p) {
        if (tst_steps++ >= MAX_TRIE_NODES) {
            fprintf(stderr, "ERROR: TST traversal limit exceeded — possible cycle in node array\n");
            pthread_mutex_unlock(&t->lock);
            return 0;
        }
        if (*curr == TST_NIL) {
            if (!create_if_missing) { pthread_mutex_unlock(&t->lock); return 0; }

            /* Create new node on the fly */
            if (t->node_count >= MAX_TRIE_NODES) {
                fprintf(stderr, "ERROR: TST node limit reached\n");
                pthread_mutex_unlock(&t->lock);
                return UINT32_MAX;
            }
            *curr = t->node_count++;
            memset(&t->nodes[*curr], 0, sizeof(TSTNode));
            t->nodes[*curr].c = *p;
        }

        TSTNode* node = &t->nodes[*curr];

        if (*p < node->c) {
            curr = &node->left;
        } else if (*p > node->c) {
            curr = &node->right;
        } else {
            /* Character match — advance to next character */
            p++;
            word_len++;
            if (*p == '\0') {
                /* End of word */
                if (node->token_id == 0 && create_if_missing) {
                    if (t->token_sequence >= MAX_VOCAB - 1) {
                        fprintf(stderr, "ERROR: Vocab limit reached\n");
                        pthread_mutex_unlock(&t->lock);
                        return UINT32_MAX;
                    }
                    node->token_id = ++t->token_sequence;

                    /* Copy word into string pool for reverse lookup */
                    /* M-4 FIX: Renamed from `len` to `str_copy_len` to eliminate shadowing
                     * of the outer `int len` variable used during clean_word construction. */
                    size_t str_copy_len = word_len + 1;
                    /* CRITICAL FIX (Issue #14): Check pool capacity before assigning token ID
                     * to prevent silent aliasing */
                    if (t->pool_used + str_copy_len > MAX_STRING_POOL) {
                        fprintf(stderr, "ERROR: String pool exhausted (used %u + %zu > %u)\n",
                                t->pool_used, str_copy_len, MAX_STRING_POOL);
                        /* Rollback: reset token_id since we couldn't store the string */
                        node->token_id = 0;
                        t->token_sequence--;  /* Undo increment */
                        pthread_mutex_unlock(&t->lock);
                        return UINT32_MAX;
                    }
                    t->id_to_offset[node->token_id] = t->pool_used;
                    memcpy(&t->string_pool[t->pool_used], word, str_copy_len);
                    t->pool_used += (uint32_t)str_copy_len;
                }
                uint32_t result = node->token_id;
                pthread_mutex_unlock(&t->lock);
                return result;
            }
            curr = &node->mid;
        }
    }
    pthread_mutex_unlock(&t->lock);
    return 0;
}

/**
 * Reverse lookup: token ID → original word string.
 * Returns "<?>" for unknown IDs.
 * 
 * CRITICAL FIX (Issue #15): The check `id > t->token_sequence` was read without the lock,
 * creating a TOCTOU race. A concurrent tokenizer_get_id call that increments token_sequence
 * between this check and the lock may result in an out-of-bounds id_to_offset access.
 * Moved the check inside the lock for atomicity.
 * 
 * NOTE: Returned pointer is only valid while the Tokenizer exists. Callers that hold
 * this pointer after tokenizer_destroy() will have a dangling pointer.
 */
const char* tokenizer_get_word(Tokenizer* t, uint32_t id) {
    if (!t || id == 0) return "<?>";
    
    pthread_mutex_lock(&t->lock);
    if (id > t->token_sequence) {
        pthread_mutex_unlock(&t->lock);
        return "<?>";
    }
    /* FIX (Issue #18 addendum): Bounds-check the offset before dereference.
     * A corrupt save file could have id_to_offset[id] >= pool_used. */
    if (t->id_to_offset[id] >= t->pool_used) {
        pthread_mutex_unlock(&t->lock);
        return "<?>";
    }
    const char* result = &t->string_pool[t->id_to_offset[id]];
    pthread_mutex_unlock(&t->lock);
    return result;
}



/* ============================= STREAMING BUFFER & UTF-8 ==================== */

/**
 * StreamContext for zero-heap UTF-8 decoding and normalization.
 * Holds transient state for a single streaming session.
 */
typedef struct {
    char buffer[MAX_WORD_LEN];  /* Normalized UTF-8 accumulator */
    uint32_t len;

    uint32_t codepoint;         /* Current Unicode codepoint being assembled */
    uint8_t utf8_remaining;     /* Bytes needed to complete codepoint */
    uint8_t utf8_total;         /* Total bytes in current sequence for overlong check */

    Tokenizer *tokenizer;
    EngineContext *gemini;
    NodeID prev_node;           /* Current graph position */
} StreamContext;

/**
 * Zero-heap UTF-8 decoder with validation.
 * Returns 1 if a full codepoint is ready (stored in *out_cp), 0 otherwise.
 * Rejects overlong encodings, surrogates, and out-of-range codepoints (Issue #19).
 * Recovers from bad continuation by restarting at the invalid byte (Issue #20).
 * FIX: Reset utf8_total along with utf8_remaining on bad continuation bytes
 * so overlong checks use the correct sequence length.
 */
static int utf8_decode(StreamContext *ctx, unsigned char byte, uint32_t *out_cp) {
    if (ctx->utf8_remaining == 0) {
        if (byte < 0x80) {
            *out_cp = byte;
            return 1;
        }
        else if ((byte & 0xE0) == 0xC0) {
            ctx->codepoint = byte & 0x1F;
            ctx->utf8_remaining = 1;
            ctx->utf8_total = 2;
        }
        else if ((byte & 0xF0) == 0xE0) {
            ctx->codepoint = byte & 0x0F;
            ctx->utf8_remaining = 2;
            ctx->utf8_total = 3;
        }
        else if ((byte & 0xF8) == 0xF0) {
            ctx->codepoint = byte & 0x07;
            ctx->utf8_remaining = 3;
            ctx->utf8_total = 4;
        }
        else {
            return 0; /* Invalid - drop */
        }
        return 0;
    }

    if ((byte & 0xC0) != 0x80) {
        /* HIGH FIX (Issue #20): Bad continuation byte — recover by treating it as start of new sequence.
         * FIX: Also reset utf8_total so the overlong check uses correct byte count. */
        ctx->utf8_remaining = 0;
        ctx->utf8_total = 0;
        /* Restart state machine with this byte */
        if (byte < 0x80) {
            *out_cp = byte;
            return 1;
        }
        else if ((byte & 0xE0) == 0xC0) {
            ctx->codepoint = byte & 0x1F;
            ctx->utf8_remaining = 1;
            ctx->utf8_total = 2;
        }
        else if ((byte & 0xF0) == 0xE0) {
            ctx->codepoint = byte & 0x0F;
            ctx->utf8_remaining = 2;
            ctx->utf8_total = 3;
        }
        else if ((byte & 0xF8) == 0xF0) {
            ctx->codepoint = byte & 0x07;
            ctx->utf8_remaining = 3;
            ctx->utf8_total = 4;
        }
        return 0;
    }

    ctx->codepoint = (ctx->codepoint << 6) | (byte & 0x3F);
    ctx->utf8_remaining--;

    if (ctx->utf8_remaining == 0) {
        /* HIGH FIX (Issue #19): Validate codepoint for overlong, surrogates, and out-of-range */
        if (ctx->codepoint > 0x10FFFF) {
            /* Out of valid Unicode range */
            return 0;
        }
        if (ctx->codepoint >= 0xD800 && ctx->codepoint <= 0xDFFF) {
            /* UTF-16 surrogate — invalid in UTF-8 per RFC 3629 */
            return 0;
        }
        
        /* Overlong check: verify codepoint is within the minimum range for its byte length */
        if (ctx->utf8_total == 2 && ctx->codepoint < 0x80) return 0;
        if (ctx->utf8_total == 3 && ctx->codepoint < 0x800) return 0;
        if (ctx->utf8_total == 4 && ctx->codepoint < 0x10000) return 0;
        *out_cp = ctx->codepoint;
        return 1;
    }

    return 0;
}

static inline bool is_unicode_space(uint32_t cp) {
    /* MEDIUM FIX (Issue #24): Include Unicode spaces beyond ASCII */
    if (cp == 0x20 || cp == 0x0A || cp == 0x09 || cp == 0x0D) return true;  /* ASCII space, LF, TAB, CR */
    if (cp == 0x00A0) return true;  /* U+00A0 NO-BREAK SPACE */
    if (cp >= 0x2000 && cp <= 0x200A) return true;  /* U+2000–U+200A: en space, em space, etc. */
    if (cp == 0x2028) return true;  /* U+2028 LINE SEPARATOR */
    if (cp == 0x2029) return true;  /* U+2029 PARAGRAPH SEPARATOR */
    if (cp == 0x3000) return true;  /* U+3000 IDEOGRAPHIC SPACE */
    return false;
}

typedef bool (*PunctFilter)(uint32_t codepoint);

static bool luganda_punct_filter(uint32_t cp) {
    return (cp == '-' || cp == '\'');
}

static inline bool is_unicode_punct_ex(uint32_t cp, PunctFilter allow_fn) {
    if (allow_fn && allow_fn(cp)) return false;
    if (cp < 128) {
        return ispunct(cp);
    }
    /* Simple ranges for common Unicode punctuation */
    return (cp >= 0x2000 && cp <= 0x206F) || /* General Punctuation */
           (cp >= 0x3000 && cp <= 0x303F) || /* CJK Symbols and Punctuation */
           (cp >= 0xFF00 && cp <= 0xFFEF);   /* Halfwidth and Fullwidth Forms */
}

static void stream_flush(StreamContext *ctx) {
    if (ctx->len == 0) return;
    
    ctx->buffer[ctx->len] = '\0';
    
    /* Intern surface form to canonical lexeme ID */
    uint32_t lexeme_id = le_intern_lexeme(ctx->gemini, ctx->buffer);
    if (lexeme_id == UINT32_MAX) { ctx->len = 0; return; } /* Interning failed, skip */
    
    /* Ingest canonical lexeme (not token occurrence ID) into engine */
    #pragma GCC diagnostic push
    #pragma GCC diagnostic ignored "-Wdeprecated-declarations"
    ctx->prev_node = le_process_token(ctx->gemini, lexeme_id, ctx->prev_node);
    #pragma GCC diagnostic pop
    
    ctx->len = 0;
}

/**
 * Push a raw byte into the stream.
 * Handles UTF-8 decoding, case folding, and token boundary detection.
 */
static void stream_push_byte(StreamContext *ctx, unsigned char byte) {
    uint32_t cp;

    if (!utf8_decode(ctx, byte, &cp))
        return;

    /* Case normalization (Basic Latin) */
    if (cp < 128 && isalpha(cp))
        cp = (uint32_t)tolower(cp);

    if (is_unicode_space(cp)) {
        stream_flush(ctx);
        return;
    }

    if (is_unicode_punct_ex(cp, luganda_punct_filter)) {
        stream_flush(ctx);

        /* Handle punctuation as a token if ASCII */
        if (cp < 128) {
            ctx->buffer[0] = (char)cp;
            ctx->buffer[1] = '\0';
            /* Intern punctuation to canonical lexeme ID */
            uint32_t lexeme_id = le_intern_lexeme(ctx->gemini, ctx->buffer);
            if (lexeme_id != UINT32_MAX) {
              #pragma GCC diagnostic push
              #pragma GCC diagnostic ignored "-Wdeprecated-declarations"
              ctx->prev_node = le_process_token(ctx->gemini, lexeme_id, ctx->prev_node);
              #pragma GCC diagnostic pop
            }
            /* Reset chain on sentence terminators */
            if (cp == '.' || cp == '?' || cp == '!') {
                ctx->prev_node = LE_INVALID;
            }
        } else {
            /* Drop non-ASCII punct for now (or improve logic later) */
            /* WARNING: Non-ASCII structural punctuation (e.g. CJK 、。「」) is silently dropped. */
        }
        
        /* No need to clear ctx->len; stream_flush guarantees it */
        return;
    }

    /* Accumulate word characters */
    int bytes_needed = (cp < 0x80) ? 1 : (cp < 0x800) ? 2 : (cp < 0x10000) ? 3 : 4;
    
    if (ctx->len + bytes_needed >= MAX_WORD_LEN) {
        stream_flush(ctx);
    }

    /* Re-encode normalized codepoint to UTF-8 buffer */
    if (cp < 0x80) {
        ctx->buffer[ctx->len++] = (char)cp;
    } else if (cp < 0x800) {
        ctx->buffer[ctx->len++] = (char)(0xC0 | (cp >> 6));
        ctx->buffer[ctx->len++] = (char)(0x80 | (cp & 0x3F));
    } else if (cp < 0x10000) {
        ctx->buffer[ctx->len++] = (char)(0xE0 | (cp >> 12));
        ctx->buffer[ctx->len++] = (char)(0x80 | ((cp >> 6) & 0x3F));
        ctx->buffer[ctx->len++] = (char)(0x80 | (cp & 0x3F));
    } else {
        ctx->buffer[ctx->len++] = (char)(0xF0 | (cp >> 18));
        ctx->buffer[ctx->len++] = (char)(0x80 | ((cp >> 12) & 0x3F));
        ctx->buffer[ctx->len++] = (char)(0x80 | ((cp >> 6) & 0x3F));
        ctx->buffer[ctx->len++] = (char)(0x80 | (cp & 0x3F));
    }
}



/* ============================= PERSISTENCE ================================= */

/**
 * Save tokenizer to a binary file.
 * CRITICAL FIX (Issue #16): Saves individual fields, excluding the live pthread_mutex_t.
 * The mutex field cannot be serialized portably (may contain thread IDs, kernel handles).
 * CRITICAL FIX (Issue #11): Acquire lock to prevent concurrent tokenizer_get_id calls from
 * modifying state during serialization.
 */
bool tokenizer_save(Tokenizer* t, const char* path) {
    if (!t || !path) return false;

    /* CRITICAL FIX: Acquire lock to prevent concurrent modifications during save */
    pthread_mutex_lock(&t->lock);

    /* CRITICAL FIX: Write fields directly to file to eliminate 27MB intermediate buffer
     * The intermediate copy is unnecessary since fwrite on a buffered FILE* is already
     * internally buffered. This eliminates the large allocation and makes allocation
     * failure impossible for this operation. */
    FILE* f = fopen(path, "wb");
    if (!f) {
        pthread_mutex_unlock(&t->lock);
        perror("tokenizer_save");
        return false;
    }

    size_t written = 0;
    written += fwrite(&t->magic, sizeof(t->magic), 1, f);
    written += fwrite(&t->version, sizeof(t->version), 1, f);
    written += fwrite(t->nodes, sizeof(t->nodes), 1, f);
    written += fwrite(&t->node_count, sizeof(t->node_count), 1, f);
    written += fwrite(&t->token_sequence, sizeof(t->token_sequence), 1, f);
    written += fwrite(t->string_pool, sizeof(t->string_pool), 1, f);
    written += fwrite(&t->pool_used, sizeof(t->pool_used), 1, f);
    written += fwrite(t->id_to_offset, sizeof(t->id_to_offset), 1, f);

    pthread_mutex_unlock(&t->lock);

    /* HIGH FIX (Issue #17): Check fclose return value for flush errors */
    int close_ret = fclose(f);
    if (close_ret != 0) {
        fprintf(stderr, "ERROR: fclose failed during tokenizer_save\n");
        return false;
    }

    return (written == 9); /* 9 items written */
}

/**
 * Load tokenizer from a binary file.
 * Validates magic, version, and field ranges (Issue #18).
 * Caller must tokenizer_destroy() the result.
 */
Tokenizer* tokenizer_load(const char* path) {
    if (!path) return NULL;

    FILE* f = fopen(path, "rb");
    if (!f) { perror("tokenizer_load"); return NULL; }

    /* FIX M-6: malloc leaves string_pool[pool_used..MAX_STRING_POOL-1]
     * uninitialized. Any future traversal past pool_used (e.g. strlen on a
     * string stored near the boundary) could read garbage. calloc costs ~6 MB
     * once at load time and eliminates the risk. */
    Tokenizer* t = (Tokenizer*)calloc(1, sizeof(Tokenizer));
    if (!t) {
        fclose(f);
        return NULL;
    }

    /* Read fields individually, matching tokenizer_save */
    size_t read = 0;
    read += fread(&t->magic, sizeof(t->magic), 1, f);
    read += fread(&t->version, sizeof(t->version), 1, f);
    read += fread(t->nodes, sizeof(t->nodes), 1, f);
    read += fread(&t->node_count, sizeof(t->node_count), 1, f);
    read += fread(&t->token_sequence, sizeof(t->token_sequence), 1, f);
    read += fread(t->string_pool, sizeof(t->string_pool), 1, f);
    read += fread(&t->pool_used, sizeof(t->pool_used), 1, f);
    read += fread(t->id_to_offset, sizeof(t->id_to_offset), 1, f);
    fclose(f);
    
    /* Reset transient state after load */
    t->word_len = 0;
    memset(t->word_buf, 0, sizeof(t->word_buf));

    if (read != 8) {
        fprintf(stderr, "ERROR: Tokenizer file truncated (read %zu fields, expected 8)\n", read);
        free(t);
        return NULL;
    }

    if (t->magic != 0x54535420) {
        fprintf(stderr, "ERROR: Invalid tokenizer magic (expected 0x54535420, got 0x%08X)\n",
                t->magic);
        free(t);
        return NULL;
    }

    if (t->version != 1) {
        fprintf(stderr, "ERROR: Tokenizer version mismatch (expected 1, got %u)\n",
                t->version);
        free(t);
        return NULL;
    }

    /* HIGH FIX (Issue #18): Validate loaded field ranges to prevent out-of-bounds access */
    if (t->node_count > MAX_TRIE_NODES || t->token_sequence >= MAX_VOCAB ||
        t->pool_used > MAX_STRING_POOL) {
        fprintf(stderr, "ERROR: Tokenizer file contains out-of-range values\n");
        fprintf(stderr, "  node_count=%u (max %u)\n", t->node_count, MAX_TRIE_NODES);
        fprintf(stderr, "  token_sequence=%u (max %u)\n", t->token_sequence, MAX_VOCAB - 1);
        fprintf(stderr, "  pool_used=%u (max %u)\n", t->pool_used, MAX_STRING_POOL);
        free(t);
        return NULL;
    }
    
    /* Validate id_to_offset ranges */
    for (uint32_t i = 1; i <= t->token_sequence; i++) {
        if (t->id_to_offset[i] >= t->pool_used) {
            fprintf(stderr, "ERROR: id_to_offset[%u] = %u exceeds pool_used %u\n",
                    i, t->id_to_offset[i], t->pool_used);
            free(t);
            return NULL;
        }
    }

    /* Re-initialize mutex after load (Issue #16: mutex was not serialized) */
    if (pthread_mutex_init(&t->lock, NULL) != 0) {
        fprintf(stderr, "ERROR: Failed to initialize tokenizer mutex after load\n");
        free(t);
        return NULL;
    }

    return t;
}

uint32_t tokenizer_vocab_size(Tokenizer* t) {
    if (!t) return 0;
    return t->token_sequence;
}


/* ============================= ENGINE INTEGRATION ========================== */

/**
 * High-level: feed raw text through UTF-8 stream + engine in one call.
 * This handles multi-byte decoding, normalization, and graph traversal.
 * Returns the final NodeID (for chaining with the next call).
 */
NodeID tokenizer_process_text(Tokenizer* t, EngineContext* engine,
                              const char* text, NodeID prev_node) {
    if (!t || !engine || !text) return prev_node;

    StreamContext ctx;
    memset(&ctx, 0, sizeof(StreamContext));
    ctx.tokenizer = t;
    ctx.gemini = engine;
    ctx.prev_node = prev_node;
    
    for (const char *p = text; *p; p++) {
        stream_push_byte(&ctx, (unsigned char)*p);
    }
    
    /* Ensure trailing content is flushed */
    stream_flush(&ctx);
    
    return ctx.prev_node;
}