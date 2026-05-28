/*
 * gemini_ngram_prior.c — N-gram prior injection into Gemini edge weights
 */

#include "gemini_ngram_prior.h"
#include "libgemini.h"
#include "gemini_internal.h"

#include <math.h>
#include <stdbool.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ── Endian-Safe Binary Reads ───────────────────────────────────────────────── */

static uint16_t read_u16_le(FILE *fp) {
    uint8_t b[2];
    if (fread(b, 1, 2, fp) != 2) return 0;
    return (uint16_t)(b[0] | (b[1] << 8));
}

static uint32_t read_u32_le(FILE *fp) {
    uint8_t b[4];
    if (fread(b, 1, 4, fp) != 4) return 0;
    return (uint32_t)(b[0] | (b[1] << 8) | (b[2] << 16) | (b[3] << 24));
}

/* ── Internal n-gram record ─────────────────────────────────────────────────── */

typedef struct {
    char     words[NGRAM_PRIOR_MAX_ORDER][NGRAM_PRIOR_MAX_WORD_LEN];
    uint8_t  order;
    uint32_t frequency;
} NgramRecord;

/* ── Dynamic Unigram Table ──────────────────────────────────────────────────── */

#define INITIAL_UNIGRAM_CAPACITY 65537

typedef struct UnigramEntry {
    char     word[NGRAM_PRIOR_MAX_WORD_LEN];
    uint32_t freq;
} UnigramEntry;

typedef struct {
    UnigramEntry *slots;
    uint32_t      capacity;
    uint32_t      used;
    uint32_t      total_tokens;
} UnigramTable;

static uint32_t ug_hash(const char *w) {
    uint32_t h = 5381;
    int c;
    while ((c = (unsigned char)*w++)) h = ((h << 5) + h) ^ (uint32_t)c;
    return h;
}

static void ug_add_internal(UnigramTable *ug, const char *word, uint32_t freq, bool is_resize);

static void ug_resize(UnigramTable *ug) {
    uint32_t old_cap = ug->capacity;
    UnigramEntry *old_slots = ug->slots;

    ug->capacity = old_cap * 2 + 1; 
    ug->slots = calloc(ug->capacity, sizeof(UnigramEntry));
    ug->used = 0; 
    uint32_t saved_total = ug->total_tokens;

    for (uint32_t i = 0; i < old_cap; i++) {
        if (old_slots[i].word[0] != '\0') {
            ug_add_internal(ug, old_slots[i].word, old_slots[i].freq, true);
        }
    }
    
    ug->total_tokens = saved_total; 
    free(old_slots);
}

static void ug_add_internal(UnigramTable *ug, const char *word, uint32_t freq, bool is_resize) {
    if (ug->used * 10 >= ug->capacity * 7) {
        ug_resize(ug);
    }

    uint32_t h = ug_hash(word) % ug->capacity;
    for (uint32_t i = 0; i < ug->capacity; i++) {
        uint32_t idx = (h + i) % ug->capacity;
        if (ug->slots[idx].word[0] == '\0') {
            snprintf(ug->slots[idx].word, NGRAM_PRIOR_MAX_WORD_LEN, "%s", word);
            ug->slots[idx].freq = freq;
            ug->used++;
            if (!is_resize) ug->total_tokens += freq;
            return;
        }
        if (strncmp(ug->slots[idx].word, word, NGRAM_PRIOR_MAX_WORD_LEN) == 0) {
            ug->slots[idx].freq += freq;
            if (!is_resize) ug->total_tokens += freq;
            return;
        }
    }
}

static void ug_add(UnigramTable *ug, const char *word, uint32_t freq) {
    ug_add_internal(ug, word, freq, false);
}

static uint32_t ug_get(const UnigramTable *ug, const char *word) {
    if (!ug || ug->capacity == 0) return 0;
    uint32_t h = ug_hash(word) % ug->capacity;
    for (uint32_t i = 0; i < ug->capacity; i++) {
        uint32_t idx = (h + i) % ug->capacity;
        if (ug->slots[idx].word[0] == '\0') return 0;
        if (strncmp(ug->slots[idx].word, word, NGRAM_PRIOR_MAX_WORD_LEN) == 0)
            return ug->slots[idx].freq;
    }
    return 0;
}

/* ── File reading ───────────────────────────────────────────────────────────── */

static int read_ngram_record(FILE *fp, NgramRecord *rec) {
    uint8_t size;
    if (fread(&size, 1, 1, fp) != 1) return 0;   
    
    rec->frequency = read_u32_le(fp);
    if (size == 0 || size > NGRAM_PRIOR_MAX_ORDER) return -1;
    rec->order = size;

    for (int i = 0; i < (int)size; i++) {
        uint16_t len = read_u16_le(fp);
        if (len == 0 || len >= NGRAM_PRIOR_MAX_WORD_LEN) return -1;
        if (fread(rec->words[i], 1, len, fp) != len) return -1;
        rec->words[i][len] = '\0';
    }
    return 1;
}

/* ── Edge weight injection ──────────────────────────────────────────────────── */

static float compute_edge_weight(uint32_t bigram_freq, uint32_t unigram_freq, uint32_t vocab_size, float scale) {
    const float alpha = 0.01f;
    float v    = (float)vocab_size;
    float cond = ((float)bigram_freq + alpha) / ((float)unigram_freq + alpha * v);
    float log2p = log2f(cond);
    float w     = scale * (1.0f + log2p);   
    if (w < 0.1f) w = 0.1f;                 
    return w;
}

/* ── gemini_load_ngram_prior ────────────────────────────────────────────────── */

int gemini_load_ngram_prior(EngineContext *ctx, Tokenizer *tok, const char *path, float scale, NgramPriorResult *result) {
    if (!ctx || !tok || !path) return -1;
    if (scale <= 0.0f) scale = NGRAM_PRIOR_DEFAULT_SCALE;

    if (result) memset(result, 0, sizeof(*result));

    FILE *fp = fopen(path, "rb");
    if (!fp) {
        fprintf(stderr, "gemini_ngram_prior: cannot open '%s'\n", path);
        return -1;
    }

    uint32_t magic   = read_u32_le(fp);
    uint32_t version = read_u32_le(fp);
    uint32_t count   = read_u32_le(fp);
    (void)version;
    (void)count;

    if (magic != NGRAM_PRIOR_FILE_MAGIC) {
        fprintf(stderr, "gemini_ngram_prior: bad header in '%s'\n", path);
        fclose(fp);
        return -2;
    }

    UnigramTable *ug = calloc(1, sizeof(UnigramTable));
    if (!ug) { fclose(fp); return -1; }
    
    ug->capacity = INITIAL_UNIGRAM_CAPACITY;
    ug->slots = calloc(ug->capacity, sizeof(UnigramEntry));

    NgramRecord rec;
    long data_start = ftell(fp);
    int  rstat;

    while ((rstat = read_ngram_record(fp, &rec)) == 1) {
        if (rec.order == 1) {
            ug_add(ug, rec.words[0], rec.frequency);
        }
    }
    uint32_t vocab_size = ug->used > 0 ? ug->used : 1;

    fseek(fp, data_start, SEEK_SET);
    int edges_created = 0, edges_boosted = 0;
    int bigrams_loaded = 0, trigrams_loaded = 0;
    int nodes_created = 0;

    while ((rstat = read_ngram_record(fp, &rec)) == 1) {
        if (rec.order < 2) continue;   

        if (rec.order == 2) {
            bigrams_loaded++;
            const char *w1 = rec.words[0];
            const char *w2 = rec.words[1];

            uint32_t freq_w1 = ug_get(ug, w1);
            if (freq_w1 == 0) freq_w1 = 1;

            TokenID id1 = tokenizer_get_id(tok, w1, true);
            TokenID id2 = tokenizer_get_id(tok, w2, true);

            uint32_t nc_before = atomic_load(&ctx->node_count);
            NodeID n1 = le_get_or_create_node(ctx, id1);
            NodeID n2 = le_get_or_create_node(ctx, id2);
            uint32_t nc_after  = atomic_load(&ctx->node_count);
            nodes_created += (int)(nc_after - nc_before);

            if (n1 == LE_INVALID || n2 == LE_INVALID) continue;

            float w = compute_edge_weight(rec.frequency, freq_w1, vocab_size, scale);
            le_add_edge_bulk(ctx, n1, n2, w);
            edges_created++;
        }
        else if (rec.order == 3) {
            trigrams_loaded++;
            const char *w1 = rec.words[0];
            const char *w2 = rec.words[1];
            const char *w3 = rec.words[2];

            uint32_t freq_w12 = ug_get(ug, w2);
            if (freq_w12 == 0) freq_w12 = 1;

            TokenID id2 = tokenizer_get_id(tok, w2, true);
            TokenID id3 = tokenizer_get_id(tok, w3, true);

            uint32_t nc_before = atomic_load(&ctx->node_count);
            NodeID  n2  = le_get_or_create_node(ctx, id2);
            NodeID  n3  = le_get_or_create_node(ctx, id3);
            uint32_t nc_after  = atomic_load(&ctx->node_count);
            nodes_created += (int)(nc_after - nc_before);

            if (n2 == LE_INVALID || n3 == LE_INVALID) continue;

            float w = compute_edge_weight(rec.frequency, freq_w12, vocab_size, scale * 0.5f);
            le_add_edge_bulk(ctx, n2, n3, w);
            edges_boosted++;

            (void)w1;   
        }
    }

    fclose(fp);
    free(ug->slots);
    free(ug);

    if (result) {
        result->bigrams_loaded  = bigrams_loaded;
        result->trigrams_loaded = trigrams_loaded;
        result->edges_created   = edges_created;
        result->edges_boosted   = edges_boosted;
        result->nodes_created   = nodes_created;
        result->scale_used      = scale;
    }

    return (rstat == -1) ? -1 : 0;
}

void gemini_ngram_prior_print(const NgramPriorResult *r) {
    if (!r) { printf("NgramPriorResult: (null)\n"); return; }
    printf("N-gram Prior Injection Results:\n");
    printf("  Bigrams loaded   : %d\n", r->bigrams_loaded);
    printf("  Trigrams loaded  : %d\n", r->trigrams_loaded);
    printf("  Edges created    : %d\n", r->edges_created);
    printf("  Edges boosted    : %d\n", r->edges_boosted);
    printf("  Nodes created    : %d\n", r->nodes_created);
    printf("  Scale factor     : %.2f\n", r->scale_used);
}