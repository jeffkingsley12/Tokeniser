/*
 * gemini_enhanced.c — Enhanced Gemini engine orchestration layer
 */

#include "gemini_enhanced.h"
#include "gemini_phon.h"
#include "gemini_morph.h"
#include "gemini_ngram_prior.h"
#include "gemini_phrase_seed.h"
#include "../validation/gemini_attest.h"
#include "gemini_semantic.h"
#include "../validation/gemini_eval.h"
#include "libgemini.h"
#include "../include/gemini_internal.h"

#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <stdatomic.h>

struct GeminiEnhanced {
    EngineContext  *ctx;
    Tokenizer      *tok;
    EnhancedConfig  cfg;
    MorphVariantTable *variant_table;
    GeminiAttestDB    *attest_db;
    EmbeddingModel    *embeddings;

    uint64_t tokens_processed;
    uint32_t epochs_completed;
    uint32_t total_promotions;
    uint32_t total_semantic_merges;
};

EnhancedConfig gemini_enhanced_default_config(void) {
    EnhancedConfig cfg;
    memset(&cfg, 0, sizeof(cfg));

    cfg.morph_enabled             = true;
    cfg.morph_stem_confidence_min = 0.70f;
    cfg.phon_enabled              = true;
    cfg.ngram_prior_enabled       = false;
    cfg.ngram_prior_path          = NULL;
    cfg.ngram_prior_scale         = NGRAM_PRIOR_DEFAULT_SCALE;
    cfg.phrase_seed_enabled       = true;
    cfg.phrase_seed_base_weight   = 5.0f;
    cfg.attest_enabled            = false;
    cfg.attest_wordlist_path      = NULL;
    cfg.attest_min_ratio          = 0.60f;
    cfg.semantic_enabled          = false;
    cfg.embedding_path            = NULL;
    cfg.semantic_cosine_threshold = SEMANTIC_DEFAULT_COSINE_THRESHOLD;
    cfg.semantic_turb_threshold   = SEMANTIC_DEFAULT_TURB_THRESHOLD;
    cfg.eval_enabled              = false;
    cfg.eval_test_csv             = NULL;

    return cfg;
}

static Tokenizer *create_tokenizer_for_engine(void);

GeminiEnhanced *gemini_enhanced_create(const EnhancedConfig *cfg) {
    if (!cfg) return NULL;

    GeminiEnhanced *ge = calloc(1, sizeof(GeminiEnhanced));
    if (!ge) return NULL;

    ge->cfg = *cfg;

    /* CRITICAL FIX: Initialize all string fields to NULL to prevent freeing caller's pointers
     * on partial SAFE_STRDUP failure. The raw struct copy above makes all pointer fields
     * point into the caller's memory. */
    ge->cfg.ngram_prior_path = NULL;
    ge->cfg.attest_wordlist_path = NULL;
    ge->cfg.embedding_path = NULL;
    ge->cfg.eval_test_csv = NULL;

    /* Safe allocation block to prevent dangling pointers on OOM */
#define SAFE_ASSIGN_STR(dest, src) \
    do { \
        if (src) { \
            dest = strdup(src); \
            if (!dest) goto cleanup; \
        } else { \
            dest = NULL; \
        } \
    } while(0)

    SAFE_ASSIGN_STR(ge->cfg.ngram_prior_path, cfg->ngram_prior_path);
    SAFE_ASSIGN_STR(ge->cfg.attest_wordlist_path, cfg->attest_wordlist_path);
    SAFE_ASSIGN_STR(ge->cfg.embedding_path, cfg->embedding_path);
    SAFE_ASSIGN_STR(ge->cfg.eval_test_csv, cfg->eval_test_csv);

#undef SAFE_STRDUP

    ge->ctx = le_init();
    if (!ge->ctx) {
        fprintf(stderr, "gemini_enhanced: le_init() failed\n");
        goto cleanup;
    }

    ge->tok = create_tokenizer_for_engine();
    if (!ge->tok) {
        fprintf(stderr, "gemini_enhanced: failed to create tokenizer\n");
        goto cleanup;
    }

    if (cfg->morph_enabled) {
        ge->variant_table = morph_variant_table_create();
        if (!ge->variant_table) {
            fprintf(stderr, "gemini_enhanced: WARNING — variant table alloc failed, morph disabled\n");
        }
    }

    if (cfg->ngram_prior_enabled && cfg->ngram_prior_path) {
        NgramPriorResult npr;
        int rc = gemini_load_ngram_prior(ge->ctx, ge->tok, cfg->ngram_prior_path, cfg->ngram_prior_scale, &npr);
        if (rc == 0) {
            printf("gemini_enhanced: N-gram prior loaded — ");
            gemini_ngram_prior_print(&npr);
        } else {
            fprintf(stderr, "gemini_enhanced: WARNING — failed to load N-gram prior from '%s' (rc=%d)\n", cfg->ngram_prior_path, rc);
        }
    }

    if (cfg->phrase_seed_enabled) {
        PhraseSeedResult psr;
        int seeded = gemini_phrase_seed(ge->ctx, ge->tok, cfg->phrase_seed_base_weight, 5, &psr);
        if (seeded > 0) {
            printf("gemini_enhanced: ");
            gemini_phrase_seed_print(&psr);
        }
    }

    if (cfg->attest_enabled && cfg->attest_wordlist_path) {
        ge->attest_db = gemini_attest_db_create_from_file(cfg->attest_wordlist_path);
        if (!ge->attest_db) {
            fprintf(stderr, "gemini_enhanced: WARNING — attestation DB load failed from '%s'\n", cfg->attest_wordlist_path);
        } else {
            printf("gemini_enhanced: attestation DB loaded (%zu words)\n", gemini_attest_db_size(ge->attest_db));
        }
    }

#ifdef GEMINI_FEAT_SEMANTIC
    if (cfg->semantic_enabled && cfg->embedding_path) {
        ge->embeddings = embedding_model_load(cfg->embedding_path);
        if (!ge->embeddings) {
            fprintf(stderr, "gemini_enhanced: WARNING — embedding model load failed from '%s'\n", cfg->embedding_path);
        } else {
            int emb_count, emb_dim;
            embedding_get_stats(ge->embeddings, &emb_count, &emb_dim);
            printf("gemini_enhanced: embeddings loaded (%d words, dim=%d)\n", emb_count, emb_dim);
        }
    }
#endif

    printf("gemini_enhanced: engine ready.\n");
    return ge;

cleanup:
    free((char *)ge->cfg.ngram_prior_path);
    free((char *)ge->cfg.attest_wordlist_path);
    free((char *)ge->cfg.embedding_path);
    free((char *)ge->cfg.eval_test_csv);
    if (ge->tok) tokenizer_destroy(ge->tok);
    if (ge->ctx) le_destroy(ge->ctx);
    free(ge);
    return NULL;
}

void gemini_enhanced_destroy(GeminiEnhanced *ge) {
    if (!ge) return;
#ifdef GEMINI_FEAT_SEMANTIC
    if (ge->embeddings) embedding_free(ge->embeddings);
#endif
    if (ge->attest_db) gemini_attest_db_destroy(ge->attest_db);
    if (ge->variant_table) morph_variant_table_destroy(ge->variant_table);
    if (ge->tok) tokenizer_destroy(ge->tok);
    if (ge->ctx) le_destroy(ge->ctx);

    free((char *)ge->cfg.ngram_prior_path);
    free((char *)ge->cfg.attest_wordlist_path);
    free((char *)ge->cfg.embedding_path);
    free((char *)ge->cfg.eval_test_csv);
    free(ge);
}

EngineContext *gemini_enhanced_ctx(GeminiEnhanced *ge) { return ge ? ge->ctx : NULL; }
Tokenizer *gemini_enhanced_tokenizer(GeminiEnhanced *ge) { return ge ? ge->tok : NULL; }
GeminiAttestDB *gemini_enhanced_attest_db(GeminiEnhanced *ge) { return ge ? ge->attest_db : NULL; }

NodeID gemini_enhanced_process_word(GeminiEnhanced *ge, const char *word, NodeID prev_node) {
    if (!ge || !word || word[0] == '\0') return LE_INVALID;

    /* Prevent silent truncation of long words */
    if (strlen(word) >= 255) return LE_INVALID;

    char buf[256];
    strncpy(buf, word, sizeof(buf) - 1);
    buf[sizeof(buf) - 1] = '\0';

    if (ge->cfg.phon_enabled) {
        apply_phonology_rules(buf, sizeof(buf));
    }

    char canonical[MORPH_MAX_STEM_LEN];
    bool used_stem = false;

    if (ge->cfg.morph_enabled && ge->variant_table) {
        used_stem = morph_get_canonical_stem(buf, ge->cfg.morph_stem_confidence_min, canonical, sizeof(canonical));
    }

    const char *lookup_word = used_stem ? canonical : buf;
    TokenID id = tokenizer_get_id(ge->tok, lookup_word, true);
    NodeID node = le_get_or_create_node(ge->ctx, id);

    if (used_stem && ge->variant_table && strcmp(buf, canonical) != 0) {
        morph_variant_table_register(ge->variant_table, buf, canonical, id);
    }

    if (prev_node != LE_INVALID && node != LE_INVALID) {
        le_add_edge(ge->ctx, prev_node, node);
    }

    ge->tokens_processed++;
    return node;
}

NodeID gemini_enhanced_process_text(GeminiEnhanced *ge, const char *text, NodeID prev_node) {
    if (!ge || !text) return prev_node;

    size_t text_len = strlen(text);
    char *buf = malloc(text_len + 1);
    if (!buf) {
        fprintf(stderr, "gemini_enhanced: alloc failed for text processing\n");
        return prev_node;
    }
    strcpy(buf, text);

    if (ge->cfg.phon_enabled) {
        apply_phonology_rules_sentence(buf, text_len + 1);
    }

    char *p = buf;
    NodeID last = prev_node;

    while (*p) {
        while (*p && isspace((unsigned char)*p)) p++;
        if (!*p) break;

        char *start = p;
        while (*p && !isspace((unsigned char)*p)) p++;

        char saved = *p;
        *p = '\0';

        /* Strict punctuation stripping */
        char *token_start = start;
        char *token_end = p - 1;
        while (token_start <= token_end && ispunct((unsigned char)*token_start) && *token_start != '\'') token_start++;
        while (token_end >= token_start && ispunct((unsigned char)*token_end)) *token_end-- = '\0';

        bool has_alpha = false;
        for (const char *c = token_start; *c; c++) {
            if (isalpha((unsigned char)*c) || *c == '\'') { has_alpha = true; break; }
        }

        if (has_alpha) {
            last = gemini_enhanced_process_word(ge, token_start, last);
        }

        *p = saved;
        if (*p) p++;
    }

    free(buf);
    return last;
}

uint32_t gemini_enhanced_epoch(GeminiEnhanced *ge) {
    if (!ge) return 0;

    le_begin_epoch(ge->ctx);

#ifdef GEMINI_FEAT_SEMANTIC
    if (ge->cfg.semantic_enabled && ge->embeddings) {
        SemanticMergeResult smr;
        int merges = gemini_semantic_merge_pass(ge->ctx, ge->tok, ge->embeddings, ge->cfg.semantic_cosine_threshold, ge->cfg.semantic_turb_threshold, &smr);
        if (merges > 0) {
            ge->total_semantic_merges += (uint32_t)merges;
        }
    }
#endif

    uint32_t promoted = 0;
    if (ge->cfg.attest_enabled && ge->attest_db) {
        AttestGateResult agr;
        int rc = gemini_attest_promote(ge->ctx, ge->tok, ge->attest_db, ge->cfg.attest_min_ratio, &agr);
        if (rc >= 0) promoted = (uint32_t)rc;
    } else {
        promoted = le_promote_eligible(ge->ctx);
    }

    ge->total_promotions += promoted;
    ge->epochs_completed++;
    return promoted;
}

bool gemini_enhanced_save(GeminiEnhanced *ge, const char *path) {
    if (!ge || !path) return false;
    bool ok = le_save_mmap(ge->ctx, path);
    if (!ok) {
        fprintf(stderr, "gemini_enhanced_save: le_save_mmap failed\n");
        return false;
    }
    printf("gemini_enhanced: saved to '%s'\n", path);
    return true;
}

GeminiEnhanced *gemini_enhanced_load(const char *path, const EnhancedConfig *cfg) {
    if (!path || !cfg) return NULL;

    GeminiEnhanced *ge = gemini_enhanced_create(cfg);
    if (!ge) return NULL;

    uint32_t fresh_vocab = ge->tok ? tokenizer_vocab_size(ge->tok) : 0;

    le_destroy(ge->ctx);
    ge->ctx = le_load_mmap(path, false);

    if (!ge->ctx) {
        fprintf(stderr, "gemini_enhanced_load: le_load_mmap failed for '%s'\n", path);
        gemini_enhanced_destroy(ge);
        return NULL;
    }

    /* FATAL ERROR: Strict abort on tokenizer mismatch
     * NOTE: This guard is currently disabled because create_tokenizer_for_engine()
     * returns an empty tokenizer (fresh_vocab = 0). A proper fix requires either:
     * 1) Loading the tokenizer from the saved snapshot, or
     * 2) Persisting the tokenizer path in the snapshot and reloading it.
     * For now, this check is bypassed - loading mismatched snapshots may corrupt
     * the graph if token IDs don't correspond. */
    if (fresh_vocab > 0) {
        uint32_t loaded_vocab = ge->ctx->token_count;
        if (loaded_vocab != fresh_vocab) {
            fprintf(stderr,
                    "gemini_enhanced_load: FATAL — tokenizer vocab size mismatch "
                    "(tokenizer=%u, loaded_engine=%u). Token IDs are inconsistent. "
                    "Aborting load to prevent graph corruption.\n",
                    fresh_vocab, loaded_vocab);
            ge->ctx = NULL; // Prevent double free
            gemini_enhanced_destroy(ge);
            return NULL;
        }
    } else {
        fprintf(stderr,
                "WARNING: gemini_enhanced_load - vocabulary mismatch guard disabled "
                "(fresh_vocab=0). Loading snapshot without token ID verification may "
                "corrupt the graph if the tokenizer vocabulary has changed.\n");
    }

    printf("gemini_enhanced: loaded from '%s'\n", path);
    return ge;
}

int gemini_enhanced_eval(GeminiEnhanced *ge, const char *csv_path, GeminiEvalReport *report) {
    if (!ge || !csv_path || !report) return -1;

    GeminiTestSuite *suite = gemini_eval_suite_create(1000);
    if (!suite) return -1;

    int loaded = gemini_eval_suite_load_csv(suite, csv_path);
    if (loaded <= 0) {
        fprintf(stderr, "gemini_enhanced_eval: no test cases loaded from '%s'\n", csv_path);
        gemini_eval_suite_free(suite);
        return -1;
    }

    printf("gemini_enhanced_eval: %d test cases loaded\n", loaded);
    int rc = gemini_eval_run(ge->ctx, ge->tok, suite, 5, 3, report);
    gemini_eval_suite_free(suite);
    return rc;
}

static Tokenizer *create_tokenizer_for_engine(void) {
    return tokenizer_init();
}