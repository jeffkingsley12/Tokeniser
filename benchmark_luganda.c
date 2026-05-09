#define _POSIX_C_SOURCE 200809L
#include "tokenizer.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <stdint.h>
#include <stdbool.h>
#include <math.h>

#if defined(__x86_64__) || defined(__i386__)
#include <x86intrin.h>
static inline uint64_t get_cycles(void) {
    return __rdtsc();
}
#else
static inline uint64_t get_cycles(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;
}
#endif

#define MAX_TOKENS_PER_CASE 1024
#define WARMUP_ITERATIONS 10
#define MEASURE_ITERATIONS 100
#define MAX_BENCHMARK_CASES 50

/* Benchmark case flags */
#define FLAG_EMOJI       (1 << 0)  /* Contains emoji */
#define FLAG_MIXED_LANG  (1 << 1)  /* Mixed English/Luganda */
#define FLAG_CRLF        (1 << 2)  /* Windows line endings */
#define FLAG_UNICODE_WS  (1 << 3)  /* Unicode whitespace */
#define FLAG_LOANWORDS   (1 << 4)  /* English loanwords */
#define FLAG_MISSPELL    (1 << 5)  /* Common misspellings */
#define FLAG_LONG_INPUT  (1 << 6)  /* Long message (100+ syllables) */
#define FLAG_EDGE_SPACE  (1 << 7)  /* Leading/trailing spaces */

typedef struct {
    const char *name;
    const char *text;
    uint32_t expected_tokens;
    uint64_t flags;
    const char *description;
} BenchmarkCase;

typedef struct {
    uint64_t total_bytes;
    uint64_t total_tokens;
    uint64_t fast_path_hits;
    uint64_t trie_traversals;
    uint64_t utf8_skips;
    double tokens_per_ms;
    double avg_token_len;
    double fast_path_ratio;
    uint64_t cycles_total;
    uint64_t cycles_per_token;
} BenchmarkStats;

typedef struct {
    BenchmarkCase cases[MAX_BENCHMARK_CASES];
    uint32_t case_count;
    uint32_t total_expected_tokens;
    size_t total_bytes;
} BenchmarkSuite;

/* UTF-8 byte sequences for special characters */
#define SPACE_CHAR "\xE2\x96\x81"  /* ▁ */
#define NBSP_CHAR "\xC2\xA0"       /* non-breaking space */
#define PRAYER_HANDS "\xF0\x9F\x99\x8F"  /* 🙏 */
#define LAUGHING "\xF0\x9F\x98\x82"       /* 😂 */
#define THUMBS_UP "\xF0\x9F\x91\x8D"      /* 👍 */
#define FAMILY "\xF0\x9F\x91\xA8\xE2\x80\x8D\xF0\x9F\x91\xA9\xE2\x80\x8D\xF0\x9F\x91\xA7\xE2\x80\x8D\xF0\x9F\x91\xA6"  /* 👨‍👩‍👧‍👦 */

/* Realistic WhatsApp-style Luganda test cases */
static const BenchmarkCase luganda_cases[] = {
    {
        .name = "greeting_simple",
        .text = "Ol" SPACE_CHAR "oli" SPACE_CHAR "ot" SPACE_CHAR "yo",
        .expected_tokens = 16,
        .flags = 0,
        .description = "Simple greeting with ▁ spaces"
    },
    {
        .name = "whatsapp_mixed",
        .text = "Weebale" SPACE_CHAR "nnyo" PRAYER_HANDS SPACE_CHAR "ok" SPACE_CHAR "webale",
        .expected_tokens = 23,
        .flags = FLAG_EMOJI | FLAG_MIXED_LANG,
        .description = "WhatsApp message with emoji and mixed language"
    },
    {
        .name = "windows_line_endings",
        .text = "Oli" SPACE_CHAR "ot" SPACE_CHAR "yo\r\nKale" SPACE_CHAR "bulungi",
        .expected_tokens = 20,
        .flags = FLAG_CRLF,
        .description = "Windows CRLF line endings"
    },
    {
        .name = "nonbreaking_space",
        .text = "Ssebo" NBSP_CHAR "weebale",
        .expected_tokens = 8,
        .flags = FLAG_UNICODE_WS,
        .description = "Non-breaking space between words"
    },
    {
        .name = "loanwords_french",
        .text = "Merci" SPACE_CHAR "bekka" SPACE_CHAR "nnyo" SPACE_CHAR "beaucoup",
        .expected_tokens = 22,
        .flags = FLAG_LOANWORDS,
        .description = "French loanwords in Luganda context"
    },
    {
        .name = "misspellings_common",
        .text = "Webale" SPACE_CHAR "nyo" SPACE_CHAR "instead" SPACE_CHAR "of" SPACE_CHAR "weebale",
        .expected_tokens = 28,
        .flags = FLAG_MISSPELL,
        .description = "Common misspellings and variants"
    },
    {
        .name = "edge_leading_space",
        .text = SPACE_CHAR SPACE_CHAR "Muli" SPACE_CHAR "munya" SPACE_CHAR SPACE_CHAR,
        .expected_tokens = 20,
        .flags = FLAG_EDGE_SPACE,
        .description = "Multiple leading and trailing spaces"
    },
    {
        .name = "emoji_sequence",
        .text = "Oli" SPACE_CHAR "okola" LAUGHING LAUGHING SPACE_CHAR "gyendi" THUMBS_UP,
        .expected_tokens = 24,
        .flags = FLAG_EMOJI,
        .description = "Multiple emoji in sequence"
    },
    {
        .name = "english_insertion",
        .text = "Let" SPACE_CHAR "me" SPACE_CHAR "explain" SPACE_CHAR "okukola" SPACE_CHAR " kino",
        .expected_tokens = 27,
        .flags = FLAG_MIXED_LANG,
        .description = "English phrase inserted in Luganda"
    },
    {
        .name = "numbers_and_punct",
        .text = "Nina" SPACE_CHAR "123" SPACE_CHAR "shillings" SPACE_CHAR "only!",
        .expected_tokens = 26,
        .flags = FLAG_MIXED_LANG,
        .description = "Numbers and punctuation"
    },
    {
        .name = "long_message",
        .text = "Njagala" SPACE_CHAR "kukwata" SPACE_CHAR "olukalala" SPACE_CHAR "lwa" SPACE_CHAR "kino" SPACE_CHAR
               "ekyawa" SPACE_CHAR "nnya" SPACE_CHAR "okukola" SPACE_CHAR "ebintu" SPACE_CHAR
               "byona" SPACE_CHAR "bulungi" SPACE_CHAR "nga" SPACE_CHAR "byetaagisa" SPACE_CHAR
               "okutuuka" SPACE_CHAR "ku" SPACE_CHAR "mukka" SPACE_CHAR "gwa" SPACE_CHAR
               "omulimu" SPACE_CHAR "gwaffe",
        .expected_tokens = 93,
        .flags = FLAG_LONG_INPUT,
        .description = "Long message with multiple clauses"
    },
    {
        .name = "repetitive_pattern",
        .text = "Bulungi" SPACE_CHAR "bulungi" SPACE_CHAR "bulungi" SPACE_CHAR "very" SPACE_CHAR "good",
        .expected_tokens = 23,
        .flags = FLAG_MIXED_LANG,
        .description = "Repetitive pattern for compression testing"
    },
    {
        .name = "whatsapp_typing",
        .text = "Am" SPACE_CHAR "still" SPACE_CHAR "typing" SPACE_CHAR "..." SPACE_CHAR "wait",
        .expected_tokens = 29,
        .flags = FLAG_MIXED_LANG,
        .description = "WhatsApp typing indicator style"
    },
    {
        .name = "zero_width_joiner",
        .text = "Family" SPACE_CHAR "emoji" FAMILY SPACE_CHAR "here",
        .expected_tokens = 41,
        .flags = FLAG_EMOJI,
        .description = "Complex emoji with zero-width joiners"
    },
    {
        .name = "mixed_whitespace",
        .text = "Tab" SPACE_CHAR " \t " SPACE_CHAR "spaces\n\r" SPACE_CHAR "mixed",
        .expected_tokens = 25,
        .flags = FLAG_UNICODE_WS,
        .description = "Various whitespace combinations"
    }
};

/* Initialize benchmark suite */
static BenchmarkSuite* create_benchmark_suite(void) {
    BenchmarkSuite *suite = calloc(1, sizeof(BenchmarkSuite));
    if (!suite) return NULL;

    suite->case_count = sizeof(luganda_cases) / sizeof(luganda_cases[0]);
    if (suite->case_count > MAX_BENCHMARK_CASES) {
        suite->case_count = MAX_BENCHMARK_CASES;
    }

    for (uint32_t i = 0; i < suite->case_count; i++) {
        suite->cases[i] = luganda_cases[i];
        suite->total_expected_tokens += suite->cases[i].expected_tokens;
        suite->total_bytes += strlen(suite->cases[i].text);
    }

    return suite;
}

static void destroy_benchmark_suite(BenchmarkSuite *suite) {
    free(suite);
}

/* Validate tokenization output against expected tokens */
static bool validate_output(const uint32_t *tokens, uint32_t token_count, 
                           uint32_t expected_count, const char *case_name) {
    (void)tokens; /* Suppress unused parameter warning */
    if (token_count != expected_count) {
        printf("VALIDATION FAIL [%s]: expected %u tokens, got %u\n", 
               case_name, expected_count, token_count);
        return false;
    }
    return true;
}

/* Run single benchmark case */
static bool run_case(const Tokenizer *tok, const BenchmarkCase *case_info, 
                    BenchmarkStats *stats) {
    uint32_t out[MAX_TOKENS_PER_CASE];
    uint64_t start_cycles, end_cycles;
    uint32_t token_count;
    bool validation_passed = true;

    /* Warm-up */
    for (int i = 0; i < WARMUP_ITERATIONS; i++) {
        tokenizer_encode_fused(tok, case_info->text, out, MAX_TOKENS_PER_CASE);
    }

    /* Reset tokenizer stats */
    memset((void *)&tok->stats, 0, sizeof(tok->stats));

    /* Timed measurement */
    start_cycles = get_cycles();
    for (int i = 0; i < MEASURE_ITERATIONS; i++) {
        token_count = (uint32_t)tokenizer_encode_fused(tok, case_info->text, out, MAX_TOKENS_PER_CASE);
    }
    end_cycles = get_cycles();

    /* Calculate statistics */
    uint64_t total_cycles = end_cycles - start_cycles;
    uint64_t avg_cycles_per_iter = total_cycles / MEASURE_ITERATIONS;
    
    stats->cycles_total = avg_cycles_per_iter;
    stats->cycles_per_token = token_count > 0 ? avg_cycles_per_iter / token_count : 0;
    stats->total_tokens = token_count;
    stats->total_bytes = strlen(case_info->text);
    stats->avg_token_len = token_count > 0 ? (double)stats->total_bytes / token_count : 0;

    /* Get fast-path statistics */
    stats->fast_path_hits = atomic_load(&tok->stats.fast_path_hits);
    stats->trie_traversals = atomic_load(&tok->stats.trie_traversals);
    stats->utf8_skips = atomic_load(&tok->stats.louds_single_syl_fallbacks);

    uint64_t total_operations = stats->fast_path_hits + stats->trie_traversals;
    stats->fast_path_ratio = total_operations > 0 ? 
        (double)(int64_t)stats->fast_path_hits / (double)(int64_t)total_operations : 0.0;

    /* Convert cycles to milliseconds (approximate 3GHz base) */
    stats->tokens_per_ms = avg_cycles_per_iter > 0 ? 
        ((double)token_count * 3000000000.0) / ((double)avg_cycles_per_iter * 1000000.0) : 0.0;

    /* Validate output */
    validation_passed = validate_output(out, token_count, case_info->expected_tokens, 
                                       case_info->name);

    return validation_passed;
}

/* Print case results in TSV format */
static void print_case_tsv(const BenchmarkCase *case_info, const BenchmarkStats *stats) {
    printf("%s\t%u\t%zu\t%u\t%.2f\t%.2f\t%.2f\t%lu\t%lu\t%lu\t%lu\t%s\n",
           case_info->name,
           case_info->expected_tokens,
           stats->total_bytes,
           (uint32_t)stats->total_tokens,
           stats->avg_token_len,
           stats->tokens_per_ms,
           stats->fast_path_ratio * 100.0,
           stats->fast_path_hits,
           stats->trie_traversals,
           stats->utf8_skips,
           stats->cycles_per_token,
           case_info->description);
}

/* Print summary statistics */
static void print_summary(const BenchmarkSuite *suite, const BenchmarkStats *all_stats, 
                         bool *validation_results) {
    uint32_t passed = 0;
    BenchmarkStats aggregate = {0};
    
    for (uint32_t i = 0; i < suite->case_count; i++) {
        if (validation_results[i]) passed++;
        
        aggregate.total_bytes += all_stats[i].total_bytes;
        aggregate.total_tokens += all_stats[i].total_tokens;
        aggregate.fast_path_hits += all_stats[i].fast_path_hits;
        aggregate.trie_traversals += all_stats[i].trie_traversals;
        aggregate.utf8_skips += all_stats[i].utf8_skips;
        aggregate.cycles_total += all_stats[i].cycles_total;
    }

    aggregate.avg_token_len = aggregate.total_tokens > 0 ? 
        (double)(int64_t)aggregate.total_bytes / (double)(int64_t)aggregate.total_tokens : 0.0;
    
    uint64_t total_operations = aggregate.fast_path_hits + aggregate.trie_traversals;
    aggregate.fast_path_ratio = total_operations > 0 ? 
        (double)(int64_t)aggregate.fast_path_hits / (double)(int64_t)total_operations : 0.0;

    aggregate.tokens_per_ms = aggregate.cycles_total > 0 ? 
        ((double)(int64_t)aggregate.total_tokens * 3000000000.0) / ((double)(int64_t)aggregate.cycles_total * 1000000.0) : 0.0;

    printf("\n=== BENCHMARK SUMMARY ===\n");
    printf("Cases: %u/%u passed (%.1f%%)\n", passed, suite->case_count, 
           100.0 * passed / suite->case_count);
    printf("Total bytes: %lu\n", aggregate.total_bytes);
    printf("Total tokens: %lu\n", aggregate.total_tokens);
    printf("Avg token length: %.2f bytes\n", aggregate.avg_token_len);
    printf("Throughput: %.2f tokens/ms\n", aggregate.tokens_per_ms);
    printf("Fast-path ratio: %.1f%%\n", aggregate.fast_path_ratio * 100.0);
    printf("Fast-path hits: %lu\n", aggregate.fast_path_hits);
    printf("Trie traversals: %lu\n", aggregate.trie_traversals);
    printf("UTF-8 skips: %lu\n", aggregate.utf8_skips);

    /* Analysis by flag */
    printf("\n=== ANALYSIS BY INPUT TYPE ===\n");
    const char *flag_names[] = {
        "EMOJI", "MIXED_LANG", "CRLF", "UNICODE_WS", 
        "LOANWORDS", "MISSPELL", "LONG_INPUT", "EDGE_SPACE"
    };
    
    for (int flag = 0; flag < 8; flag++) {
        uint64_t flag_bytes = 0, flag_tokens = 0, flag_fast_hits = 0, flag_traversals = 0;
        uint32_t flag_count = 0;
        
        for (uint32_t i = 0; i < suite->case_count; i++) {
            if (suite->cases[i].flags & (1 << flag)) {
                flag_bytes += all_stats[i].total_bytes;
                flag_tokens += all_stats[i].total_tokens;
                flag_fast_hits += all_stats[i].fast_path_hits;
                flag_traversals += all_stats[i].trie_traversals;
                flag_count++;
            }
        }
        
        if (flag_count > 0) {
            uint64_t flag_ops = flag_fast_hits + flag_traversals;
            double flag_ratio = flag_ops > 0 ? (double)(int64_t)flag_fast_hits / (double)(int64_t)flag_ops : 0.0;
            printf("%-12s: %u cases, %.1f%% fast-path, %.2f tok/byte\n",
                   flag_names[flag], flag_count, flag_ratio * 100.0,
                   flag_tokens > 0 ? (double)(int64_t)flag_tokens / (double)(int64_t)flag_bytes : 0.0);
        }
    }
}

int main(int argc, char *argv[]) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <model_path> [output_format]\n", argv[0]);
        fprintf(stderr, "  output_format: 'tsv' (default) or 'summary'\n");
        return 1;
    }

    const char *model_path = argv[1];
    const char *output_format = argc >= 3 ? argv[2] : "tsv";

    Tokenizer *tok = tokenizer_load(model_path);
    if (!tok) {
        fprintf(stderr, "Failed to load model from %s\n", model_path);
        return 1;
    }

    BenchmarkSuite *suite = create_benchmark_suite();
    if (!suite) {
        fprintf(stderr, "Failed to create benchmark suite\n");
        tokenizer_destroy(tok);
        return 1;
    }

    if (strcmp(output_format, "summary") == 0) {
        printf("Running %u benchmark cases (%d warmup, %d measurements each)...\n",
               suite->case_count, WARMUP_ITERATIONS, MEASURE_ITERATIONS);
    }

    BenchmarkStats *all_stats = calloc(suite->case_count, sizeof(BenchmarkStats));
    bool *validation_results = calloc(suite->case_count, sizeof(bool));
    
    if (!all_stats || !validation_results) {
        fprintf(stderr, "Memory allocation failed\n");
        free(all_stats);
        free(validation_results);
        destroy_benchmark_suite(suite);
        tokenizer_destroy(tok);
        return 1;
    }

    /* Print TSV header */
    if (strcmp(output_format, "tsv") == 0) {
        printf("case_name\texpected_tokens\tbytes\tactual_tokens\t"
               "avg_token_len\ttokens_per_ms\tfast_path_pct\t"
               "fast_hits\ttrie_traversals\tutf8_skips\tcycles_per_token\t"
               "description\n");
    }

    /* Run all benchmark cases */
    for (uint32_t i = 0; i < suite->case_count; i++) {
        validation_results[i] = run_case(tok, &suite->cases[i], &all_stats[i]);
        
        if (strcmp(output_format, "tsv") == 0) {
            print_case_tsv(&suite->cases[i], &all_stats[i]);
        }
    }

    /* Always print summary */
    print_summary(suite, all_stats, validation_results);

    /* Cleanup */
    free(all_stats);
    free(validation_results);
    destroy_benchmark_suite(suite);
    tokenizer_destroy(tok);

    return 0;
}
