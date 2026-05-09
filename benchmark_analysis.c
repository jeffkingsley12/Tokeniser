/*
 * benchmark_analysis.c
 * 
 * Comprehensive benchmark analysis tool for Luganda tokenizer performance.
 * Compares before/after performance on SME-scale corpora with detailed metrics.
 * 
 * Features:
 * - Multi-threaded performance testing
 * - Memory usage profiling
 * - Cache performance analysis
 * - Latency distribution analysis
 * - Scalability testing
 * - Regression detection
 * 
 * Author: Cascade Performance Analysis Framework
 * Date: May 4, 2026
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <sys/time.h>
#include <sys/resource.h>
#include <pthread.h>
#include <unistd.h>
#include <assert.h>
#include <math.h>
#include <locale.h>
#include "tokenizer.h"

/* Benchmark configuration */
#define MAX_THREADS 16
#define BENCHMARK_ITERATIONS 1000
#define WARMUP_ITERATIONS 100
#define MAX_CORPUS_SIZE (100 * 1024 * 1024)  /* 100MB */

/* Performance metrics */
typedef struct {
    double min_time;
    double max_time;
    double mean_time;
    double median_time;
    double p95_time;
    double p99_time;
    double std_deviation;
    uint64_t total_tokens;
    double tokens_per_second;
    double chars_per_second;
    size_t peak_memory_kb;
    size_t cache_misses;
    size_t cache_references;
} PerformanceMetrics;

/* Thread-specific data */
typedef struct {
    int thread_id;
    const Tokenizer* tokenizer;
    const char** corpus;
    size_t corpus_size;
    PerformanceMetrics* metrics;
    pthread_barrier_t* barrier;
} ThreadData;

/* Global statistics */
static _Atomic uint64_t g_total_tokens = 0;
static _Atomic uint64_t g_total_chars = 0;
static _Atomic uint64_t g_total_operations = 0;

/* =========================================================
 *  TIMING UTILITIES
 * ========================================================= */

static double get_time_seconds() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec / 1e9;
}

static uint64_t get_rdtsc() {
    unsigned int lo, hi;
    __asm__ __volatile__ ("rdtsc" : "=a" (lo), "=d" (hi));
    return ((uint64_t)hi << 32) | lo;
}

static size_t get_memory_usage_kb() {
    struct rusage usage;
    getrusage(RUSAGE_SELF, &usage);
    return usage.ru_maxrss;  /* KB on Linux */
}

/* =========================================================
 *  STATISTICAL ANALYSIS
 * ========================================================= */

static int compare_doubles(const void* a, const void* b) {
    double da = *(const double*)a;
    double db = *(const double*)b;
    return (da > db) - (da < db);
}

static void calculate_statistics(double* times, size_t count, PerformanceMetrics* metrics) {
    if (count == 0) return;
    
    /* Sort for percentile calculations */
    qsort(times, count, sizeof(double), compare_doubles);
    
    metrics->min_time = times[0];
    metrics->max_time = times[count - 1];
    metrics->median_time = times[count / 2];
    metrics->p95_time = times[(size_t)(count * 0.95)];
    metrics->p99_time = times[(size_t)(count * 0.99)];
    
    /* Calculate mean and standard deviation */
    double sum = 0.0;
    for (size_t i = 0; i < count; i++) {
        sum += times[i];
    }
    metrics->mean_time = sum / count;
    
    double variance = 0.0;
    for (size_t i = 0; i < count; i++) {
        double diff = times[i] - metrics->mean_time;
        variance += diff * diff;
    }
    metrics->std_deviation = sqrt(variance / count);
}

/* =========================================================
 *  BENCHMARK WORKLOADS
 * ========================================================= */

/* Load corpus from file */
static char** load_corpus(const char* filename, size_t* doc_count) {
    FILE* file = fopen(filename, "r");
    if (!file) {
        fprintf(stderr, "Failed to open corpus file: %s\n", filename);
        return NULL;
    }
    
    /* Count lines first */
    size_t lines = 0;
    char buffer[8192];
    while (fgets(buffer, sizeof(buffer), file)) {
        lines++;
    }
    rewind(file);
    
    /* Allocate document array */
    char** docs = calloc(lines, sizeof(char*));
    if (!docs) {
        fclose(file);
        return NULL;
    }
    
    /* Load documents */
    size_t i = 0;
    while (fgets(buffer, sizeof(buffer), file) && i < lines) {
        size_t len = strlen(buffer);
        if (len > 0 && buffer[len - 1] == '\n') {
            buffer[len - 1] = '\0';
            len--;
        }
        
        docs[i] = malloc(len + 1);
        if (docs[i]) {
            strcpy(docs[i], buffer);
            i++;
        }
    }
    
    fclose(file);
    *doc_count = i;
    return docs;
}

/* Generate synthetic workload based on real patterns */
static char** generate_synthetic_corpus(size_t doc_count, size_t avg_doc_length) {
    char** docs = calloc(doc_count, sizeof(char*));
    if (!docs) return NULL;
    
    const char* luganda_words[] = {
        "ekyama", "omulimu", "ekibiina", "omwana", "ennyumba", "essanyu", 
        "obulungi", "amazzi", "ettaka", "emmere", "obulamu", "obuwangwa",
        "ekikula", "omwaka", "enaku", "akawuka", "obulokovu", "endya"
    };
    
    const char* english_words[] = {
        "computer", "internet", "phone", "school", "hospital", "business",
        "technology", "development", "education", "government", "company"
    };
    
    srand(time(NULL));
    
    for (size_t i = 0; i < doc_count; i++) {
        size_t doc_len = avg_doc_length + (rand() % (avg_doc_length / 2));
        char* doc = malloc(doc_len + 1);
        if (!doc) {
            /* Cleanup allocated docs */
            for (size_t j = 0; j < i; j++) free(docs[j]);
            free(docs);
            return NULL;
        }
        
        size_t pos = 0;
        while (pos < doc_len - 20) {  /* Leave room for word + space */
            /* Choose word type (70% Luganda, 20% English, 10% mixed) */
            int word_type = rand() % 100;
            const char* word;
            
            if (word_type < 70) {
                word = luganda_words[rand() % (sizeof(luganda_words) / sizeof(luganda_words[0]))];
            } else if (word_type < 90) {
                word = english_words[rand() % (sizeof(english_words) / sizeof(english_words[0]))];
            } else {
                /* Mixed pattern */
                static char mixed[64];
                snprintf(mixed, sizeof(mixed), "%s%s", 
                        luganda_words[rand() % (sizeof(luganda_words) / sizeof(luganda_words[0]))],
                        english_words[rand() % (sizeof(english_words) / sizeof(english_words[0]))]);
                word = mixed;
            }
            
            size_t word_len = strlen(word);
            if (pos + word_len + 1 < doc_len) {
                strcpy(doc + pos, word);
                pos += word_len;
                doc[pos++] = ' ';
            } else {
                break;
            }
        }
        
        doc[pos] = '\0';
        docs[i] = doc;
    }
    
    return docs;
}

/* =========================================================
 *  BENCHMARK KERNELS
 * ========================================================= */

static void benchmark_tokenization_single_thread(const Tokenizer* tok, 
                                                const char** corpus, 
                                                size_t corpus_size,
                                                PerformanceMetrics* metrics) {
    double* times = calloc(BENCHMARK_ITERATIONS, sizeof(double));
    if (!times) return;
    
    uint32_t tokens[4096];
    size_t total_tokens = 0;
    size_t total_chars = 0;
    size_t peak_memory = 0;
    
    printf("Running single-threaded benchmark (%d iterations)...\n", BENCHMARK_ITERATIONS);
    
    /* Warmup phase */
    for (int i = 0; i < WARMUP_ITERATIONS; i++) {
        for (size_t j = 0; j < corpus_size; j++) {
            tokenizer_encode(tok, corpus[j], tokens, 4096);
        }
    }
    
    /* Benchmark phase */
    for (int i = 0; i < BENCHMARK_ITERATIONS; i++) {
        double start_time = get_time_seconds();
        size_t iter_tokens = 0;
        size_t iter_chars = 0;
        
        for (size_t j = 0; j < corpus_size; j++) {
            size_t doc_len = strlen(corpus[j]);
            int token_count = tokenizer_encode(tok, corpus[j], tokens, 4096);
            iter_tokens += token_count;
            iter_chars += doc_len;
        }
        
        double end_time = get_time_seconds();
        times[i] = end_time - start_time;
        total_tokens += iter_tokens;
        total_chars += iter_chars;
        
        size_t current_memory = get_memory_usage_kb();
        if (current_memory > peak_memory) {
            peak_memory = current_memory;
        }
    }
    
    /* Calculate statistics */
    calculate_statistics(times, BENCHMARK_ITERATIONS, metrics);
    metrics->total_tokens = total_tokens / BENCHMARK_ITERATIONS;
    metrics->tokens_per_second = metrics->total_tokens / metrics->mean_time;
    metrics->chars_per_second = total_chars / (metrics->mean_time * BENCHMARK_ITERATIONS);
    metrics->peak_memory_kb = peak_memory;
    
    free(times);
}

static void* benchmark_thread_worker(void* arg) {
    ThreadData* data = (ThreadData*)arg;
    uint32_t tokens[4096];
    
    /* Wait for all threads to be ready */
    pthread_barrier_wait(data->barrier);
    
    double start_time = get_time_seconds();
    
    /* Process subset of corpus */
    size_t chunk_size = data->corpus_size / MAX_THREADS;
    size_t start_idx = data->thread_id * chunk_size;
    size_t end_idx = (data->thread_id == MAX_THREADS - 1) ? 
                     data->corpus_size : start_idx + chunk_size;
    
    size_t thread_tokens = 0;
    for (size_t i = start_idx; i < end_idx; i++) {
        int token_count = tokenizer_encode(data->tokenizer, data->corpus[i], tokens, 4096);
        thread_tokens += token_count;
    }
    
    double end_time = get_time_seconds();
    
    /* Store results */
    data->metrics->mean_time = end_time - start_time;
    data->metrics->total_tokens = thread_tokens;
    
    /* Update global counters */
    g_total_tokens += thread_tokens;
    g_total_operations++;
    
    return NULL;
}

static void benchmark_tokenization_multi_thread(const Tokenizer* tok,
                                               const char** corpus,
                                               size_t corpus_size,
                                               PerformanceMetrics* metrics) {
    pthread_t threads[MAX_THREADS];
    ThreadData thread_data[MAX_THREADS];
    PerformanceMetrics thread_metrics[MAX_THREADS];
    pthread_barrier_t barrier;
    
    printf("Running multi-threaded benchmark (%d threads)...\n", MAX_THREADS);
    
    pthread_barrier_init(&barrier, NULL, MAX_THREADS);
    
    /* Initialize thread data */
    for (int i = 0; i < MAX_THREADS; i++) {
        thread_data[i].thread_id = i;
        thread_data[i].tokenizer = tok;
        thread_data[i].corpus = corpus;
        thread_data[i].corpus_size = corpus_size;
        thread_data[i].metrics = &thread_metrics[i];
        thread_data[i].barrier = &barrier;
    }
    
    /* Reset global counters */
    g_total_tokens = 0;
    g_total_operations = 0;
    
    /* Start threads */
    double start_time = get_time_seconds();
    for (int i = 0; i < MAX_THREADS; i++) {
        pthread_create(&threads[i], NULL, benchmark_thread_worker, &thread_data[i]);
    }
    
    /* Wait for completion */
    for (int i = 0; i < MAX_THREADS; i++) {
        pthread_join(threads[i], NULL);
    }
    double end_time = get_time_seconds();
    
    pthread_barrier_destroy(&barrier);
    
    /* Calculate aggregate metrics */
    metrics->mean_time = end_time - start_time;
    metrics->total_tokens = g_total_tokens;
    metrics->tokens_per_second = metrics->total_tokens / metrics->mean_time;
    
    /* Calculate thread statistics */
    double thread_times[MAX_THREADS];
    for (int i = 0; i < MAX_THREADS; i++) {
        thread_times[i] = thread_metrics[i].mean_time;
    }
    calculate_statistics(thread_times, MAX_THREADS, metrics);
}

/* =========================================================
 *  MEMORY PROFILING
 * ========================================================= */

static void profile_memory_usage(const Tokenizer* tok) {
    printf("\n📊 Memory Usage Profile\n");
    printf("======================\n");
    
    size_t baseline_memory = get_memory_usage_kb();
    
    /* Tokenizer structure size */
    size_t tokenizer_size = sizeof(Tokenizer);
    printf("Tokenizer structure: %zu bytes\n", tokenizer_size);
    
    /* Vocabulary size */
    printf("Vocabulary size: %u tokens\n", tok->vocab_size);
    printf("Syllable table count: %u\n", tok->stbl ? tok->stbl->count : 0);
    printf("Grammar rules: %u\n", tok->rs ? tok->rs->rule_count : 0);
    printf("LOUDS entries: %u\n", tok->louds ? tok->louds->n_entries : 0);
    
    /* Fast path tables */
    printf("Fast path tables:\n");
    printf("  byte_to_syl: %zu bytes\n", sizeof(tok->byte_to_syl));
    printf("  cv_to_token: %zu bytes\n", sizeof(tok->cv_to_token));
    printf("  byte_to_token: %zu bytes\n", sizeof(tok->byte_to_token));
    
    /* Total memory usage */
    size_t total_memory = get_memory_usage_kb();
    printf("Total memory usage: %zu KB\n", total_memory);
    printf("Memory overhead: %zu KB\n", total_memory - baseline_memory);
}

/* =========================================================
 *  PERFORMANCE ANALYSIS
 * ========================================================= */

static void analyze_cache_performance(const Tokenizer* tok, const char** corpus, size_t corpus_size) {
    printf("\n🎯 Cache Performance Analysis\n");
    printf("============================\n");
    
    /* Test cache locality with different access patterns */
    uint32_t tokens[4096];
    
    /* Sequential access (good cache locality) */
    double start = get_time_seconds();
    for (int iter = 0; iter < 100; iter++) {
        for (size_t i = 0; i < corpus_size && i < 1000; i++) {
            tokenizer_encode(tok, corpus[i], tokens, 4096);
        }
    }
    double sequential_time = get_time_seconds() - start;
    
    /* Random access (poor cache locality) */
    start = get_time_seconds();
    for (int iter = 0; iter < 100; iter++) {
        for (size_t i = 0; i < corpus_size && i < 1000; i++) {
            size_t random_idx = rand() % corpus_size;
            tokenizer_encode(tok, corpus[random_idx], tokens, 4096);
        }
    }
    double random_time = get_time_seconds() - start;
    
    printf("Sequential access time: %.4f seconds\n", sequential_time);
    printf("Random access time: %.4f seconds\n", random_time);
    printf("Cache locality factor: %.2fx\n", random_time / sequential_time);
}

static void analyze_scalability(const Tokenizer* tok) {
    printf("\n📈 Scalability Analysis\n");
    printf("=======================\n");
    
    /* Test with different corpus sizes */
    size_t test_sizes[] = {100, 500, 1000, 5000, 10000};
    size_t num_sizes = sizeof(test_sizes) / sizeof(test_sizes[0]);
    
    char** test_corpus = generate_synthetic_corpus(10000, 100);
    if (!test_corpus) return;
    
    printf("Corpus Size | Tokens/sec | Chars/sec | Latency (ms)\n");
    printf("------------|------------|-----------|-------------\n");
    
    for (size_t i = 0; i < num_sizes; i++) {
        PerformanceMetrics metrics = {0};
        benchmark_tokenization_single_thread(tok, (const char**)test_corpus, test_sizes[i], &metrics);
        
        printf("%11zu | %10.0f | %9.0f | %11.2f\n",
               test_sizes[i],
               metrics.tokens_per_second,
               metrics.chars_per_second,
               metrics.mean_time * 1000);
    }
    
    /* Cleanup */
    for (size_t i = 0; i < 10000; i++) {
        free(test_corpus[i]);
    }
    free(test_corpus);
}

/* =========================================================
 *  REGRESSION DETECTION
 * ========================================================= */

static void detect_regressions(const PerformanceMetrics* baseline, const PerformanceMetrics* current) {
    printf("\n🔍 Regression Analysis\n");
    printf("======================\n");
    
    printf("Metric | Baseline | Current | Change | Status\n");
    printf("-------|----------|---------|--------|--------\n");
    
    /* Define regression thresholds */
    const double regression_threshold = -0.05;  /* 5% degradation */
    const double improvement_threshold = 0.05;  /* 5% improvement */
    
    /* Compare key metrics */
    double throughput_change = (current->tokens_per_second - baseline->tokens_per_second) / baseline->tokens_per_second;
    double latency_change = (current->mean_time - baseline->mean_time) / baseline->mean_time;
    double memory_change = (double)(current->peak_memory_kb - baseline->peak_memory_kb) / baseline->peak_memory_kb;
    
    const char* throughput_status = (throughput_change < regression_threshold) ? "🔴 REGRESSION" :
                                   (throughput_change > improvement_threshold) ? "🟢 IMPROVEMENT" : "🟡 STABLE";
    
    const char* latency_status = (latency_change > -regression_threshold) ? "🔴 REGRESSION" :
                                (latency_change < -improvement_threshold) ? "🟢 IMPROVEMENT" : "🟡 STABLE";
    
    const char* memory_status = (memory_change > -regression_threshold) ? "🔴 REGRESSION" :
                               (memory_change < -improvement_threshold) ? "🟢 IMPROVEMENT" : "🟡 STABLE";
    
    printf("Throughput | %8.0f | %8.0f | %6.1f%% | %s\n",
           baseline->tokens_per_second, current->tokens_per_second,
           throughput_change * 100, throughput_status);
    
    printf("Latency   | %8.3f | %8.3f | %6.1f%% | %s\n",
           baseline->mean_time * 1000, current->mean_time * 1000,
           latency_change * 100, latency_status);
    
    printf("Memory    | %8zu KB | %8zu KB | %6.1f%% | %s\n",
           baseline->peak_memory_kb, current->peak_memory_kb,
           memory_change * 100, memory_status);
}

/* =========================================================
 *  REPORT GENERATION
 * ========================================================= */

static void generate_report(const PerformanceMetrics* single_thread,
                           const PerformanceMetrics* multi_thread,
                           const char* output_file) {
    FILE* file = fopen(output_file, "w");
    if (!file) {
        fprintf(stderr, "Failed to open report file: %s\n", output_file);
        return;
    }
    
    fprintf(file, "# Luganda Tokenizer Performance Report\n");
    fprintf(file, "Generated: %s\n\n", __DATE__);
    
    fprintf(file, "## Single-Threaded Performance\n");
    fprintf(file, "- Mean latency: %.3f ms\n", single_thread->mean_time * 1000);
    fprintf(file, "- Median latency: %.3f ms\n", single_thread->median_time * 1000);
    fprintf(file, "- P95 latency: %.3f ms\n", single_thread->p95_time * 1000);
    fprintf(file, "- P99 latency: %.3f ms\n", single_thread->p99_time * 1000);
    fprintf(file, "- Throughput: %.0f tokens/sec\n", single_thread->tokens_per_second);
    fprintf(file, "- Character throughput: %.0f chars/sec\n", single_thread->chars_per_second);
    fprintf(file, "- Peak memory: %zu KB\n", single_thread->peak_memory_kb);
    
    fprintf(file, "\n## Multi-Threaded Performance\n");
    fprintf(file, "- Mean latency: %.3f ms\n", multi_thread->mean_time * 1000);
    fprintf(file, "- Throughput: %.0f tokens/sec\n", multi_thread->tokens_per_second);
    fprintf(file, "- Scalability factor: %.2fx\n", 
            multi_thread->tokens_per_second / single_thread->tokens_per_second);
    
    fprintf(file, "\n## Performance Distribution\n");
    fprintf(file, "- Min latency: %.3f ms\n", single_thread->min_time * 1000);
    fprintf(file, "- Max latency: %.3f ms\n", single_thread->max_time * 1000);
    fprintf(file, "- Standard deviation: %.3f ms\n", single_thread->std_deviation * 1000);
    
    fclose(file);
    printf("Detailed report saved to: %s\n", output_file);
}

/* =========================================================
 *  MAIN BENCHMARK RUNNER
 * ========================================================= */

int main(int argc, char* argv[]) {
    printf("🚀 Luganda Tokenizer Benchmark Analysis\n");
    printf("========================================\n\n");
    
    const char* model_path = (argc > 1) ? argv[1] : "tokenizer_model.bin";
    const char* corpus_path = (argc > 2) ? argv[2] : "luganda_corpus_10k.txt";
    const char* report_path = (argc > 3) ? argv[3] : "benchmark_report.md";
    
    /* Load tokenizer */
    printf("Loading tokenizer from %s...\n", model_path);
    Tokenizer* tok = tokenizer_load(model_path);
    if (!tok) {
        fprintf(stderr, "Failed to load tokenizer. Building new one...\n");
        
        /* Build simple tokenizer for testing */
        const char* docs[] = {
            "tuyige oluganda emu",
            "katonda owebyewunyo waliwo omukyala",
            "webale nnyo oli otya",
            "ndi mulungi era nsanyuse",
            "ekibiina kyaffe kirimu abantu abangi"
        };
        tok = tokenizer_build(docs, 5);
        if (!tok) {
            fprintf(stderr, "Failed to create tokenizer\n");
            return 1;
        }
    }
    
    printf("Tokenizer loaded. Vocab size: %u\n\n", tok->vocab_size);
    
    /* Load or generate corpus */
    printf("Loading corpus from %s...\n", corpus_path);
    size_t corpus_size = 0;
    char** corpus = load_corpus(corpus_path, &corpus_size);
    
    if (!corpus) {
        printf("Failed to load corpus, generating synthetic data...\n");
        corpus_size = 1000;
        corpus = generate_synthetic_corpus(corpus_size, 200);
    }
    
    if (!corpus) {
        fprintf(stderr, "Failed to load or generate corpus\n");
        tokenizer_destroy(tok);
        return 1;
    }
    
    printf("Corpus loaded: %zu documents\n\n", corpus_size);
    
    /* Run benchmarks */
    PerformanceMetrics single_thread = {0};
    PerformanceMetrics multi_thread = {0};
    
    /* Single-threaded benchmark */
    benchmark_tokenization_single_thread(tok, (const char**)corpus, corpus_size, &single_thread);
    
    /* Multi-threaded benchmark */
    benchmark_tokenization_multi_thread(tok, (const char**)corpus, corpus_size, &multi_thread);
    
    /* Additional analysis */
    profile_memory_usage(tok);
    analyze_cache_performance(tok, (const char**)corpus, corpus_size);
    analyze_scalability(tok);
    
    /* Print summary */
    printf("\n📊 Performance Summary\n");
    printf("======================\n");
    printf("Single-threaded:\n");
    printf("  Latency: %.3f ms (mean), %.3f ms (p95)\n", 
           single_thread.mean_time * 1000, single_thread.p95_time * 1000);
    printf("  Throughput: %.0f tokens/sec\n", single_thread.tokens_per_second);
    printf("  Memory: %zu KB\n", single_thread.peak_memory_kb);
    
    printf("\nMulti-threaded (%d threads):\n", MAX_THREADS);
    printf("  Throughput: %.0f tokens/sec\n", multi_thread.tokens_per_second);
    printf("  Scalability: %.2fx\n", multi_thread.tokens_per_second / single_thread.tokens_per_second);
    
    /* Generate detailed report */
    generate_report(&single_thread, &multi_thread, report_path);
    
    /* Cleanup */
    for (size_t i = 0; i < corpus_size; i++) {
        free(corpus[i]);
    }
    free(corpus);
    tokenizer_destroy(tok);
    
    printf("\n✅ Benchmark analysis completed!\n");
    return 0;
}
