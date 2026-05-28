#include "libgemini.h"
#include "tokenizer_api.h"
#include "gemini_internal.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdbool.h>

typedef struct {
    const char *path;
    uint32_t tokens_processed;
    uint32_t unique_nodes;
    uint32_t total_edges;
    uint32_t active_sccs;
    uint32_t promoted_symbols;
    uint64_t total_merges;
    uint64_t total_promotions;
    float avg_nodes_per_sym;
    uint32_t max_nodes_per_sym;
    float avg_transition_entropy;
} ModelStats;

static bool collect_stats(const char *path, ModelStats *stats) {
    EngineContext *ctx = le_load_mmap(path, false);
    if (!ctx) {
        return false;
    }

    stats->path = path;
    stats->tokens_processed = get_token_count(ctx);
    stats->unique_nodes = get_node_count(ctx);
    stats->total_edges = get_edge_count(ctx);
    stats->active_sccs = get_scc_count(ctx);
    stats->promoted_symbols = get_symbol_count(ctx);
    stats->total_merges = get_total_merges(ctx);
    stats->total_promotions = get_total_promotions(ctx);

    uint32_t total_nodes_in_syms = 0;
    uint32_t max_nodes = 0;
    float total_entropy = 0.0f;
    uint32_t symbols_with_transitions = 0;

    for (uint32_t i = 0; i < stats->promoted_symbols; i++) {
        NodeID member_buf[1024];
        uint32_t mc = le_get_symbol_nodes(ctx, i, member_buf, 1024);
        total_nodes_in_syms += mc;
        if (mc > max_nodes) {
            max_nodes = mc;
        }

        // Compute Shannon entropy of transitions
        Symbol *sym = &ctx->dawg_nodes[i];
        uint32_t t_idx = atomic_load_explicit(&sym->first_transition, memory_order_acquire);
        
        float total_weight = 0.0f;
        uint32_t temp_idx = t_idx;
        while (temp_idx != INVALID) {
            DawgTransition *t = &ctx->dawg_transitions[temp_idx];
            float w = atomic_load_float(&t->weight);
            if (w > 0.0f) {
                total_weight += w;
            }
            temp_idx = t->next;
        }

        if (total_weight > 0.0f) {
            float entropy = 0.0f;
            temp_idx = t_idx;
            while (temp_idx != INVALID) {
                DawgTransition *t = &ctx->dawg_transitions[temp_idx];
                float w = atomic_load_float(&t->weight);
                if (w > 0.0f) {
                    float p = w / total_weight;
                    entropy -= p * log2f(p);
                }
                temp_idx = t->next;
            }
            total_entropy += entropy;
            symbols_with_transitions++;
        }
    }

    stats->avg_nodes_per_sym = stats->promoted_symbols > 0 ? (float)total_nodes_in_syms / stats->promoted_symbols : 0.0f;
    stats->max_nodes_per_sym = max_nodes;
    stats->avg_transition_entropy = symbols_with_transitions > 0 ? total_entropy / symbols_with_transitions : 0.0f;

    le_unload_mmap(ctx);
    return true;
}

int main(int argc, char **argv) {
    if (argc < 3) {
        printf("Usage: %s <snapshot_a.bin> <snapshot_b.bin>\n", argv[0]);
        return 1;
    }

    ModelStats stats_a = {0};
    ModelStats stats_b = {0};

    printf("Analyzing Snapshot A: %s ...\n", argv[1]);
    if (!collect_stats(argv[1], &stats_a)) {
        fprintf(stderr, "Error loading Snapshot A: %s\n", argv[1]);
        return 1;
    }

    printf("Analyzing Snapshot B: %s ...\n", argv[2]);
    if (!collect_stats(argv[2], &stats_b)) {
        fprintf(stderr, "Error loading Snapshot B: %s\n", argv[2]);
        return 1;
    }

    printf("\n");
    printf("╔═══════════════════════════════════════════════════════════════════════════╗\n");
    printf("║                   GEMINI COGNITIVE SNAPSHOT COMPARISON                    ║\n");
    printf("╠══════════════════════════════╦══════════════════════╦═════════════════════╣\n");
    printf("║ Metric                       ║ Snapshot A           ║ Snapshot B          ║\n");
    printf("╠══════════════════════════════╬══════════════════════╬═════════════════════╣\n");
    printf("║ File Path                    ║ %-20.20s ║ %-19.19s ║\n", stats_a.path, stats_b.path);
    printf("║ Tokens Processed             ║ %-20u ║ %-19u ║\n", stats_a.tokens_processed, stats_b.tokens_processed);
    printf("║ Unique Nodes                 ║ %-20u ║ %-19u ║\n", stats_a.unique_nodes, stats_b.unique_nodes);
    printf("║ Total Edges                  ║ %-20u ║ %-19u ║\n", stats_a.total_edges, stats_b.total_edges);
    printf("║ Active SCCs                  ║ %-20u ║ %-19u ║\n", stats_a.active_sccs, stats_b.active_sccs);
    printf("║ Promoted Symbols             ║ %-20u ║ %-19u ║\n", stats_a.promoted_symbols, stats_b.promoted_symbols);
    printf("║ Total Merges                 ║ %-20lu ║ %-19lu ║\n", (unsigned long)stats_a.total_merges, (unsigned long)stats_b.total_merges);
    printf("║ Total Promotions             ║ %-20lu ║ %-19lu ║\n", (unsigned long)stats_a.total_promotions, (unsigned long)stats_b.total_promotions);
    printf("║ Avg Nodes / Promoted Sym     ║ %-20.4f ║ %-19.4f ║\n", stats_a.avg_nodes_per_sym, stats_b.avg_nodes_per_sym);
    printf("║ Max Nodes in a Promoted Sym  ║ %-20u ║ %-19u ║\n", stats_a.max_nodes_per_sym, stats_b.max_nodes_per_sym);
    printf("║ Avg Transition Entropy       ║ %-20.4f ║ %-19.4f ║\n", stats_a.avg_transition_entropy, stats_b.avg_transition_entropy);
    printf("╚══════════════════════════════╩══════════════════════╩═════════════════════╝\n");
    printf("\n");

    return 0;
}
