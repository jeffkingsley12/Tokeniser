/*
===============================================================================
COMPLETE PIPELINE DEMONSTRATION (REFACTORED)
Integrates: Word Graph + SCC Reachability + Promotion Layer
-------------------------------------------------------------------------------
This is a reference implementation showing the full transformation:
    Raw Text → Word Graph → SCCs → Metrics → Promotion → DAWG

Improvements:
1. Encapsulated state (DemoContext)
2. Real DAG reachability for cycle detection (no more stubs)
3. Hash map for O(1) node lookup
4. Basic Latin case normalization
5. Sentence boundary awareness (punctuation resets)
===============================================================================
*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include <stdbool.h>
#include <ctype.h>

/* ============================= CONFIG ====================================== */
#define MAX_NODES           8192      /* Increased for larger corpus */
#define MAX_EDGES           32768
#define MAX_SCCS            4096
#define HASH_SIZE           16384     /* Increased for better collision resistance */
#define MAX_WORD_LEN        64
#define INVALID_NODE        UINT32_MAX
#define INVALID_EDGE        UINT32_MAX
#define EPOCH_SIZE          10        
#define PROMOTION_EPOCHS    1         
#define MIN_COHERENCE       0.10f
#define MAX_ENTROPY         5.0f
#define MIN_FREQUENCY       2

/* ============================= TYPES ======================================= */
typedef uint32_t NodeID;
typedef uint32_t EdgeID;
typedef uint32_t SccID;
typedef uint32_t SymbolID;
typedef uint32_t EpochID;

typedef struct Edge {
    NodeID to;
    uint32_t count;
    EdgeID next;
} Edge;

typedef struct Node {
    char word[MAX_WORD_LEN];
    SccID scc_id;
    NodeID next_in_scc;      
    EdgeID first_edge;
    uint32_t edge_count;
    float entropy;           
} Node;

typedef struct {
    NodeID head;             
    NodeID tail;             
    uint32_t member_count;
    uint32_t internal_edges;
    uint32_t external_edges;
    float coherence;
    float avg_entropy;
    EpochID first_seen;
    EpochID last_struct_change; /* Track structural modifications only */
    uint32_t epochs_stable;
    uint64_t traversal_count;
    bool is_promoted;
    SymbolID symbol_id;
    bool is_candidate;
} SccNode;

typedef struct {
    Node nodes[MAX_NODES];
    Edge edges[MAX_EDGES];
    SccNode scc_nodes[MAX_SCCS];
    NodeID node_hash[HASH_SIZE];
    
    /* Transient state for node-to-SCC mapping */
    struct {
        SccID scc_id;
        NodeID next_in_scc;
    } transient_nodes[MAX_NODES];
    
    /* Scratch buffers for reachability search */
    SccID reach_stack[MAX_SCCS];
    bool reach_visited[MAX_SCCS];
    
    uint32_t node_count;
    uint32_t edge_count;
    uint32_t scc_count;
    uint32_t token_count;
    EpochID current_epoch;
    SymbolID next_symbol_id;
    NodeID prev_token;
} DemoContext;

/* ============================= UTILS ======================================= */

static uint32_t hash_word(const char* word) {
    uint32_t hash = 5381;
    int c;
    while ((c = *word++)) hash = ((hash << 5) + hash) + c;
    return hash;
}

static void lowercase_latin(char* s) {
    for (; *s; s++) {
        if (*s >= 'A' && *s <= 'Z') *s = *s + ('a' - 'A');
    }
}

/* ============================= GRAPH OPERATIONS ============================ */

static NodeID get_or_create_node(DemoContext* ctx, const char* word) {
    char clean[MAX_WORD_LEN];
    strncpy(clean, word, MAX_WORD_LEN - 1);
    clean[MAX_WORD_LEN - 1] = '\0';
    lowercase_latin(clean);

    uint32_t h = hash_word(clean) & (HASH_SIZE - 1);
    while (ctx->node_hash[h] != INVALID_NODE) {
        if (strcmp(ctx->nodes[ctx->node_hash[h]].word, clean) == 0)
            return ctx->node_hash[h];
        h = (h + 1) & (HASH_SIZE - 1);
    }
    
    if (ctx->node_count >= MAX_NODES) return INVALID_NODE;
    
    NodeID id = ctx->node_count++;
    ctx->node_hash[h] = id;
    
    strcpy(ctx->nodes[id].word, clean);
    ctx->nodes[id].first_edge = INVALID_EDGE;
    ctx->nodes[id].edge_count = 0;
    ctx->nodes[id].entropy = 0.0f;
    
    if (ctx->scc_count >= MAX_SCCS) return INVALID_NODE;
    SccID scc_id = ctx->scc_count++;
    atomic_store_explicit(&ctx->transient_nodes[id].scc_id, scc_id, memory_order_release);
    atomic_store_explicit(&ctx->transient_nodes[id].next_in_scc, INVALID_NODE, memory_order_release);

    /* Initialize SccNode fields individually since they are now _Atomic */
    atomic_store_explicit(&ctx->scc_nodes[scc_id].head, id, memory_order_release);
    atomic_store_explicit(&ctx->scc_nodes[scc_id].tail, id, memory_order_release);
    atomic_store_explicit(&ctx->scc_nodes[scc_id].member_count, 1, memory_order_release);
    atomic_store_explicit(&ctx->scc_nodes[scc_id].first_seen, ctx->current_epoch, memory_order_relaxed);
    atomic_store_explicit(&ctx->scc_nodes[scc_id].last_modified, ctx->current_epoch, memory_order_relaxed);
    atomic_store_explicit(&ctx->scc_nodes[scc_id].symbol_id, UINT32_MAX, memory_order_release);
    atomic_store_explicit(&ctx->scc_nodes[scc_id].is_promoted, false, memory_order_release);
    atomic_store_explicit(&ctx->scc_nodes[scc_id].is_candidate, false, memory_order_release);
    atomic_store_explicit(&ctx->scc_nodes[scc_id].is_forced, false, memory_order_release);
    atomic_store_explicit(&ctx->scc_nodes[scc_id].is_weak, false, memory_order_relaxed);
    atomic_store_explicit(&ctx->scc_nodes[scc_id].stable_epochs, 0, memory_order_relaxed);
    atomic_store_explicit(&ctx->scc_nodes[scc_id].internal_edges, 0, memory_order_relaxed);
    atomic_store_explicit(&ctx->scc_nodes[scc_id].external_edges, 0, memory_order_relaxed);
    atomic_store_explicit(&ctx->scc_nodes[scc_id].traversal_count, 0, memory_order_relaxed);
    atomic_store_explicit(&ctx->scc_nodes[scc_id].freq, 0, memory_order_relaxed);
    atomic_store_explicit(&ctx->scc_nodes[scc_id].first_scc_edge, INVALID, memory_order_relaxed);
    scc_store_coherence(&ctx->scc_nodes[scc_id], 0.0f);
    scc_store_avg_entropy(&ctx->scc_nodes[scc_id], 0.0f);
    
    return id;
}

/* Reachability check on SCC-DAG to detect cycles */
static bool is_reachable_scc(DemoContext* ctx, SccID start, SccID target) {
    if (start == target) return true;
    
    /* DFS for reachability in SCC DAG — use context buffers, not statics */
    SccID* stack = ctx->reach_stack;
    bool* visited = ctx->reach_visited;
    memset(visited, 0, ctx->scc_count * sizeof(bool));
    
    uint32_t top = 0;
    stack[top++] = start;
    visited[start] = true;
    
    while (top > 0) {
        SccID curr = stack[--top];
        if (curr == target) return true;
        
        /* Iterate all edges leaving this SCC */
        NodeID n = atomic_load_explicit(&ctx->scc_nodes[curr].head, memory_order_acquire);
        while (n != INVALID_NODE) {
            EdgeID e = ctx->nodes[n].first_edge;
            while (e != INVALID_EDGE) {
                SccID dst_scc = atomic_load_explicit(&ctx->transient_nodes[ctx->edges[e].to].scc_id, memory_order_acquire);
                if (dst_scc != curr && !visited[dst_scc]) {
                    visited[dst_scc] = true;
                    stack[top++] = dst_scc;
                }
                e = ctx->edges[e].next;
            }
            n = atomic_load_explicit(&ctx->transient_nodes[n].next_in_scc, memory_order_acquire);
        }
    }
    return false;
}

static void merge_sccs(DemoContext* ctx, SccID target_id, SccID source_id) {
    if (target_id == source_id) return;
    
    SccNode* target = &ctx->scc_nodes[target_id];
    SccNode* source = &ctx->scc_nodes[source_id];
    
    atomic_store_explicit(&ctx->transient_nodes[target->tail].next_in_scc, source->head, memory_order_release) = source->head;
    target->tail = source->tail;
    target->member_count += source->member_count;
    
    NodeID current = source->head;
    while (current != INVALID_NODE) {
        atomic_store_explicit(&ctx->transient_nodes[current].scc_id, target_id, memory_order_release);
        current = ctx->transient_nodes[current].next_in_scc;
    }
    
    target->internal_edges += source->internal_edges;
    target->external_edges += source->external_edges;
    target->traversal_count += source->traversal_count;
    target->last_struct_change = ctx->current_epoch;
    target->epochs_stable = 0;
    
    source->member_count = 0; /* Invalidate */
}

static void add_edge(DemoContext* ctx, NodeID from, NodeID to) {
    if (from == INVALID_NODE || to == INVALID_NODE) return;
    
    SccID scc_from = atomic_load_explicit(&ctx->transient_nodes[from].scc_id, memory_order_acquire);
    SccID scc_to = atomic_load_explicit(&ctx->transient_nodes[to].scc_id, memory_order_acquire);
    
    /* Existing edge check */
    EdgeID e = ctx->nodes[from].first_edge;
    while (e != INVALID_EDGE) {
        if (ctx->edges[e].to == to) {
            ctx->edges[e].count++;
            atomic_fetch_add_explicit(&ctx->scc_nodes[scc_from].traversal_count, 1, memory_order_relaxed);
            return;
        }
        e = ctx->edges[e].next;
    }
    
    /* New edge: Potential structure change */
    if (ctx->edge_count >= MAX_EDGES) return;
    
    /* Cycle detection: if scc_to can already reach scc_from, adding from->to creates a cycle */
    if (scc_from != scc_to && is_reachable_scc(ctx, scc_to, scc_from)) {
        merge_sccs(ctx, scc_from, scc_to);
        scc_to = scc_from;
    }
    
    EdgeID new_e = ctx->edge_count++;
    ctx->edges[new_e] = (Edge){ to, 1, ctx->nodes[from].first_edge };
    ctx->nodes[from].first_edge = new_e;
    ctx->nodes[from].edge_count++;
    
    if (scc_from == scc_to) atomic_fetch_add_explicit(&ctx->scc_nodes[scc_from].internal_edges, 1, memory_order_relaxed);
    else atomic_fetch_add_explicit(&ctx->scc_nodes[scc_from].external_edges, 1, memory_order_relaxed);
    
    atomic_store_explicit(&ctx->scc_nodes[scc_from].last_modified, ctx->current_epoch, memory_order_relaxed);
    atomic_store_explicit(&ctx->scc_nodes[scc_from].stable_epochs, 0, memory_order_relaxed);
    atomic_fetch_add_explicit(&ctx->scc_nodes[scc_from].traversal_count, 1, memory_order_relaxed);
    
    /* Coherence reflow */
    SccNode* s = &ctx->scc_nodes[scc_from];
    uint32_t internal = atomic_load_explicit(&s->internal_edges, memory_order_relaxed);
    uint32_t external = atomic_load_explicit(&s->external_edges, memory_order_relaxed);
    uint32_t tot = internal + external;
    scc_store_coherence(s, (tot > 0) ? (float)internal / tot : 0.0f);
}

/* ============================= ANALYSIS ==================================== */

static void update_metrics(DemoContext* ctx) {
    for (SccID i = 0; i < ctx->scc_count; i++) {
        SccNode* s = &ctx->scc_nodes[i];
        uint32_t member_count = atomic_load_explicit(&s->member_count, memory_order_relaxed);
        if (member_count == 0) continue;
        
        float entropy_sum = 0.0f;
        NodeID n = atomic_load_explicit(&s->head, memory_order_acquire);
        while (n != INVALID_NODE) {
            float node_ent = 0.0f;
            uint32_t out_tot = 0;
            EdgeID e = ctx->nodes[n].first_edge;
            while (e != INVALID_EDGE) { out_tot += ctx->edges[e].count; e = ctx->edges[e].next; }
            if (out_tot > 0) {
                e = ctx->nodes[n].first_edge;
                while (e != INVALID_EDGE) {
                    float p = (float)ctx->edges[e].count / out_tot;
                    if (p > 0) node_ent -= p * log2f(p);
                    e = ctx->edges[e].next;
                }
            }
            ctx->nodes[n].entropy = node_ent;
            entropy_sum += node_ent;
            n = atomic_load_explicit(&ctx->transient_nodes[n].next_in_scc, memory_order_acquire);
        }
        scc_store_avg_entropy(s, entropy_sum / member_count);
        
        float coherence = scc_load_coherence(s);
        float avg_entropy = scc_load_avg_entropy(s);
        uint64_t traversal_count = atomic_load_explicit(&s->traversal_count, memory_order_relaxed);
        s->is_candidate = (coherence >= MIN_COHERENCE) && 
                          (avg_entropy <= MAX_ENTROPY) && 
                          (traversal_count >= MIN_FREQUENCY) && 
                          (atomic_load_explicit(&s->stable_epochs, memory_order_relaxed) >= PROMOTION_EPOCHS);
    }
}

/* ============================= LOOP ======================================== */

static void process_token(DemoContext* ctx, const char* word) {
    NodeID id = get_or_create_node(ctx, word);
    if (id == INVALID_NODE) return;
    
    add_edge(ctx, ctx->prev_token, id);
    ctx->prev_token = id;
    ctx->token_count++;
    
    if (ctx->token_count % EPOCH_SIZE == 0) {
        ctx->current_epoch++;
        update_metrics(ctx);
        for (SccID i = 0; i < ctx->scc_count; i++) {
            SccNode* s = &ctx->scc_nodes[i];
            if (s->member_count == 0) continue;
            if (s->last_struct_change < ctx->current_epoch) s->epochs_stable++;
            if (s->is_candidate && !s->is_promoted) {
                s->is_promoted = true;
                s->symbol_id = ctx->next_symbol_id++;
                printf("[PROMOTION] Epoch %u: SCC %u promoted to Symbol %u (member_count=%u)\n", ctx->current_epoch, i, s->symbol_id, s->member_count);
            }
        }
    }
}

static void run_demo(const char* text) {
    static DemoContext ctx; 
    memset(&ctx, 0, sizeof(ctx));
    for (int i=0; i<HASH_SIZE; i++) ctx.node_hash[i] = INVALID_NODE;
    ctx.prev_token = INVALID_NODE;

    char buf[MAX_WORD_LEN];
    int bi = 0;
    for (const char* p = text; ; p++) {
        char c = *p;
        if (isalpha(c)) {
            if (bi < MAX_WORD_LEN - 1) buf[bi++] = c;
        } else {
            if (bi > 0) {
                buf[bi] = '\0';
                process_token(&ctx, buf);
                bi = 0;
            }
            if (c == '.' || c == '?' || c == '!') ctx.prev_token = INVALID_NODE;
            if (!c) break;
        }
    }
    
    printf("\nFinal Status: Nodes=%u, Edges=%u, Symbols=%u\n", ctx.node_count, ctx.edge_count, ctx.next_symbol_id);
}

int main(int argc, char** argv) {
    const char* corpus_path = (argc > 1) ? argv[1] : "luganda_corpus.txt";
    
    fprintf(stderr, "Opening corpus file: %s\n", corpus_path);
    FILE* f = fopen(corpus_path, "r");
    if (!f) {
        fprintf(stderr, "Error: Cannot open corpus file: %s\n", corpus_path);
        return 1;
    }
    
    fseek(f, 0, SEEK_END);
    long file_size = ftell(f);
    fseek(f, 0, SEEK_SET);
    
    fprintf(stderr, "Corpus file size: %ld bytes\n", file_size);
    
    char* text = malloc(file_size + 1);
    if (!text) {
        fprintf(stderr, "Error: Cannot allocate memory for corpus\n");
        fclose(f);
        return 1;
    }
    
    size_t bytes_read = fread(text, 1, file_size, f);
    text[bytes_read] = '\0';
    fclose(f);
    
    fprintf(stderr, "Training on corpus: %s (%zu bytes)\n", corpus_path, bytes_read);
    run_demo(text);
    
    free(text);
    return 0;
}
