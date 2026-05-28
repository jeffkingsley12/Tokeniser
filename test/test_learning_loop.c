#include "tokenizer_api.h"
#include "gemini_internal.h"
#include "bridge_engine.h"
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

int main() {
    printf("=== Test Gemini Learning Loop ===\n");

    /* Create dummy Tokenizer and Engine */
    int tok = tok_load("production_model.bin");
    if (tok < 0) {
        printf("Failed to load tokenizer production_model.bin\n");
        return 1;
    }

    EngineContext* ctx = le_init();
    if (!ctx) {
        printf("Failed to init engine\n");
        return 1;
    }
    ctx->promotion_epochs = 2;

    /* Dummy small repetitive corpus to trigger merging */
    const char* corpus[] = {
        "omwana omulungi",
        "omwana omulungi",
        "omwana omulungi",
        "omwana omulungi",
        "omwana omulungi",
        "omwana omulungi",
        "omwana omulungi",
        "omwana omulungi"
    };

    uint32_t initial_sccs = get_scc_count(ctx);
    printf("Initial SCC count: %u\n", initial_sccs);

    /* Run 5 epochs */
    for (int epoch = 1; epoch <= 5; epoch++) {
        printf("\n--- Epoch %d Ingestion ---\n", epoch);
        for (int i = 0; i < 8; i++) {
            engine_ingest_text(tok, ctx, corpus[i], INVALID);
        }
        
        uint32_t def_cnt = atomic_load(&ctx->deferred_scc_count);
        printf("Before le_begin_epoch: deferred_scc_count = %u\n", def_cnt);
        for (uint32_t d = 0; d < def_cnt && d < 10; d++) {
            uint32_t from = atomic_load(&ctx->deferred_scc_checks[d].from);
            uint32_t to = ctx->deferred_scc_checks[d].to;
            if (from != INVALID) {
                printf("  Deferred[%u]: %u -> %u\n", d, from, to);
            } else {
                printf("  Deferred[%u]: INVALID\n", d);
            }
        }
        
        le_begin_epoch(ctx);
        printf("Epoch %d: SCCs=%u, Merges=%lu, Promotions=%lu\n", 
               epoch, get_scc_count(ctx), 
               (unsigned long)atomic_load(&ctx->total_merges),
               (unsigned long)atomic_load(&ctx->total_promotions));
    }

    uint64_t final_merges = atomic_load(&ctx->total_merges);
    uint64_t final_promotions = atomic_load(&ctx->total_promotions);

    printf("Final Merges: %lu\n", (unsigned long)final_merges);
    printf("Final Promotions: %lu\n", (unsigned long)final_promotions);

    /* Diagnostic: verify the integrity of your first promoted unit */
    SymbolID first_unit = 0;
    NodeID nodes[128];
    uint32_t count = le_get_symbol_nodes(ctx, first_unit, nodes, 128);
    printf("Promoted Symbol 0 contains %u nodes.\n", count);
    assert(count > 1 && "Expected the promoted symbol to contain more than 1 node.");

    printf("✅ test_learning_loop PASSED\n");

    /* Cleanup (optional since test exits anyway) */
    tok_free(tok);
    return 0;
}
