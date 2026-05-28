#include "libgemini.h"
#include <assert.h>
#include <stdio.h>
#include <stdint.h>
#include <stdbool.h>

int main(void) {
    EngineContext *ctx = le_init();
    Tokenizer     *tok = tokenizer_init();
    assert(ctx && tok);

    // Feed a repeated bigram — should form an edge and eventually an SCC
    const char *sentences[] = {
        "omwana asome",
        "omwana agenda",
        "omwana asome",
        "omwana agenda",
        NULL
    };

    for (int i = 0; sentences[i]; i++) {
        tokenizer_process_text(tok, ctx, sentences[i], LE_INVALID);
    }

    uint32_t nodes = get_node_count(ctx);
    uint32_t edges = get_edge_count(ctx);
    printf("Nodes: %u  Edges: %u\n", nodes, edges);
    assert(nodes >= 3);   // at minimum: omwana, asome, agenda
    assert(edges >= 2);

    // Trigger epoch and check promotion machinery doesn't crash
    le_begin_epoch(ctx);
    uint32_t promoted = le_promote_eligible(ctx);
    printf("Promotions after first epoch: %u\n", promoted);

    // Beam search — must not crash even with minimal graph
    uint32_t start_id = tokenizer_get_id(tok, "omwana", false);
    assert(start_id != 0);
    uint32_t result_count = 0;
    BeamState *beam = le_beam_search(ctx, start_id, 3, 5, &result_count);
    printf("Beam results: %u\n", result_count);
    le_free_beam_results(beam, result_count);

    // Save/load round-trip
    const char *save_path = "test_engine_tmp.bin";
    bool saved = le_save_mmap(ctx, save_path);
    assert(saved);
    le_destroy(ctx);
    ctx = le_load_mmap(save_path, false);
    assert(ctx);
    assert(get_node_count(ctx) == nodes);
    le_unload_mmap(ctx);
    remove(save_path);

    tokenizer_destroy(tok);
    printf("PASSED\n");
    return 0;
}
