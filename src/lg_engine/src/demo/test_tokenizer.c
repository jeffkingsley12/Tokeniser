#include "libgemini.h"
#include <assert.h>
#include <string.h>
#include <stdio.h>
#include <stdint.h>

static int failures = 0;
#define CHECK(cond, msg) \
    do { if (!(cond)) { fprintf(stderr, "FAIL: %s\n", msg); failures++; } \
         else { printf("PASS: %s\n", msg); } } while(0)

int main(void) {
    Tokenizer *t = tokenizer_init();
    assert(t);

    // Basic round-trip
    uint32_t id = tokenizer_get_id(t, "omwana", true);
    CHECK(id != 0, "omwana gets a non-zero ID");
    CHECK(strcmp(tokenizer_get_word(t, id), "omwana") == 0, "reverse lookup");

    // Same word twice → same ID
    uint32_t id2 = tokenizer_get_id(t, "omwana", true);
    CHECK(id == id2, "idempotent insertion");

    // Unknown word, no-create → 0
    uint32_t miss = tokenizer_get_id(t, "zzznope", false);
    CHECK(miss == 0, "miss returns 0, not LE_INVALID");

    // Token 0 is always invalid
    CHECK(strcmp(tokenizer_get_word(t, 0), "<?>") == 0, "ID 0 returns sentinel");

    // Save/load round-trip
    const char *save_path = "test_tok_tmp.bin";
    tokenizer_save(t, save_path);
    tokenizer_destroy(t);
    t = tokenizer_load(save_path);
    assert(t);
    CHECK(tokenizer_get_id(t, "omwana", false) == id, "survives save/load");
    /* Thread Safety: Verify tokenizer's mutex is correctly re-initialized during load */
    CHECK(tokenizer_get_id(t, "omwana", true) == id, "tokenizer mutex functional after load");
    remove(save_path);

    tokenizer_destroy(t);
    printf("\n%s — %d failure(s)\n", failures ? "FAILED" : "PASSED", failures);
    return failures ? 1 : 0;
}
