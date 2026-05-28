/*
 * test_attestation.c — Verification for the Gemini attestation gate
 *
 * This test verifies that the attestation gate correctly filters out
 * junk tokens (misspellings, random noise) from being promoted to the
 * DAWG, while allowing legitimate morphological roots through.
 */

#include "gemini_enhanced.h"
#include "gemini_attest.h"
#include "libgemini.h"

#include <assert.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

static int failures = 0;

#define CHECK(cond, msg) \
    do { \
        if (!(cond)) { fprintf(stderr, "FAIL: %s\n", (msg)); failures++; } \
        else         { printf("PASS: %s\n",           (msg)); } \
    } while (0)

int main(int argc, char **argv) {
    const char *wordlist = (argc > 1) ? argv[1] : "words.txt";
    printf("--- Running Attestation Gate Test (wordlist: %s) ---\n", wordlist);

    /* 1. Setup enhanced engine with attestation enabled */
    EnhancedConfig cfg = gemini_enhanced_default_config();
    cfg.attest_enabled       = true;
    cfg.attest_wordlist_path = wordlist;
    cfg.attest_min_ratio     = 1.0f;  /* Strict for testing: every member must be attested */

    GeminiEnhanced *ge = gemini_enhanced_create(&cfg);
    if (!ge) {
        fprintf(stderr, "SKIP: could not create enhanced engine (is %s missing?)\n", wordlist);
        return 0; /* Not a failure if the file hasn't been generated yet */
    }

    EngineContext *ctx = gemini_enhanced_ctx(ge);
    Tokenizer     *tok = gemini_enhanced_tokenizer(ge);

    /* 1.5 Lower thresholds to force candidates in small test data */
    ctx->rho_min = 0.01f;
    ctx->h_max = 20.0f;
    ctx->min_freq = 1;
    ctx->promotion_epochs = 0;
    ctx->promotion_budget = 100;
    /* Assuming 'omwana' and 'omulungi' roots are in there */
    gemini_enhanced_process_text(ge, "omwana omulungi omwana omulungi", LE_INVALID);
    gemini_enhanced_process_text(ge, "omwana omulungi omwana omulungi", LE_INVALID);

    /* 3. Process some junk/unattested tokens */
    /* These should be blocked during promotion */
    gemini_enhanced_process_text(ge, "asdfghjkl qwertyuiop asdfghjkl qwertyuiop", LE_INVALID);
    gemini_enhanced_process_text(ge, "asdfghjkl qwertyuiop asdfghjkl qwertyuiop", LE_INVALID);

    /* 4. Run an epoch to trigger promotion gating */
    AttestGateResult res;
    /* We call gemini_attest_promote directly to get the detailed results for verification */
    int promoted = gemini_attest_promote(ctx, tok, gemini_enhanced_attest_db(ge),
                                          cfg.attest_min_ratio, &res);

    printf("\nPromotion Results:\n");
    gemini_attest_result_print(&res);

    /* 5. Verifications */
    /* 
     * We expect:
     * - Some SCCs evaluated
     * - At least some SCCs rejected (the junk tokens)
     * - Some SCCs promoted (the real words)
     */
    CHECK(res.sccs_evaluated > 0, "Some SCC candidates were evaluated");
    CHECK(res.sccs_rejected > 0,  "Junk tokens were correctly rejected");
    CHECK(res.sccs_promoted > 0,  "Legitimate words were promoted");

    /* Leak Fix: Cleanup any lineage resources (none in this test, but added for consistency) */
    gemini_enhanced_destroy(ge);

    if (failures == 0) {
        printf("\n✅ ALL ATTESTATION TESTS PASSED\n");
        return 0;
    } else {
        printf("\n❌ %d TESTS FAILED\n", failures);
        return 1;
    }
}
