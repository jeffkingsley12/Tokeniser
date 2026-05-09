#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include "tokenizer.h"
#include "truth_layer.h"

/* Extern declaration for internal function */
int truth_trie_insert(TruthTrie *tt, const uint16_t *sids, size_t len, uint32_t category);

int main() {
    TruthTrie *tt = truth_trie_create();
    if (!tt) {
        fprintf(stderr, "Failed to create TruthTrie\n");
        return 1;
    }

    uint16_t seed[] = { 100, 101 };
    uint32_t category = TRUTH_PREFIX;

    /* 1. Insert seed */
    int ret = truth_trie_insert(tt, seed, 2, category);
    assert(ret == 0);

    /* 2. Compute failure links */
    tt->has_csr = true;
    ret = truth_trie_compute_failure_links(tt);
    assert(ret == 0);

    /* 3. Match */
    TruthMatchInfo match = {0};
    int len = truth_match_ids(tt, seed, 2, &match);
    
    printf("Matched len: %d (expected 2)\n", len);
    printf("Matched category mask: %u (expected %u)\n", match.category_mask, category);

    if (len != 2 || match.category_mask != category) {
        printf("TEST FAILED!\n");
        return 1;
    }

    /* Extra test: test suffix matching via Aho Corasick */
    uint16_t stream[] = { 50, 100, 101, 60 };
    TruthMatch matches[10];
    int num_matches = aho_corasick_find_matches(tt, stream, 4, matches, 10);
    printf("Found %d matches in stream (expected 1)\n", num_matches);
    if (num_matches > 0) {
        printf("First match: start=%u, len=%u, mask=%u\n", 
               matches[0].start_pos, matches[0].length, matches[0].category_mask);
        assert(matches[0].start_pos == 1);
        assert(matches[0].length == 2);
        assert(matches[0].category_mask == category);
    }

    /* Test overlapping matches */
    uint16_t seed2[] = { 101, 60 };
    truth_trie_insert(tt, seed2, 2, TRUTH_ROOT);
    truth_trie_compute_failure_links(tt);
    
    num_matches = aho_corasick_find_matches(tt, stream, 4, matches, 10);
    printf("Found %d matches in stream with overlap (expected 2)\n", num_matches);
    for (int i = 0; i < num_matches; i++) {
        printf("Match %d: start=%u, len=%u, mask=%u\n", 
               i, matches[i].start_pos, matches[i].length, matches[i].category_mask);
    }
    assert(num_matches == 2);

    /* Issue #2: Edge contiguity test */
    /* Insert "mba" (200, 201) and "mwa" (200, 202) */
    uint16_t mba[] = { 200, 201 };
    uint16_t mwa[] = { 200, 202 };
    
    truth_trie_insert(tt, mba, 2, TRUTH_PREFIX);
    
    /* Simulate intervening allocations by inserting another unrelated word */
    uint16_t unrelated[] = { 300, 301 };
    truth_trie_insert(tt, unrelated, 2, TRUTH_ROOT);
    
    /* Now insert "mwa". This will add an edge for 202 to node 'm' (which was node 1) */
    /* Since 'unrelated' edges were added after 'mba' edges, 'mwa' edge insertion 
     * should trigger the shift logic. */
    truth_trie_insert(tt, mwa, 2, TRUTH_PREFIX);
    
    truth_trie_compute_failure_links(tt);
    
    /* Verify both matches */
    num_matches = aho_corasick_find_matches(tt, mba, 2, matches, 10);
    printf("MBA match: %d\n", num_matches);
    assert(num_matches == 1);
    num_matches = aho_corasick_find_matches(tt, mwa, 2, matches, 10);
    printf("MWA match: %d\n", num_matches);
    assert(num_matches == 1);

    printf("TEST PASSED!\n");
    truth_trie_destroy(tt);
    return 0;
}
