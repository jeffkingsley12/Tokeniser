/* Test suite for grapheme cluster support */
#include <stdio.h>
#include <string.h>
#include "grapheme_scanner.h"

static int tests_passed = 0;
static int tests_failed = 0;

#define TEST(name) static void test_##name(void)
#define ASSERT(cond) do { if (!(cond)) { \
    printf("  FAIL: %s at line %d\n", #cond, __LINE__); \
    tests_failed++; return; } } while(0)
#define ASSERT_EQ(a, b) ASSERT((a) == (b))

TEST(simple_ascii) {
    const uint8_t *p = (const uint8_t *)"hello";
    grapheme_t g;
    int len = grapheme_next(p, 5, &g);
    ASSERT_EQ(len, 1);
    ASSERT_EQ(g.len, 1);
    ASSERT_EQ(g.ptr[0], 'h');
    tests_passed++;
}

TEST(emoji_single) {
    /* 😊 = U+1F60A = 0xF0 0x9F 0x98 0x8A */
    const uint8_t emoji[] = {0xF0, 0x9F, 0x98, 0x8A, 0};
    grapheme_t g;
    int len = grapheme_next(emoji, 4, &g);
    ASSERT_EQ(len, 4);
    ASSERT(g.props & U_PROP_EMOJI);
    tests_passed++;
}

TEST(emoji_with_modifier) {
    /* 👩🏽 = base + skin tone modifier */
    const uint8_t *p = (const uint8_t *)"\xf0\x9f\x91\xa9\xf0\x9f\x8f\xbd";
    grapheme_t g;
    int len = grapheme_next(p, 8, &g);
    ASSERT_EQ(len, 8); /* Should include both codepoints */
    tests_passed++;
}

TEST(flag_uganda) {
    /* 🇺🇬 = U+1F1FA U+1F1EC = two regional indicators */
    const uint8_t *p = (const uint8_t *)"\xf0\x9f\x87\xba\xf0\x9f\x87\xac";
    grapheme_t g;
    int len = grapheme_next(p, 8, &g);
    ASSERT_EQ(len, 8); /* Both RIs should form one cluster */
    tests_passed++;
}

TEST(combining_mark) {
    /* e + combining acute = é as two codepoints */
    const uint8_t *p = (const uint8_t *)"e\xcc\x81";
    grapheme_t g;
    int len = grapheme_next(p, 3, &g);
    ASSERT_EQ(len, 3); /* base + combining mark */
    tests_passed++;
}

int main(void) {
    printf("=== Grapheme Scanner Test Suite ===\n\n");
    
    test_simple_ascii();
    printf("✓ Simple ASCII\n");
    
    test_emoji_single();
    printf("✓ Single emoji\n");
    
    test_emoji_with_modifier();
    printf("✓ Emoji with modifier\n");
    
    test_flag_uganda();
    printf("✓ Uganda flag (RI pair)\n");
    
    test_combining_mark();
    printf("✓ Combining mark\n");
    
    printf("\n=== Results ===\n");
    printf("Passed: %d\n", tests_passed);
    printf("Failed: %d\n", tests_failed);
    
    return tests_failed > 0 ? 1 : 0;
}
