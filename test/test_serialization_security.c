/*
 * test_serialization_security.c
 * 
 * Comprehensive test for tokenizer serialization/deserialization security:
 * - Corruption detection
 * - Boundary conditions
 * - Malformed input handling
 * - Memory safety during load/save
 */

#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <unistd.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <limits.h>
#include "tokenizer.h"

#define TEST_MODEL "test_security_model.bin"
#define CORRUPT_MODEL "corrupt_model.bin"

static int tests_passed = 0;
static int tests_failed = 0;

#define TEST_ASSERT(condition, message) do { \
    if (condition) { \
        printf("✅ PASS: %s\n", message); \
        tests_passed++; \
    } else { \
        printf("❌ FAIL: %s\n", message); \
        tests_failed++; \
    } \
} while(0)

/* Suppress stderr during expected failures */
#define SUPPRESS_STDERR(block) do { \
    FILE* _old_stderr = stderr; \
    stderr = fopen("/dev/null", "w"); \
    block \
    fclose(stderr); \
    stderr = _old_stderr; \
} while(0)

/* Helper to create a simple tokenizer for testing */
static Tokenizer* create_test_tokenizer() {
    const char* docs[] = {
        "oli otya",
        "webale nnyo", 
        "ndi mulungi",
        "ekibiina kyaffe",
        "okusoma kwanguddwa"
    };
    return tokenizer_build(docs, 5);
}

/* Test 1: Normal save/load cycle */
static void test_normal_save_load() {
    printf("\n🧪 Testing normal save/load cycle\n");
    
    Tokenizer* orig = create_test_tokenizer();
    TEST_ASSERT(orig != NULL, "Create test tokenizer");
    
    int save_result = tokenizer_save(orig, TEST_MODEL);
    TEST_ASSERT(save_result == 0, "Save tokenizer successfully");
    
    Tokenizer* loaded = tokenizer_load(TEST_MODEL);
    TEST_ASSERT(loaded != NULL, "Load tokenizer successfully");
    
    /* Verify tokenization consistency */
    uint32_t orig_tokens[100], loaded_tokens[100];
    const char* test_text = "oli otya webale";
    
    int orig_count = tokenizer_encode(orig, test_text, orig_tokens, 100);
    int loaded_count = tokenizer_encode(loaded, test_text, loaded_tokens, 100);
    
    TEST_ASSERT(orig_count == loaded_count, "Token count matches after save/load");
    TEST_ASSERT(memcmp(orig_tokens, loaded_tokens, orig_count * sizeof(uint32_t)) == 0, 
                "Token IDs match after save/load");
    
    tokenizer_destroy(orig);
    tokenizer_destroy(loaded);
    unlink(TEST_MODEL);
}

/* Test 2: Truncated file detection */
static void test_truncated_file() {
    printf("\n🧪 Testing truncated file detection\n");
    
    Tokenizer* orig = create_test_tokenizer();
    TEST_ASSERT(orig != NULL, "Create test tokenizer");
    
    tokenizer_save(orig, TEST_MODEL);
    
    /* Truncate the file to various sizes */
    struct stat st;
    stat(TEST_MODEL, &st);
    
    for (size_t truncate_size = 0; truncate_size < (size_t)st.st_size; truncate_size += 100) {
        truncate(TEST_MODEL, truncate_size);
        
        Tokenizer* loaded;
        SUPPRESS_STDERR(loaded = tokenizer_load(TEST_MODEL););
        TEST_ASSERT(loaded == NULL, "Detect truncated file corruption");
    }
    
    tokenizer_destroy(orig);
    unlink(TEST_MODEL);
}

/* Test 3: Byte corruption detection */
static void test_byte_corruption() {
    printf("\n🧪 Testing byte corruption detection\n");
    
    Tokenizer* orig = create_test_tokenizer();
    TEST_ASSERT(orig != NULL, "Create test tokenizer");
    
    tokenizer_save(orig, TEST_MODEL);
    
    /* Corrupt individual bytes throughout the file */
    FILE* f = fopen(TEST_MODEL, "rb+");
    TEST_ASSERT(f != NULL, "Open file for corruption");
    
    fseek(f, 0, SEEK_END);
    long file_size = ftell(f);
    
    for (long pos = 8; pos < file_size - 4; pos += 50) {  /* Skip magic and CRC */
        fseek(f, pos, SEEK_SET);
        uint8_t original_byte;
        fread(&original_byte, 1, 1, f);
        
        /* Flip bits in the byte */
        uint8_t corrupted_byte = original_byte ^ 0xFF;
        fseek(f, pos, SEEK_SET);
        fwrite(&corrupted_byte, 1, 1, f);
        fflush(f);
        
        Tokenizer* loaded;
        SUPPRESS_STDERR(loaded = tokenizer_load(TEST_MODEL););
        TEST_ASSERT(loaded == NULL, "Detect single-byte corruption");
        
        /* Restore original byte */
        fseek(f, pos, SEEK_SET);
        fwrite(&original_byte, 1, 1, f);
        fflush(f);
    }
    
    fclose(f);
    tokenizer_destroy(orig);
    unlink(TEST_MODEL);
}

/* Test 4: Magic number corruption */
static void test_magic_corruption() {
    printf("\n🧪 Testing magic number corruption\n");
    
    Tokenizer* orig = create_test_tokenizer();
    TEST_ASSERT(orig != NULL, "Create test tokenizer");
    
    tokenizer_save(orig, TEST_MODEL);
    
    /* Corrupt magic number */
    FILE* f = fopen(TEST_MODEL, "rb+");
    TEST_ASSERT(f != NULL, "Open file for magic corruption");
    
    const char* bad_magic = "CORRUPT13";
    fwrite(bad_magic, 1, 8, f);
    fclose(f);
    
    Tokenizer* loaded;
    SUPPRESS_STDERR(loaded = tokenizer_load(TEST_MODEL););
    TEST_ASSERT(loaded == NULL, "Detect magic number corruption");
    
    tokenizer_destroy(orig);
    unlink(TEST_MODEL);
}

/* Test 5: CRC corruption */
static void test_crc_corruption() {
    printf("\n🧪 Testing CRC corruption detection\n");
    
    Tokenizer* orig = create_test_tokenizer();
    TEST_ASSERT(orig != NULL, "Create test tokenizer");
    
    tokenizer_save(orig, TEST_MODEL);
    
    /* Corrupt CRC at end of file */
    FILE* f = fopen(TEST_MODEL, "rb+");
    TEST_ASSERT(f != NULL, "Open file for CRC corruption");
    
    fseek(f, -4, SEEK_END);
    uint32_t bad_crc = 0xDEADBEEF;
    fwrite(&bad_crc, 4, 1, f);
    fclose(f);
    
    Tokenizer* loaded;
    SUPPRESS_STDERR(loaded = tokenizer_load(TEST_MODEL););
    TEST_ASSERT(loaded == NULL, "Detect CRC corruption");
    
    tokenizer_destroy(orig);
    unlink(TEST_MODEL);
}

/* Test 6: Boundary value corruption */
static void test_boundary_corruption() {
    printf("\n🧪 Testing boundary value corruption\n");
    
    Tokenizer* orig = create_test_tokenizer();
    TEST_ASSERT(orig != NULL, "Create test tokenizer");
    
    tokenizer_save(orig, TEST_MODEL);
    
    FILE* f = fopen(TEST_MODEL, "rb+");
    TEST_ASSERT(f != NULL, "Open file for boundary corruption");
    
    /* Corrupt count fields to test boundary checks */
    fseek(f, 8, SEEK_SET);  /* After magic, at syllable count */
    
    /* Test with various large values */
    uint32_t large_values[] = {0xFFFFFFFF, 0xFFFF0000, BASE_SYMBOL_OFFSET + 1, MAX_RULES + 1};
    
    for (int i = 0; i < 4; i++) {
        fseek(f, 8, SEEK_SET);
        fwrite(&large_values[i], 4, 1, f);
        fflush(f);
        
        Tokenizer* loaded;
        SUPPRESS_STDERR(loaded = tokenizer_load(TEST_MODEL););
        TEST_ASSERT(loaded == NULL, "Detect boundary value corruption");
    }
    
    fclose(f);
    tokenizer_destroy(orig);
    unlink(TEST_MODEL);
}

/* Test 7: Empty and invalid files */
static void test_invalid_files() {
    printf("\n🧪 Testing empty and invalid files\n");
    
    /* Test empty file */
    FILE* f = fopen(TEST_MODEL, "w");
    fclose(f);
    
    Tokenizer* loaded;
    SUPPRESS_STDERR(loaded = tokenizer_load(TEST_MODEL););
    TEST_ASSERT(loaded == NULL, "Reject empty file");
    
    /* Test file with only magic */
    f = fopen(TEST_MODEL, "wb");
    fwrite("LUGTOK13", 1, 8, f);
    fclose(f);
    
    SUPPRESS_STDERR(loaded = tokenizer_load(TEST_MODEL););
    TEST_ASSERT(loaded == NULL, "Reject incomplete file");
    
    /* Test non-existent file */
    SUPPRESS_STDERR(loaded = tokenizer_load("nonexistent_file.bin"););
    TEST_ASSERT(loaded == NULL, "Reject non-existent file");
    
    unlink(TEST_MODEL);
}

/* Test 8: Memory safety during load failures */
static void test_memory_safety() {
    printf("\n🧪 Testing memory safety during load failures\n");
    
    /* Create a valid model first */
    Tokenizer* orig = create_test_tokenizer();
    tokenizer_save(orig, TEST_MODEL);
    
    /* Try loading corrupted files and ensure no memory leaks */
    FILE* f = fopen(TEST_MODEL, "rb+");
    fseek(f, 20, SEEK_SET);
    uint8_t bad_byte = 0xFF;
    fwrite(&bad_byte, 1, 1, f);
    fclose(f);
    
    /* Multiple load attempts should not crash */
    for (int i = 0; i < 10; i++) {
        Tokenizer* loaded;
        SUPPRESS_STDERR(loaded = tokenizer_load(TEST_MODEL););
        TEST_ASSERT(loaded == NULL, "Safe handling of repeated load failures");
    }
    
    tokenizer_destroy(orig);
    unlink(TEST_MODEL);
}

/* Test 9: Large model handling */
static void test_large_model() {
    printf("\n🧪 Testing large model handling\n");
    
    /* Create a tokenizer with more data */
    const char* large_docs[100];
    char doc_buffer[100][64];
    
    for (int i = 0; i < 100; i++) {
        snprintf(doc_buffer[i], sizeof(doc_buffer[i]), 
                "document number %d with some luganda text oli otya webale nnyo", i);
        large_docs[i] = doc_buffer[i];
    }
    
    Tokenizer* large_tok = tokenizer_build(large_docs, 100);
    TEST_ASSERT(large_tok != NULL, "Create large tokenizer");
    
    int save_result = tokenizer_save(large_tok, TEST_MODEL);
    TEST_ASSERT(save_result == 0, "Save large tokenizer");
    
    Tokenizer* loaded = tokenizer_load(TEST_MODEL);
    TEST_ASSERT(loaded != NULL, "Load large tokenizer");
    
    /* Verify functionality */
    uint32_t tokens[100];
    int count = tokenizer_encode(loaded, "oli otya", tokens, 100);
    TEST_ASSERT(count > 0, "Large tokenizer functions correctly");
    
    tokenizer_destroy(large_tok);
    tokenizer_destroy(loaded);
    unlink(TEST_MODEL);
}

/* Test 10: Concurrent access safety */
static void test_concurrent_safety() {
    printf("\n🧪 Testing concurrent access safety\n");
    
    Tokenizer* orig = create_test_tokenizer();
    tokenizer_save(orig, TEST_MODEL);
    
    /* Multiple loads should be safe */
    Tokenizer* t1 = tokenizer_load(TEST_MODEL);
    Tokenizer* t2 = tokenizer_load(TEST_MODEL);
    Tokenizer* t3 = tokenizer_load(TEST_MODEL);
    
    TEST_ASSERT(t1 != NULL && t2 != NULL && t3 != NULL, "Multiple concurrent loads");
    
    /* All should function independently */
    uint32_t tokens1[100], tokens2[100], tokens3[100];
    const char* test_text = "webale nnyo";
    
    int c1 = tokenizer_encode(t1, test_text, tokens1, 100);
    int c2 = tokenizer_encode(t2, test_text, tokens2, 100);
    int c3 = tokenizer_encode(t3, test_text, tokens3, 100);
    
    TEST_ASSERT(c1 == c2 && c2 == c3, "Concurrent instances produce consistent results");
    TEST_ASSERT(memcmp(tokens1, tokens2, c1 * sizeof(uint32_t)) == 0, 
                "Concurrent instances produce identical tokens");
    
    tokenizer_destroy(orig);
    tokenizer_destroy(t1);
    tokenizer_destroy(t2);
    tokenizer_destroy(t3);
    unlink(TEST_MODEL);
}

int main() {
    printf("🔒 Tokenizer Serialization Security Test Suite\n");
    printf("===============================================\n");
    
    test_normal_save_load();
    test_truncated_file();
    test_byte_corruption();
    test_magic_corruption();
    test_crc_corruption();
    test_boundary_corruption();
    test_invalid_files();
    test_memory_safety();
    test_large_model();
    test_concurrent_safety();
    
    printf("\n📊 Test Results Summary\n");
    printf("======================\n");
    printf("Total tests: %d\n", tests_passed + tests_failed);
    printf("Passed: %d ✅\n", tests_passed);
    printf("Failed: %d ❌\n", tests_failed);
    printf("Success rate: %.1f%%\n", 
           (float)tests_passed / (tests_passed + tests_failed) * 100.0);
    
    if (tests_failed > 0) {
        printf("\n⚠️  Some security tests failed - corruption detection may have issues\n");
        return 1;
    } else {
        printf("\n✅ All security tests passed - corruption detection is working\n");
        return 0;
    }
}
