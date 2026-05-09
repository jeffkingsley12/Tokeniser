/*
 * Minimal test to identify specific corruption detection issues
 */

#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/stat.h>
#include "tokenizer.h"

int main() {
    printf("🔍 Minimal Corruption Detection Test\n");
    printf("====================================\n");
    
    /* Create a simple tokenizer */
    const char* docs[] = {"oli otya", "webale nnyo"};
    Tokenizer* orig = tokenizer_build(docs, 2);
    if (!orig) {
        printf("❌ Failed to create tokenizer\n");
        return 1;
    }
    printf("✅ Created test tokenizer\n");
    
    /* Save the tokenizer */
    const char* model_path = "minimal_test.bin";
    if (tokenizer_save(orig, model_path) != 0) {
        printf("❌ Failed to save tokenizer\n");
        tokenizer_destroy(orig);
        return 1;
    }
    printf("✅ Saved tokenizer\n");
    
    /* Test 1: Normal load */
    Tokenizer* loaded = tokenizer_load(model_path);
    if (loaded) {
        printf("✅ Normal load works\n");
        tokenizer_destroy(loaded);
    } else {
        printf("❌ Normal load failed\n");
        tokenizer_destroy(orig);
        unlink(model_path);
        return 1;
    }
    
    /* Test 2: Corrupt magic number */
    FILE* f = fopen(model_path, "rb+");
    if (f) {
        const char* bad_magic = "CORRUPT13";
        fwrite(bad_magic, 1, 8, f);
        fclose(f);
        
        loaded = tokenizer_load(model_path);
        if (loaded == NULL) {
            printf("✅ Magic corruption detected\n");
        } else {
            printf("❌ Magic corruption NOT detected\n");
            tokenizer_destroy(loaded);
        }
    }
    
    /* Restore original file */
    tokenizer_save(orig, model_path);
    
    /* Test 3: Corrupt single byte at position 20 */
    f = fopen(model_path, "rb+");
    if (f) {
        fseek(f, 20, SEEK_SET);
        uint8_t bad_byte = 0xFF;
        fwrite(&bad_byte, 1, 1, f);
        fclose(f);
        
        loaded = tokenizer_load(model_path);
        if (loaded == NULL) {
            printf("✅ Byte corruption detected\n");
        } else {
            printf("❌ Byte corruption NOT detected\n");
            tokenizer_destroy(loaded);
        }
    }
    
    /* Test 4: Truncate file */
    struct stat st;
    if (stat(model_path, &st) == 0) {
        truncate(model_path, st.st_size / 2);
        loaded = tokenizer_load(model_path);
        if (loaded == NULL) {
            printf("✅ Truncation detected\n");
        } else {
            printf("❌ Truncation NOT detected\n");
            tokenizer_destroy(loaded);
        }
    }
    
    tokenizer_destroy(orig);
    unlink(model_path);
    
    printf("\n🏁 Minimal corruption test completed\n");
    return 0;
}
