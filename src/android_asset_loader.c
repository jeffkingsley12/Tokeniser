#include "tokenizer.h"
#include <android/asset_manager.h>
#include <android/asset_manager_jni.h>
#include <jni.h>
#include <stdint.h>
#include <inttypes.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    Tokenizer *tokenizer;
    AAsset *asset;
    const void *buffer;
    size_t length;
} AndroidTokenizerContext;

/*
 * Initialize tokenizer from an Android asset.
 * Uses zero-copy mapping (AASSET_MODE_BUFFER).
 */
JNIEXPORT jlong JNICALL
Java_com_example_nlp_Tokenizer_nativeInitFromAsset(JNIEnv *env, jobject thiz,
                                                   jobject asset_manager,
                                                   jstring filename) {
    if (!env || !asset_manager || !filename) return 0;

    const char *utf8_file = (*env)->GetStringUTFChars(env, filename, NULL);
    if (!utf8_file) return 0;

    AAssetManager *mgr = AAssetManager_fromJava(env, asset_manager);
    if (!mgr) {
        (*env)->ReleaseStringUTFChars(env, filename, utf8_file);
        return 0;
    }

    AAsset *asset = AAssetManager_open(mgr, utf8_file, AASSET_MODE_BUFFER);
    (*env)->ReleaseStringUTFChars(env, filename, utf8_file);

    if (!asset) return 0;

    /* Validate asset length before cast to size_t.
     * AAsset_getLength() returns off_t which is signed; -1 signals error.
     * A negative or zero value must be rejected before the (size_t) cast,
     * because casting -1 to size_t yields SIZE_MAX. */
    off_t length_signed = AAsset_getLength(asset);
    if (length_signed <= 0 || (uintmax_t)length_signed > (uintmax_t)SIZE_MAX) {
        AAsset_close(asset);
        return 0;
    }
    size_t length = (size_t)length_signed;

    /* Get direct pointer to memory-mapped asset data (zero-copy) */
    const void *buffer = AAsset_getBuffer(asset);
    if (!buffer) {
        AAsset_close(asset);
        return 0;
    }

    /* Initialize tokenizer directly from memory */
    Tokenizer *t = tokenizer_create();
    if (!t) {
        AAsset_close(asset);
        return 0;
    }

    /* Load model from memory buffer */
    if (tokenizer_load_from_memory(t, buffer, length) != 0) {
        tokenizer_destroy(t);
        AAsset_close(asset);
        return 0;
    }

    /* Create context to keep asset reference */
    AndroidTokenizerContext *ctx = calloc(1, sizeof(AndroidTokenizerContext));
    if (!ctx) {
        tokenizer_destroy(t);
        AAsset_close(asset);
        return 0;
    }

    ctx->tokenizer = t;
    ctx->asset = asset;  /* Keep open - buffer remains mapped */
    ctx->buffer = buffer;
    ctx->length = length;

    return (jlong)(uintptr_t)ctx;
}

/*
 * Tokenize a string using the provided context.
 */
JNIEXPORT jintArray JNICALL
Java_com_example_nlp_Tokenizer_nativeTokenize(JNIEnv *env, jobject thiz,
                                             jlong context_handle,
                                             jstring text) {
    if (!env || !context_handle || !text) return NULL;

    AndroidTokenizerContext *ctx = (AndroidTokenizerContext *)(uintptr_t)context_handle;
    if (!ctx->tokenizer) return NULL;

    const char *utf8_text = (*env)->GetStringUTFChars(env, text, NULL);
    if (!utf8_text) return NULL;

    /* Tokenize */
    uint32_t tokens[MAX_SYLLABLES];
    int token_count = tokenizer_encode(ctx->tokenizer, utf8_text, tokens, MAX_SYLLABLES);

    (*env)->ReleaseStringUTFChars(env, text, utf8_text);

    if (token_count <= 0) return NULL;

    /* Create Java int array */
    jintArray result = (*env)->NewIntArray(env, token_count);
    if (!result) return NULL;

    /* Copy tokens to Java array.
     * jint is int32_t (JNI spec §12.4), uint32_t is the same width.
     * Token IDs must be in range [0, INT32_MAX]; values above that would
     * silently reinterpret. The tokenizer must guarantee this invariant. */
    (*env)->SetIntArrayRegion(env, result, 0, token_count, (const jint *)tokens);

    return result;
}

/*
 * Destroy context and close asset.
 */
JNIEXPORT void JNICALL
Java_com_example_nlp_Tokenizer_nativeDestroy(JNIEnv *env, jobject thiz,
                                            jlong context_handle) {
    if (!context_handle) return;

    AndroidTokenizerContext *ctx = (AndroidTokenizerContext *)(uintptr_t)context_handle;

    if (ctx->tokenizer) {
        tokenizer_destroy(ctx->tokenizer);
        ctx->tokenizer = NULL;
    }

    if (ctx->asset) {
        AAsset_close(ctx->asset);  /* This unmaps the buffer */
        ctx->asset = NULL;
    }

    free(ctx);
}

/*
 * Get basic info about the tokenizer.
 */
JNIEXPORT jstring JNICALL
Java_com_example_nlp_Tokenizer_nativeGetInfo(JNIEnv *env, jobject thiz,
                                            jlong context_handle) {
    if (!env || !context_handle) return NULL;

    AndroidTokenizerContext *ctx = (AndroidTokenizerContext *)(uintptr_t)context_handle;
    if (!ctx->tokenizer) return NULL;

    /* Guard against NULL sub-pointers before dereferencing */
    if (!ctx->tokenizer->stbl || !ctx->tokenizer->grammar) return NULL;

    char info[256];
    snprintf(info, sizeof(info),
             "Tokenizer: %zu syllables, %u tokens, zero-copy mapped (%zu bytes)",
             ctx->tokenizer->stbl->n_syls,
             ctx->tokenizer->grammar->n_rules,
             ctx->length);

    return (*env)->NewStringUTF(env, info);
}
