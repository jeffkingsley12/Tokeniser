#include <jni.h>
#include <android/log.h>
#include <android/asset_manager.h>
#include <android/asset_manager_jni.h>
#include <stdlib.h>
#include <string.h>
#include <inttypes.h>
#include "tokenizer.h"

#define LOG_TAG "NativeTokenizer"
#define LOGD(...) __android_log_print(ANDROID_LOG_DEBUG, LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)

/*
 * Helper to throw a Java exception.
 * Must not be called if an exception is already pending.
 */
static void throw_exception(JNIEnv *env, const char *msg) {
    if ((*env)->ExceptionCheck(env)) return;   /* already pending */
    jclass cls = (*env)->FindClass(env, "java/lang/RuntimeException");
    if (cls) {
        (*env)->ThrowNew(env, cls, msg);
        (*env)->DeleteLocalRef(env, cls);
    }
}

JNIEXPORT jlong JNICALL
Java_com_luganda_tokenizer_NativeTokenizer_nativeInit(JNIEnv *env, jobject thiz,
                                                      jstring path) {
    if (!path) return 0;

    const char *c_path = (*env)->GetStringUTFChars(env, path, NULL);
    if (!c_path) return 0;

    MmapTokenizer *mt = tokenizer_load_mmap(c_path);
    if (!mt) {
        LOGE("Failed to load model from %s", c_path);
        (*env)->ReleaseStringUTFChars(env, path, c_path);
        return 0;
    }

    LOGD("Loaded model successfully from %s", c_path);
    (*env)->ReleaseStringUTFChars(env, path, c_path);
    return (jlong)(uintptr_t)mt;
}

/*
 * Load model directly from an Android asset (zero-copy path).
 * Ownership of the AAsset is transferred to MmapTokenizer::asset_handle
 * and will be closed during nativeDestroy.
 */
JNIEXPORT jlong JNICALL
Java_com_luganda_tokenizer_NativeTokenizer_nativeInitFromAsset(JNIEnv *env, jobject thiz,
                                                               jobject assetMgr,
                                                               jstring assetName) {
    if (!assetMgr || !assetName) return 0;

    AAssetManager *mgr = AAssetManager_fromJava(env, assetMgr);
    if (!mgr) return 0;

    const char *c_name = (*env)->GetStringUTFChars(env, assetName, NULL);
    if (!c_name) return 0;

    AAsset *asset = AAssetManager_open(mgr, c_name, AASSET_MODE_BUFFER);
    if (!asset) {
        LOGE("Failed to open asset: %s", c_name);
        (*env)->ReleaseStringUTFChars(env, assetName, c_name);
        return 0;
    }

    /* AAsset_getLength() returns off_t (signed). -1 signals error; 0
     * means empty — both are fatal.  Also guard against SIZE_MAX overflow
     * when casting to size_t on platforms where off_t > size_t. */
    off_t size = AAsset_getLength(asset);
    if (size <= 0 || (uintmax_t)size > (uintmax_t)SIZE_MAX) {
        LOGE("Invalid asset size (%" PRId64 ") for %s", (int64_t)size, c_name);
        AAsset_close(asset);
        (*env)->ReleaseStringUTFChars(env, assetName, c_name);
        return 0;
    }

    const void *buffer = AAsset_getBuffer(asset);
    if (!buffer) {
        LOGE("Failed to get buffer for asset: %s", c_name);
        AAsset_close(asset);
        (*env)->ReleaseStringUTFChars(env, assetName, c_name);
        return 0;
    }

    MmapTokenizer *mt = tokenizer_load_mmap_from_buffer(buffer, (size_t)size);
    if (!mt) {
        LOGE("Failed to load model from asset buffer");
        AAsset_close(asset);
        (*env)->ReleaseStringUTFChars(env, assetName, c_name);
        return 0;
    }

    /* Zero-copy loading: do not close the asset here.
     * Transfer ownership to the tokenizer; nativeDestroy will close it. */
    mt->fd           = -1;
    mt->asset_handle = asset;

    LOGD("Loaded model from asset: %s", c_name);
    (*env)->ReleaseStringUTFChars(env, assetName, c_name);
    return (jlong)(uintptr_t)mt;
}

JNIEXPORT jintArray JNICALL
Java_com_luganda_tokenizer_NativeTokenizer_nativeEncode(JNIEnv *env, jobject thiz,
                                                        jlong handle, jstring text) {
    if (!handle || !text) return NULL;

    /* Isolate local references created during this call. */
    if ((*env)->PushLocalFrame(env, 16) != JNI_OK)
        return NULL;

    MmapTokenizer *mt = (MmapTokenizer *)(uintptr_t)handle;

    const char *c_text = (*env)->GetStringUTFChars(env, text, NULL);
    if (!c_text) {
        (*env)->PopLocalFrame(env, NULL);
        return NULL;
    }

    size_t text_len = strlen(c_text);

    /* Empty string: return a zero-length array rather than allocating
     * a zero-byte buffer (malloc(0) behaviour is implementation-defined). */
    if (text_len == 0) {
        (*env)->ReleaseStringUTFChars(env, text, c_text);
        jintArray empty = (*env)->NewIntArray(env, 0);
        return (jintArray)(*env)->PopLocalFrame(env, (jobject)empty);
    }

    /* Worst-case: every byte is its own token. */
    uint32_t *out_buf = malloc(text_len * sizeof(uint32_t));
    if (!out_buf) {
        (*env)->ReleaseStringUTFChars(env, text, c_text);
        throw_exception(env, "Out of memory");
        (*env)->PopLocalFrame(env, NULL);
        return NULL;
    }

    int n_tokens = tokenizer_encode_fused(&mt->base, c_text, out_buf,
                                          (uint32_t)text_len);
    (*env)->ReleaseStringUTFChars(env, text, c_text);

    if (n_tokens < 0) {
        free(out_buf);
        (*env)->PopLocalFrame(env, NULL);
        return NULL;
    }

    jintArray result = (*env)->NewIntArray(env, n_tokens);
    if (result) {
        /* jint is int32_t (JNI spec §12.4); uint32_t has the same width.
         * Token IDs must be ≤ INT32_MAX — the tokenizer guarantees this. */
        (*env)->SetIntArrayRegion(env, result, 0, n_tokens,
                                  (const jint *)out_buf);
    }

    free(out_buf);
    return (jintArray)(*env)->PopLocalFrame(env, (jobject)result);
}

JNIEXPORT void JNICALL
Java_com_luganda_tokenizer_NativeTokenizer_nativeDestroy(JNIEnv *env, jobject thiz,
                                                         jlong handle) {
    if (!handle) return;

    MmapTokenizer *mt = (MmapTokenizer *)(uintptr_t)handle;

    /* Close the Android asset BEFORE unmapping memory; the unmap in
     * tokenizer_destroy_mmap would otherwise pull the rug out from
     * under the asset's internal buffer on some NDK versions. */
    if (mt->asset_handle) {
        AAsset_close((AAsset *)mt->asset_handle);
        mt->asset_handle = NULL;
    }

    tokenizer_destroy_mmap(mt);
}
