#include "tokenizer_api.h"
#include "tokenizer.h"
#include <stdio.h>

/**
 * Implementation of the flat C API for FFI.
 * Manages a small pool of active tokenizer instances.
 */

#include <pthread.h>

#define MAX_HANDLES 16
static MmapTokenizer *g_handles[MAX_HANDLES] = {0};
static pthread_mutex_t g_lock = PTHREAD_MUTEX_INITIALIZER;

int tok_load(const char *path) {
    if (!path) return -1;
    
    pthread_mutex_lock(&g_lock);
    int handle = -1;
    for (int i = 0; i < MAX_HANDLES; i++) {
        if (!g_handles[i]) {
            MmapTokenizer *mt = tokenizer_load_mmap(path);
            if (mt) {
                g_handles[i] = mt;
                handle = i;
            }
            break;
        }
    }
    if (handle == -1) {
        fprintf(stderr, "[tok_api] Failed to load tokenizer or maximum handle count reached (%d)\n", MAX_HANDLES);
    }
    pthread_mutex_unlock(&g_lock);
    return handle;
}

int tok_encode(int handle, const char *text, uint32_t *out, int cap) {
    if (handle < 0 || handle >= MAX_HANDLES || !g_handles[handle]) return -1;
    if (!text || !out || cap <= 0) return -1;
    
    return tokenizer_encode(&g_handles[handle]->base, text, out, (uint32_t)cap);
}

const char *tok_decode(int handle, uint32_t id) {
    if (handle < 0 || handle >= MAX_HANDLES || !g_handles[handle]) return NULL;
    return tokenizer_decode(&g_handles[handle]->base, id);
}

void tok_free(int handle) {
    pthread_mutex_lock(&g_lock);
    if (handle >= 0 && handle < MAX_HANDLES && g_handles[handle]) {
        tokenizer_destroy_mmap(g_handles[handle]);
        g_handles[handle] = NULL;
    }
    pthread_mutex_unlock(&g_lock);
}
