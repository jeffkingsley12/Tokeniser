#define _POSIX_C_SOURCE 200809L
#include "tokenizer.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#define LINE_BUF_SIZE 16384

typedef struct {
    char **docs;
    uint32_t n_docs;
} Corpus;

static Corpus *load_corpus(const char *path, size_t max_bytes) {
    FILE *f = fopen(path, "r");
    if (!f) return NULL;
    Corpus *c = calloc(1, sizeof(Corpus));
    if (!c) { fclose(f); return NULL; }
    uint32_t cap = 1024;
    c->docs = malloc(cap * sizeof(char *));
    if (!c->docs) { free(c); fclose(f); return NULL; }
    char line[LINE_BUF_SIZE];
    size_t total_bytes = 0;
    while (fgets(line, sizeof(line), f)) {
        size_t len = strlen(line);
        if (len > 0 && line[len-1] == '\n') line[--len] = '\0';
        if (len == 0) continue;
        if (c->n_docs == cap) {
            cap *= 2;
            char **tmp = realloc(c->docs, cap * sizeof(char *));
            if (!tmp) {
                fprintf(stderr, "OOM during realloc\n");
                for (uint32_t i = 0; i < c->n_docs; i++) free(c->docs[i]);
                free(c->docs);
                free(c);
                fclose(f);
                return NULL;
            }
            c->docs = tmp;
        }
        c->docs[c->n_docs++] = strdup(line);
        total_bytes += len;
        if (total_bytes >= max_bytes) break;
    }
    fclose(f);
    return c;
}

static void free_corpus(Corpus *c) {
    for (uint32_t i = 0; i < c->n_docs; i++) free(c->docs[i]);
    free(c->docs);
    free(c);
}

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <corpus.txt>\n", argv[0]);
        return 1;
    }
    // Test on 5MB to be fast
    Corpus *c = load_corpus(argv[1], 5 * 1024 * 1024);
    if (!c) {
        fprintf(stderr, "Failed to load corpus\n");
        return 1;
    }

    printf("Building Model A...\n");
    Tokenizer *tokA = tokenizer_build((const char **)c->docs, c->n_docs);
    if (!tokA) {
        fprintf(stderr, "tokenizer_build A failed\n");
        free_corpus(c);
        return 1;
    }
    tokenizer_save(tokA, "modelA.bin");
    
    printf("Building Model B...\n");
    Tokenizer *tokB = tokenizer_build((const char **)c->docs, c->n_docs);
    if (!tokB) {
        fprintf(stderr, "tokenizer_build B failed\n");
        tokenizer_destroy(tokA);
        free_corpus(c);
        return 1;
    }
    tokenizer_save(tokB, "modelB.bin");

    tokenizer_destroy(tokA);
    tokenizer_destroy(tokB);
    free_corpus(c);

    printf("Comparing modelA.bin and modelB.bin...\n");
    FILE *fA = fopen("modelA.bin", "rb");
    FILE *fB = fopen("modelB.bin", "rb");
    if (!fA || !fB) {
        fprintf(stderr, "Failed to open output models\n");
        return 1;
    }

    fseek(fA, 0, SEEK_END);
    size_t sizeA = ftell(fA);
    fseek(fA, 0, SEEK_SET);

    fseek(fB, 0, SEEK_END);
    size_t sizeB = ftell(fB);
    fseek(fB, 0, SEEK_SET);

    if (sizeA != sizeB) {
        fprintf(stderr, "Determinism failed: sizeA (%zu) != sizeB (%zu)\n", sizeA, sizeB);
        return 1;
    }

    uint8_t *bufA = malloc(sizeA);
    uint8_t *bufB = malloc(sizeB);
    fread(bufA, 1, sizeA, fA);
    fread(bufB, 1, sizeB, fB);

    if (memcmp(bufA, bufB, sizeA) != 0) {
        fprintf(stderr, "Determinism failed: Binary content differs!\n");
        return 1;
    }

    printf("Determinism PASSED! Files are bit-for-bit identical.\n");
    free(bufA); free(bufB);
    fclose(fA); fclose(fB);
    unlink("modelA.bin");
    unlink("modelB.bin");
    return 0;
}
