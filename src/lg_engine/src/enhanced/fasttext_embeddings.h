/**
 * fasttext_embeddings.h — Stub for FastText embedding model
 * 
 * This is a minimal stub to allow compilation when the FastText embedding
 * model is not available. The actual implementation can be linked in later.
 */

#ifndef FASTTEXT_EMBEDDINGS_H
#define FASTTEXT_EMBEDDINGS_H

#include <stdint.h>

typedef struct EmbeddingModel {
    void *model;  /* Opaque pointer to FastText model */
} EmbeddingModel;

/**
 * Compute cosine similarity between two words using the embedding model.
 * Returns 0.0 if model is NULL or if either word is OOV.
 */
static inline float embedding_semantic_score(EmbeddingModel *embeddings, 
                                              const char *word_a, 
                                              const char *word_b) {
    (void)embeddings;
    (void)word_a;
    (void)word_b;
    return 0.0f;  /* Stub: no embeddings available */
}

#endif /* FASTTEXT_EMBEDDINGS_H */
