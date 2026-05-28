/*
 * gemini_phon.h — Phonological normalization for the Gemini tokenizer
 *
 * Collapses surface spelling variants that arise from Luganda vowel
 * coalescence rules before the token enters the TST and graph:
 *   ki + agenda → kyagenda
 *   tu + asoma  → twasoma
 *
 * All operations are zero-heap: they work on the caller's buffer in-place.
 */

#ifndef GEMINI_PHON_H
#define GEMINI_PHON_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Apply Luganda phonological coalescence rules in-place.
 *
 * Transforms synthesized hyphenated or space-joined forms like "ki-ogera"
 * into their surface realizations "kyogera".
 *
 * Rules implemented:
 *   u + V → wV  (except u+u → uu)
 *   i + V → yV  (except i+i → ii)
 *   Strip internal hyphens used as morpheme boundaries
 *
 * @param word   NUL-terminated mutable buffer (modified in-place)
 * @param buflen Total buffer capacity (for safety)
 * @return       New string length after normalization
 */
int apply_phonology_rules(char *word, size_t buflen);

/**
 * Normalize a full phrase/sentence in-place, applying phonological rules
 * to each whitespace-delimited token.
 *
 * @param text   Mutable NUL-terminated buffer
 * @param buflen Total buffer capacity
 */
void apply_phonology_rules_sentence(char *text, size_t buflen);

/**
 * Test whether two strings are phonologically equivalent.
 * Normalizes both strings internally and compares.
 * Returns 1 if equivalent, 0 otherwise.
 */
int phon_equivalent(const char *a, const char *b);

#ifdef __cplusplus
}
#endif

#endif /* GEMINI_PHON_H */
