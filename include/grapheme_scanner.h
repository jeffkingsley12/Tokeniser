/*
 * grapheme_scanner.h
 *
 * UAX #29 grapheme cluster scanner with emoji support.
 *
 * Architecture:
 *   UTF-8 bytes → [grapheme_next()] → cluster →
 *       if LUGANDA_LETTER → syllabifier
 *       else → emit atomic token (emoji / symbol / punct)
 *
 * Key principle: the syllabifier never sees partial graphemes.
 *
 * All functions in this header are marked static inline and are safe
 * to include in multiple translation units.
 */

#ifndef GRAPHEME_SCANNER_H
#define GRAPHEME_SCANNER_H

#include "unicode_props.h"
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Additional Luganda-specific property bits (not defined in unicode_props.h) */
#ifndef U_PROP_VOWEL
#define U_PROP_VOWEL  (1u << 6)   /* Luganda vowel nucleus */
#endif
#ifndef U_PROP_PUNCT
#define U_PROP_PUNCT  (1u << 7)   /* Punctuation */
#endif

/* -----------------------------------------------------------------------
 * Grapheme cluster descriptor.
 * Filled by grapheme_next(); callers must not write to it.
 * ----------------------------------------------------------------------- */
typedef struct {
    const uint8_t *ptr;   /* pointer to first byte of cluster in input */
    int            len;   /* byte length of cluster (≥ 1)              */
    uint32_t       first_cp;   /* first Unicode codepoint               */
    uint8_t        props;      /* OR of all U_PROP_* flags in cluster   */
} grapheme_t;

/* -----------------------------------------------------------------------
 * utf8_decode_unsafe — fast UTF-8 decoder for trusted (validated) input.
 *
 * Preconditions:
 *   - p is not NULL and points to a valid UTF-8 sequence.
 *   - The caller has already verified that p + return_value ≤ buffer_end.
 *
 * Returns the number of bytes consumed (1–4) and sets *cp.
 * Does NOT validate continuation bytes; use utf8_decode_safe() for
 * adversarial input.
 * ----------------------------------------------------------------------- */
static inline int utf8_decode_unsafe(const uint8_t *p, uint32_t *cp) {
    uint8_t b0 = p[0];
    if ((b0 & 0x80u) == 0u) {
        *cp = b0;
        return 1;
    }
    if ((b0 & 0xE0u) == 0xC0u) {
        *cp = ((uint32_t)(b0 & 0x1Fu) << 6)
            | (uint32_t)(p[1] & 0x3Fu);
        return 2;
    }
    if ((b0 & 0xF0u) == 0xE0u) {
        *cp = ((uint32_t)(b0 & 0x0Fu) << 12)
            | ((uint32_t)(p[1] & 0x3Fu) << 6)
            |  (uint32_t)(p[2] & 0x3Fu);
        return 3;
    }
    *cp = ((uint32_t)(b0 & 0x07u) << 18)
        | ((uint32_t)(p[1] & 0x3Fu) << 12)
        | ((uint32_t)(p[2] & 0x3Fu) <<  6)
        |  (uint32_t)(p[3] & 0x3Fu);
    return 4;
}

/* Property predicate helpers (operate on the uint8_t prop byte from uprop()) */
static inline bool is_emoji_base(uint8_t prop) {
    return (prop & U_PROP_EMOJI) != 0;
}
static inline bool is_extend(uint8_t prop) {
    return (prop & U_PROP_EXTEND) != 0;
}
static inline bool is_zwj(uint8_t prop) {
    return (prop & U_PROP_ZWJ) != 0;
}
static inline bool is_regional_indicator(uint8_t prop) {
    return (prop & U_PROP_RI) != 0;
}
static inline bool is_letter(uint8_t prop) {
    return (prop & U_PROP_LETTER) != 0;
}
static inline bool is_luganda_vowel(uint8_t prop) {
    return (prop & U_PROP_VOWEL) != 0;
}
static inline bool is_symbol(uint8_t prop) {
    return (prop & U_PROP_SYMBOL) != 0;
}
static inline bool is_punctuation(uint8_t prop) {
    return (prop & U_PROP_PUNCT) != 0;
}
/* Returns true when the cluster should bypass the syllabifier and be
 * emitted as an atomic (opaque) token — emoji and other symbols. */
static inline bool is_atomic_cluster(uint8_t prop) {
    return (prop & (U_PROP_EMOJI | U_PROP_SYMBOL)) != 0;
}

/* -----------------------------------------------------------------------
 * grapheme_next — scan one grapheme cluster from valid UTF-8 input.
 *
 * Implements a simplified UAX #29 rule set:
 *   GB9   Extend chars stay in the current cluster.
 *   GB9a  ZWJ stays with the preceding cluster.
 *   GB11  ZWJ + emoji forms an emoji ZWJ sequence.
 *   GB12/13  Regional Indicator pairs form a single cluster.
 *   Skin-tone modifiers (U+1F3FB..U+1F3FF) attach to a preceding emoji base.
 *
 * Parameters:
 *   p    - pointer to the first byte of the next cluster (must be valid UTF-8)
 *   len  - number of bytes remaining in the buffer starting at p
 *   out  - output descriptor; must not be NULL
 *
 * Returns:
 *   > 0  bytes consumed (written into out->len as well)
 *   == 0 if len ≤ 0 or p is NULL (nothing consumed)
 *   < 0  if the first codepoint overruns [p, p+len) — malformed input
 *
 * Thread safety: pure function, no global state.
 * ----------------------------------------------------------------------- */
static inline int grapheme_next(const uint8_t *p, int len, grapheme_t *out) {
    if (len <= 0 || !p || !out) return 0;

    const uint8_t *start = p;
    const uint8_t *end   = p + len;

    /* Decode the first (base) codepoint */
    uint32_t cp;
    int consumed = utf8_decode_unsafe(p, &cp);
    p += consumed;
    if (p > end) return -1;   /* sequence overruns buffer — malformed */

    uint8_t prop          = uprop(cp);
    uint8_t combined_props = prop;
    out->first_cp = cp;
    out->props    = prop;

    /* Cluster extension state */
    bool prev_is_emoji = is_emoji_base(prop);
    bool prev_is_ri    = is_regional_indicator(prop);
    int  ri_count      = prev_is_ri ? 1 : 0;
    bool prev_is_zwj   = is_zwj(prop);

    while (p < end) {
        uint32_t next_cp;
        int next_consumed = utf8_decode_unsafe(p, &next_cp);
        if (p + next_consumed > end) break;  /* sequence overruns buffer */

        uint8_t next_prop = uprop(next_cp);

        /* GB9: Extend (combining marks, variation selectors) */
        if (is_extend(next_prop)) {
            p              += next_consumed;
            combined_props |= next_prop;
            prev_is_emoji   = false;
            prev_is_ri      = false;
            prev_is_zwj     = false;
            continue;
        }

        /* GB9a: ZWJ stays attached to the preceding cluster */
        if (is_zwj(next_prop)) {
            p              += next_consumed;
            combined_props |= next_prop;
            prev_is_zwj     = true;
            prev_is_emoji   = false;
            prev_is_ri      = false;
            continue;
        }

        /* GB11: emoji ZWJ sequence — ZWJ + emoji base continues cluster */
        if (prev_is_zwj && is_emoji_base(next_prop)) {
            p              += next_consumed;
            combined_props |= next_prop;
            prev_is_emoji   = true;
            prev_is_zwj     = false;
            prev_is_ri      = false;
            continue;
        }

        /* Skin-tone modifiers U+1F3FB..U+1F3FF attach to a preceding emoji */
        if (prev_is_emoji && (next_prop & U_PROP_EMOJI) &&
            next_cp >= 0x1F3FBu && next_cp <= 0x1F3FFu) {
            p              += next_consumed;
            combined_props |= next_prop;
            prev_is_emoji   = true;
            prev_is_ri      = false;
            prev_is_zwj     = false;
            continue;
        }

        /* GB12/13: Regional Indicator pairs (flag emoji).
         * Only the second RI in each pair extends the cluster; a third RI
         * starts a new cluster (ri_count % 2 == 0 at that point). */
        if (prev_is_ri && is_regional_indicator(next_prop) &&
            ri_count % 2 == 1) {
            p              += next_consumed;
            combined_props |= next_prop;
            ri_count++;
            prev_is_ri      = true;
            prev_is_emoji   = false;
            prev_is_zwj     = false;
            continue;
        }

        /* All other cases: break before the next grapheme cluster */
        break;
    }

    out->ptr   = start;
    out->len   = (int)(p - start);
    out->props = combined_props;
    return out->len;
}

/* -----------------------------------------------------------------------
 * Convenience predicates on grapheme_t
 * ----------------------------------------------------------------------- */

/* True when the grapheme's first codepoint is a letter (any script). */
static inline bool grapheme_is_luganda_letter(const grapheme_t *g) {
    return g && is_letter(g->props);
}

/* True for a single-byte ASCII whitespace character.
 * Multi-byte whitespace (e.g. U+00A0 NBSP) is intentionally excluded —
 * the syllabifier handles it as a non-letter byte. */
static inline bool grapheme_is_whitespace(const grapheme_t *g) {
    return g && g->len == 1 &&
           (g->ptr[0] == ' '  || g->ptr[0] == '\t' ||
            g->ptr[0] == '\n' || g->ptr[0] == '\r');
}

/* True when the grapheme is punctuation (either by Unicode property or
 * by being a common ASCII punctuation character). */
static inline bool grapheme_is_punctuation(const grapheme_t *g) {
    if (!g) return false;
    if (is_punctuation(g->props)) return true;
    return g->len == 1 && (g->ptr[0] == '.' || g->ptr[0] == ',' ||
                           g->ptr[0] == '!' || g->ptr[0] == '?');
}

#ifdef __cplusplus
}
#endif

#endif /* GRAPHEME_SCANNER_H */
