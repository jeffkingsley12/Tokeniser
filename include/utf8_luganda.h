#ifndef UTF8_LUGANDA_H
#define UTF8_LUGANDA_H
#pragma once

/*
 * utf8_luganda.h
 *
 * Self-contained UTF-8 decoder and Luganda phonological character classifier.
 *
 * This header is the ONLY place in the codebase that hard-codes Unicode
 * codepoint values for Luganda characters.  The syllabifier reads these
 * functions; all other C files get character-class information indirectly
 * through the syllabifier API.
 *
 * Luganda orthography uses:
 *
 *   Plain ASCII vowels         a  e  i  o  u         (U+0061 … U+0075)
 *   Long vowels                aa ee ii oo uu         (two plain vowels)
 *   Velar nasal                ŋ                      (U+014B)
 *   Palatal nasal digraph      ny                     (two ASCII chars)
 *   Geminate nasals            nn  mm  ŋŋ             (doubled)
 *   Semivowels                 w  y                   (U+0077, U+0079)
 *
 * High-tone marks (scholarly / dictionary text only):
 *   Toned short vowels         á  é  í  ó  ú         (U+00E1 … U+00FA)
 *   Toned long vowels          áa éé íi óo úu        (toned + plain)
 *
 * Confusable characters that must NOT appear in training data:
 *   Cyrillic о (U+043E) looks like Latin o — always replace with Latin o
 *   Cyrillic а (U+0430) looks like Latin a — always replace with Latin a
 *   Greek ο (U+03BF) looks like Latin o — always replace with Latin o
 *   (syllable_table_generator.py normalises these before C code sees them)
 *
 * Thread safety: all functions are pure (no global state).
 */

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>

/* =========================================================
 *  UTF-8 byte-level utilities
 * ========================================================= */

/*
 * utf8_char_len — byte length of the UTF-8 sequence starting at `p`.
 *
 * Returns 1–4 on a valid lead byte.
 * Returns 1 on an invalid or continuation byte (treats it as a raw byte
 * and advances by 1 so the caller always makes forward progress).
 *
 * `p` must not be NULL; the caller is responsible for bounds checking
 * before passing a pointer that may be at end-of-buffer.
 */
static inline int utf8_char_len(const uint8_t *p)
{
    uint8_t b = *p;
    if ((b & 0x80u) == 0x00u) return 1;   /* 0xxxxxxx — ASCII */
    if ((b & 0xE0u) == 0xC0u) return 2;   /* 110xxxxx — 2-byte */
    if ((b & 0xF0u) == 0xE0u) return 3;   /* 1110xxxx — 3-byte */
    if ((b & 0xF8u) == 0xF0u) return 4;   /* 11110xxx — 4-byte */
    return 1;   /* continuation byte or invalid — advance by 1 */
}

/*
 * utf8_decode — decode one UTF-8 codepoint.
 *
 * `p`   : pointer to the first byte of the sequence.
 * `len` : byte length as returned by utf8_char_len(p).
 *
 * Returns the Unicode codepoint.
 * Returns U+FFFD (REPLACEMENT CHARACTER) for invalid sequences.
 *
 * This function does NOT validate continuation bytes; it trusts that the
 * input was produced by a well-formed UTF-8 source (the YAML generator
 * already validates this).  For adversarial input use utf8_decode_safe().
 *
 * WARNING: Caller is responsible for ensuring p + len does not exceed
 * buffer bounds.  Use utf8_decode_bounded() for untrusted input.
 */
static inline uint32_t __attribute__((unused)) utf8_decode(const uint8_t *p, int len)
{
    switch (len) {
    case 1:
        return (uint32_t)p[0];
    case 2:
        return ((uint32_t)(p[0] & 0x1Fu) << 6)
             |  (uint32_t)(p[1] & 0x3Fu);
    case 3:
        return ((uint32_t)(p[0] & 0x0Fu) << 12)
             | ((uint32_t)(p[1] & 0x3Fu) <<  6)
             |  (uint32_t)(p[2] & 0x3Fu);
    case 4:
        return ((uint32_t)(p[0] & 0x07u) << 18)
             | ((uint32_t)(p[1] & 0x3Fu) << 12)
             | ((uint32_t)(p[2] & 0x3Fu) <<  6)
             |  (uint32_t)(p[3] & 0x3Fu);
    default:
        return 0xFFFDu;   /* U+FFFD REPLACEMENT CHARACTER */
    }
}

/*
 * utf8_decode_safe — like utf8_decode but validates continuation bytes.
 *
 * Returns U+FFFD if any continuation byte is malformed.
 * Use this when processing untrusted input (e.g. user-supplied text
 * that was not processed by the YAML generator).
 */
static inline uint32_t __attribute__((unused)) utf8_decode_safe(const uint8_t *p, int len)
{
    for (int i = 1; i < len; i++) {
        if ((p[i] & 0xC0u) != 0x80u) return 0xFFFDu;
    }
    return utf8_decode(p, len);
}

/*
 * utf8_decode_bounded — safe decoder with buffer bounds checking.
 *
 * `p`   : pointer to the first byte of the sequence.
 * `end` : pointer to one byte past the last valid byte of the buffer.
 * `len` : byte length as returned by utf8_char_len(p).
 *
 * Returns the Unicode codepoint if the sequence is fully within [p, end).
 * Returns U+FFFD (REPLACEMENT CHARACTER) if the sequence would exceed bounds
 * or contains invalid continuation bytes.
 *
 * Use this for parsing untrusted input that may be malformed or truncated.
 */
static inline uint32_t __attribute__((unused)) utf8_decode_bounded(
    const uint8_t *p, const uint8_t *end, int len)
{
    if (p + len > end) return 0xFFFDu;  /* would exceed buffer */
    return utf8_decode_safe(p, len);
}

/*
 * utf8_next — advance `*pp` by one codepoint and return it.
 *
 * Typical usage:
 *
 *   const uint8_t *p = (const uint8_t *)text;
 *   while (*p) {
 *       uint32_t cp = utf8_next(&p);
 *       // process cp
 *   }
 *
 * `*pp` is advanced past the consumed bytes.
 * `*pp` must point to a valid, NUL-terminated UTF-8 string.
 */
static inline uint32_t __attribute__((unused)) utf8_next(const uint8_t **pp)
{
    int len = utf8_char_len(*pp);
    uint32_t cp = utf8_decode(*pp, len);
    *pp += len;
    return cp;
}

/*
 * utf8_strlen — number of codepoints in a NUL-terminated UTF-8 string.
 * O(n) in the byte length of the string.
 */
static inline size_t __attribute__((unused)) utf8_strlen(const char *s)
{
    const uint8_t *p = (const uint8_t *)s;
    size_t count = 0;
    while (*p) {
        p += utf8_char_len(p);
        count++;
    }
    return count;
}

/* =========================================================
 *  Luganda-specific phonological character classification
 *
 *  All functions take a Unicode codepoint (uint32_t) as returned by
 *  utf8_decode() / utf8_next().  Never pass raw bytes to these functions.
 *
 *  The classifications below follow the Luganda orthographic standard
 *  used in the Namirembe Diocese publications and the ULCA dictionary.
 * ========================================================= */

/*
 * lug_is_vowel — true for any Luganda vowel nucleus codepoint.
 *
 * Covers both plain vowels (a e i o u) and their tone-marked equivalents
 * (á é í ó ú — acute accent = high tone, used in scholarly text).
 * Both short and long vowel nuclei use the same codepoints; "long vowel"
 * is represented as two consecutive vowel codepoints (e.g. "aa" = a + a).
 *
 * Does NOT include Cyrillic/Greek look-alikes — those must be normalised
 * by the preprocessor (syllable_table_generator.py) before reaching here.
 */
static inline bool __attribute__((unused)) lug_is_vowel(uint32_t cp)
{
    /* Fast path: ASCII range */
    if (cp < 0x80u) {
        switch ((uint8_t)cp) {
        case 'a': case 'e': case 'i': case 'o': case 'u':
        case 'A': case 'E': case 'I': case 'O': case 'U':
            return true;
        default:
            return false;
        }
    }
    /* High-tone (acute accent) — scholarly / dictionary orthography */
    switch (cp) {
    case 0x00E1u: /* á */ case 0x00E9u: /* é */ case 0x00EDu: /* í */
    case 0x00F3u: /* ó */ case 0x00FAu: /* ú */
        return true;
    default:
        return false;
    }
}

static inline uint32_t __attribute__((unused)) lug_vowel_base(uint32_t cp)
{
    switch (cp) {
    case 0x00E1u: return 0x0061u; /* á → a */
    case 0x00E9u: return 0x0065u; /* é → e */
    case 0x00EDu: return 0x0069u; /* í → i */
    case 0x00F3u: return 0x006Fu; /* ó → o */
    case 0x00FAu: return 0x0075u; /* ú → u */
    default:      return cp;
    }
}

static inline bool __attribute__((unused)) lug_is_toned(uint32_t cp)
{
    return cp == 0x00E1u || cp == 0x00E9u || cp == 0x00EDu
        || cp == 0x00F3u || cp == 0x00FAu;
}

static inline bool __attribute__((unused)) lug_is_nasal(uint32_t cp)
{
    if (cp < 0x80u) {
        switch ((uint8_t)cp) {
        case 'm': case 'n':
        case 'M': case 'N':
            return true;
        default:
            return false;
        }
    }
    return cp == 0x014Bu; /* ŋ */
}

static inline bool __attribute__((unused)) lug_is_semivowel(uint32_t cp)
{
    if (cp < 0x80u) {
        switch ((uint8_t)cp) {
        case 'w': case 'y':
        case 'W': case 'Y':
            return true;
        default:
            return false;
        }
    }
    return false;
}

static inline bool __attribute__((unused)) lug_is_consonant(uint32_t cp)
{
    /* Fast path: ASCII range */
    if (cp < 0x80u) {
        switch ((uint8_t)cp) {
        case 'b': case 'c': case 'd': case 'f': case 'g': case 'h':
        case 'j': case 'k': case 'l': case 'm': case 'n': case 'p':
        case 'q': case 'r': case 's': case 't': case 'v': case 'w':
        case 'x': case 'y': case 'z':
        case 'B': case 'C': case 'D': case 'F': case 'G': case 'H':
        case 'J': case 'K': case 'L': case 'M': case 'N': case 'P':
        case 'Q': case 'R': case 'S': case 'T': case 'V': case 'W':
        case 'X': case 'Y': case 'Z':
            return true;
        default:
            return false;
        }
    }
    /* Non-ASCII Luganda consonant: ŋ only */
    return cp == 0x014Bu; /* ŋ */
}

static inline bool __attribute__((unused)) lug_is_luganda_consonant(uint32_t cp)
{
    /* Fast path: ASCII range, excluding q and x */
    if (cp < 0x80u) {
        switch ((uint8_t)cp) {
        case 'b': case 'c': case 'd': case 'f': case 'g': case 'h':
        case 'j': case 'k': case 'l': case 'm': case 'n': case 'p':
        case 'r': case 's': case 't': case 'v': case 'w':
        case 'y': case 'z':
        case 'B': case 'C': case 'D': case 'F': case 'G': case 'H':
        case 'J': case 'K': case 'L': case 'M': case 'N': case 'P':
        case 'R': case 'S': case 'T': case 'V': case 'W':
        case 'Y': case 'Z':
            return true;
        default:
            return false;  /* excludes q, x */
        }
    }
    /* Non-ASCII Luganda consonant: ŋ only */
    return cp == 0x014Bu; /* ŋ */
}

/*
 * lug_is_eng — true only for ŋ (U+014B), the velar nasal.
 *
 * Provided as a named predicate because ŋ has special combinatorial
 * rules in the phonotactics (it forms NV syllables alone, unlike m/n
 * which also form NCV clusters).
 */
static inline bool __attribute__((unused)) lug_is_eng(uint32_t cp)
{
    return cp == 0x014Bu;
}

/*
 * lug_is_geminate_lead — true when codepoint `a` followed by `b`
 * constitutes a geminate (doubled) consonant onset.
 *
 * Both ASCII geminates (bb, cc, dd …) and the ŋŋ geminate are handled.
 * The digraph "ny" doubling is "nny" (n + ny), handled separately.
 */
static inline bool __attribute__((unused)) lug_is_geminate(uint32_t a, uint32_t b)
{
    /* ŋŋ geminate */
    if (a == 0x014Bu && b == 0x014Bu) return true;
    /* ASCII geminates: same consonant repeated */
    if (a == b && a < 0x80u && lug_is_consonant(a)) return true;
    return false;
}

/*
 * lug_lowercase — fold a Luganda codepoint to lowercase.
 *
 * Handles ASCII A-Z, Ŋ (U+014A) → ŋ (U+014B), and the toned capitals
 * (Á É Í Ó Ú → á é í ó ú).
 * Used for normalisation; does NOT handle general Unicode case folding.
 *
 * NOTE: Grave-accented uppercase vowels (À È Ì Ò Ù) are not part of
 * standard Luganda orthography and are not handled. If your input may
 * contain them, extend this function's switch statement.
 */
static inline uint32_t __attribute__((unused)) lug_lowercase(uint32_t cp)
{
    /* ASCII A-Z */
    if (cp >= 0x0041u && cp <= 0x005Au) return cp + 0x0020u;
    /* Ŋ → ŋ */
    if (cp == 0x014Au) return 0x014Bu;
    /* Toned uppercase: Á É Í Ó Ú */
    if (cp == 0x00C1u) return 0x00E1u; /* Á → á */
    if (cp == 0x00C9u) return 0x00E9u; /* É → é */
    if (cp == 0x00CDu) return 0x00EDu; /* Í → í */
    if (cp == 0x00D3u) return 0x00F3u; /* Ó → ó */
    if (cp == 0x00DAu) return 0x00FAu; /* Ú → ú */
    return cp;
}

/*
 * lug_is_confusable — true for Cyrillic/Greek look-alikes that are
 * NOT valid Luganda codepoints but visually resemble Latin letters.
 *
 * The YAML generator rejects/normalises these, but the runtime
 * syllabifier should still guard against them (e.g. in user-supplied text).
 */
static inline bool __attribute__((unused)) lug_is_confusable(uint32_t cp)
{
    switch (cp) {
    case 0x03BFu: /* GREEK SMALL LETTER OMICRON (looks like o) */
    case 0x0430u: /* CYRILLIC SMALL LETTER A */
    case 0x043Au: /* CYRILLIC SMALL LETTER KA */
    case 0x043Eu: /* CYRILLIC SMALL LETTER O */
    case 0x0443u: /* CYRILLIC SMALL LETTER U */
    case 0x0456u: /* CYRILLIC SMALL LETTER BYELORUSSIAN-UKRAINIAN I */
    case 0x014Au: /* LATIN CAPITAL LETTER ENG (Ŋ — use lowercase ŋ) */
        return true;
    default:
        return false;
    }
}

/* =========================================================
 *  Phonotactic syllable-position predicates
 *
 *  These combine the basic character tests into the specific patterns
 *  the syllabifier needs.  They mirror the Luganda syllable hierarchy:
 *
 *    NCSV > GSV > NCV > CSV > GV > CV > V > NV(ŋ)
 *
 *  See the PHONOTACTIC SYLLABLE TYPES section in syllables.yaml.
 * ========================================================= */

/*
 * lug_is_onset_nasal — nasal that can begin a prenasalised syllable.
 *
 * 'm' prenasalises labials (mb, mf, mp, mv).
 * 'n' prenasalises coronals and velars (nc, nd, ng, nj, nk, ns, nt, nz).
 * 'ŋ' forms bare NV syllables only; it does NOT prenasalise.
 *
 * Returns true for m and n; NOT for ŋ.
 */
static inline bool __attribute__((unused)) lug_is_onset_nasal(uint32_t cp)
{
    return cp == 0x006Du /* m */ || cp == 0x006Eu /* n */;
}

/*
 * lug_nasal_valid_cluster — true if nasal `n_cp` may precede consonant `c_cp`
 * in a prenasalised onset.
 *
 * Encoding of Luganda prenasalisation constraints:
 *   m + {b, f, p, v}           → mb mf mp mv
 *   n + {c, d, g, j, k, s, t, z} → nc nd ng nj nk ns nt nz
 *
 * This function returns false for invalid clusters (e.g. n+b or m+g)
 * so the syllabifier treats them as separate syllables.
 */
static inline bool __attribute__((unused)) lug_nasal_valid_cluster(uint32_t n_cp, uint32_t c_cp)
{
    if (n_cp == 0x006Du) { /* m */
        switch (c_cp) {
        case 'b': case 'f': case 'p': case 'v': return true;
        default:                                  return false;
        }
    }
    if (n_cp == 0x006Eu) { /* n */
        switch (c_cp) {
        case 'c': case 'd': case 'g': case 'j':
        case 'k': case 's': case 't': case 'z': return true;
        default:                                 return false;
        }
    }
    return false;
}

/* =========================================================
 *  Additional utility functions for preprocessing pipelines
 * ========================================================= */

/*
 * lug_is_long_vowel — detect if two consecutive vowels form a long vowel.
 *
 * Handles both plain and toned vowels by comparing their base forms.
 * Returns true if cp1 and cp2 are the same vowel nucleus (ignoring tone).
 *
 * Example: lug_is_long_vowel(0x0061, 0x00E1) returns true (a + á → long a).
 */
static inline bool __attribute__((unused)) lug_is_long_vowel(uint32_t cp1, uint32_t cp2)
{
    return lug_is_vowel(cp1) && lug_is_vowel(cp2)
        && lug_vowel_base(cp1) == lug_vowel_base(cp2);
}

/*
 * lug_is_ny_lead — check if a codepoint is the start of the "ny" digraph.
 *
 * Returns true only for 'n' (U+006E); caller must check the next codepoint
 * for 'y' to confirm the digraph.
 *
 * Useful for lookahead parsing when processing syllables that include
 * the palatal nasal (ñ is typically represented as "ny" in Luganda).
 */
static inline bool __attribute__((unused)) lug_is_ny_lead(uint32_t cp)
{
    return cp == 0x006Eu; /* 'n' */
}

/*
 * utf8_encode_len — number of bytes needed to encode a codepoint as UTF-8.
 *
 * Inverse of utf8_char_len(). Returns 1–4 depending on the codepoint value.
 * Useful for pre-allocating buffers or computing encoded string length.
 */
static inline int __attribute__((unused)) utf8_encode_len(uint32_t cp)
{
    if (cp < 0x80u) return 1;
    if (cp < 0x800u) return 2;
    if (cp < 0x10000u) return 3;
    return 4;
}

/*
 * lug_is_valid_luganda_codepoint — comprehensive validity check.
 *
 * Returns true if the codepoint is a valid Luganda character:
 * - a vowel (plain or toned), or
 * - a consonant (excluding alien q/x), or
 * - whitespace
 *
 * Returns false for confusables and characters outside the Luganda set.
 * Use this for high-level input validation and sanitization.
 */
static inline bool __attribute__((unused)) lug_is_valid_luganda_codepoint(uint32_t cp)
{
    if (lug_is_confusable(cp)) return false;
    if (lug_is_vowel(cp)) return true;
    if (lug_is_luganda_consonant(cp)) return true;
    /* Whitespace and ASCII punctuation are acceptable */
    if (cp < 0x80u && (cp == ' ' || cp == '\t' || cp == '\n'
                       || cp == '\r' || cp == '.' || cp == ','
                       || cp == '!' || cp == '?')) {
        return true;
    }
    return false;
}

#endif /* UTF8_LUGANDA_H */
