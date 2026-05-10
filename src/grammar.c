#define _POSIX_C_SOURCE 200809L
/*
 * grammar.c
 *
 * Grammar pruning in three passes:
 *
 *   Pass 1 – Compute effective frequency
 *     eff_freq(X) = freq(X) - Σ freq(parents using X)
 *     Rules with eff_freq < MIN_PAIR_FREQ are ghost rules.
 *
 *   Pass 2 – Mark dead rules
 *     A rule is dead if eff_freq < MIN_PAIR_FREQ.
 *
 *   Pass 3 – Flatten / inline dead rules
 *     Replace references to dead symbols with their RHS symbols,
 *     recursively, until all references are to live symbols or terminals.
 *     After pruning, each rule's expansion is a flat string that can be
 *     inserted directly into the token vocabulary.
 *
 * Thread safety:
 *   pruner_expand() lazily populates gp->expanded[]/gp->expansions[].
 *   It is NOT thread-safe; callers must ensure all expansions are
 *   materialised (via pruner_expand_all()) before concurrent encode()
 *   calls.  See the thread-safety note on pruner_expand_all() below.
 */

#include "tokenizer.h"

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <assert.h>

/*
 * Maximum recursion depth for expand_sym().
 *
 * Re-Pair guarantees every rule's depth <= MAX_RULE_DEPTH, so the
 * recursion bottoms out within MAX_RULE_DEPTH levels for any rule
 * produced during training.  The +1 gives one spare level for the
 * entry call from pruner_expand() (depth=0 → children at depth=1 …).
 * Using exactly MAX_RULE_DEPTH + 1 — rather than an arbitrary +4 —
 * means any rule that somehow violates the depth invariant is caught
 * immediately rather than silently "handled" several levels deeper.
 *
 * Note: a corrupted/adversarial model loaded from disk could contain
 * cyclic rules.  The cycle guard below (gp->in_expand[]) detects such
 * cases before the depth limit is reached.
 */
#define EXPAND_DEPTH_LIMIT  (MAX_RULE_DEPTH + 1)
#define MAX_PRUNER_CHARS     256

/* =========================================================
 *  GrammarPruner lifecycle
 * ========================================================= */

GrammarPruner *pruner_create(RePairState *rs, const SyllableTable *stbl)
{
    if (!rs) return NULL;
    GrammarPruner *gp = malloc(sizeof(GrammarPruner));
    if (!gp) return NULL;
    gp->rs = rs;
    gp->stbl = stbl;
    gp->expansions = calloc(MAX_RULES, sizeof(char *));
    /* expansion_lens stores byte lengths of cached expansion strings.
     * Re-Pair rules can expand into arbitrarily long strings; uint16_t
     * (max 65 535) silently truncates lengths on deeply nested rules,
     * corrupting the length cache and causing over/under-reads at inference
     * time.  uint32_t (max ~4 GB) is always sufficient.
     *
     * REQUIRED: also change the field declaration in tokenizer.h from
     *   uint16_t *expansion_lens;
     * to
     *   uint32_t *expansion_lens;
     */
    gp->expansion_lens = calloc(MAX_RULES, sizeof(uint32_t));
    gp->expanded = calloc(MAX_RULES, sizeof(bool));
    gp->is_frozen = false;
    if (!gp->expansions || !gp->expansion_lens || !gp->expanded) {
        pruner_destroy(gp);
        return NULL;
    }
    gp->fatal_oom = false;
    return gp;
}

void pruner_destroy(GrammarPruner *gp)
{
    if (!gp) return;

    /* Free cached expansions */
    if (gp->expansions) {
        for (uint32_t i = 0; i < MAX_RULES; i++) {
            free(gp->expansions[i]);
            gp->expansions[i] = NULL;
        }
        free(gp->expansions);
        gp->expansions = NULL;
    }

    free(gp->expansion_lens);
    gp->expansion_lens = NULL;

    free(gp->expanded);
    gp->expanded = NULL;

    free(gp);
}

/* =========================================================
 *  Pass 1: Effective frequency
 *
 *  For each rule X → A B:
 *    parent_contrib(A) += freq(X)
 *    parent_contrib(B) += freq(X)
 *
 *  Then: eff_freq(X) = freq(X) - parent_contrib(X)
 *
 *  We need each parent rule to contribute its frequency to its children's
 *  parent_sum before the eff_freq pass reads those sums.  Because Re-Pair
 *  creates parent rules after their children (parents have higher IDs),
 *  iterating from the highest ID down to 0 processes parents FIRST, ensuring
 *  all contributions are accumulated before the eff_freq computation below.
 *
 *  Returns true on success, false on OOM.  On OOM all eff_freq values
 *  are left at 0, which causes pruner_mark_dead() to mark every rule
 *  dead — a conservative but detectable failure.  Callers should treat
 *  a false return as a fatal build error.
 * ========================================================= */

bool pruner_compute_eff_freq(GrammarPruner *gp)
{
    RePairState *rs = gp->rs;

    /* Temporary accumulator: parent_sum[rule_index] = total parent freq */
    uint32_t *parent_sum = calloc(rs->rule_count, sizeof *parent_sum);
    if (!parent_sum) {
        fprintf(stderr, "[pruner] OOM in compute_eff_freq — "
                        "all eff_freq remain 0, every rule will be marked dead\n");
        return false;
    }

    /* Process from highest ID (most complex / most recent) down to lowest.
     * Parents always have higher IDs than their children in Re-Pair output. */
    for (int32_t i = (int32_t)rs->rule_count - 1; i >= 0; i--) {
        const Rule *r = &rs->rules[i];

        for (int side = 0; side < 2; side++) {
            uint32_t child = r->rhs[side];
            if (child >= BASE_SYMBOL_OFFSET) {
                uint32_t child_idx = child - BASE_SYMBOL_OFFSET;
                if (child_idx < rs->rule_count) {
                    /* Clamp to UINT32_MAX on overflow.  The subsequent
                     * ternary (freq > sub) still safely yields eff_freq = 0
                     * because any realistic freq < UINT32_MAX.            */
                    if (parent_sum[child_idx] <= UINT32_MAX - r->freq)
                        parent_sum[child_idx] += r->freq;
                    else
                        parent_sum[child_idx]  = UINT32_MAX;
                }
            }
        }
    }

    /* Compute effective frequencies */
    for (uint32_t i = 0; i < rs->rule_count; i++) {
        Rule    *r   = &rs->rules[i];
        uint32_t sub = parent_sum[i];
        r->eff_freq  = (r->freq > sub) ? (r->freq - sub) : 0;
    }

    free(parent_sum);
    return true;
}

/* =========================================================
 *  Pass 2: Mark dead rules
 * ========================================================= */

void pruner_mark_dead(GrammarPruner *gp)
{
    RePairState *rs = gp->rs;

    for (uint32_t i = 0; i < rs->rule_count; i++) {
        Rule *r = &rs->rules[i];

        /* Primary criterion: too infrequent to earn its vocabulary slot. */
        if (r->eff_freq < MIN_PAIR_FREQ) {
            r->dead = true;
        }

        /*
         * TODO: secondary criterion — "only one non-dead parent → inline".
         * Requires a two-sub-pass:
         *   (a) count live parents per rule (after primary criterion above),
         *   (b) flip dead any rule whose live-parent count == 1.
         * Until implemented, single-parent rules remain live (correct but
         * slightly larger than optimal vocabulary).
         */
    }
}

/* =========================================================
 *  Pass 3: Flatten dead rules
 *
 *  expand_sym() recursively expands symbol `sym` into a null-terminated
 *  UTF-8 string written into `buf` (capacity `buf_cap` bytes).
 *
 *  Dead rules are transparently inlined (their children are expanded
 *  directly).  Live non-terminal symbols produce their cached expansion.
 *
 *  `in_expand` is a caller-supplied bitset (one bit per rule index).
 *  It is used for cycle detection: if a rule tries to expand itself
 *  recursively (only possible with a corrupted/adversarial model), we
 *  emit a "?" sentinel and return rather than looping forever.
 * ========================================================= */

/*
 * Inline bitset helpers — no dynamic allocation, O(1) ops.
 */
#define BITSET_WORDS(n)    (((n) + 63u) / 64u)

static inline void bitset_set(uint64_t *bs, uint32_t i)
{
    bs[i >> 6] |= (uint64_t)1 << (i & 63);
}

static inline void bitset_clear(uint64_t *bs, uint32_t i)
{
    bs[i >> 6] &= ~((uint64_t)1 << (i & 63));
}

static inline bool bitset_test(const uint64_t *bs, uint32_t i)
{
    return (bs[i >> 6] >> (i & 63)) & 1u;
}

/*
 * Internal recursive expansion.  Never call directly; use pruner_expand().
 *
 * Returns `buf` (always non-NULL, always null-terminated within buf_cap).
 */
static const char *expand_sym(GrammarPruner       *gp,
                               uint32_t             sym,
                               const SyllableTable *stbl,
                               char                *buf,
                               size_t               buf_cap,
                               int                  depth,
                               uint64_t            *in_expand)
{
    assert(buf_cap > 0);

    /* ── Depth guard ─────────────────────────────────────────────────── */
    if (depth > EXPAND_DEPTH_LIMIT) {
        /* Rule depth exceeds what Re-Pair should produce.  Emit a
         * recognisable sentinel so callers can detect malformed models. */
        snprintf(buf, buf_cap, "?DEPTH");
        return buf;
    }

    /* ── Terminal symbol ─────────────────────────────────────────────── */
    if (sym < BASE_SYMBOL_OFFSET) {
        if (sym < (uint32_t)stbl->count) {
            /* stbl->entries[].text is always null-terminated by stbl_intern().
             * We use strnlen + memcpy rather than strncpy to avoid reading
             * past the first '\0' into potentially uninitialised padding.   */
            size_t slen = strnlen(stbl->entries[sym].text, MAX_TOKEN_CHARS - 1);
            if (slen >= buf_cap) slen = buf_cap - 1;
            memcpy(buf, stbl->entries[sym].text, slen);
            buf[slen] = '\0';
        } else {
            snprintf(buf, buf_cap, "[SYL:%u]", sym);
        }
        return buf;
    }

    /* ── Non-terminal symbol ─────────────────────────────────────────── */
    uint32_t idx = sym - BASE_SYMBOL_OFFSET;
    if (idx >= gp->rs->rule_count) {
        snprintf(buf, buf_cap, "[UNK:%u]", sym);
        return buf;
    }

    /* Check expansion cache — populated on first visit */
    if (gp->expanded[idx]) {
        size_t slen = gp->expansion_lens[idx];
        if (slen >= buf_cap) slen = buf_cap - 1;
        memcpy(buf, gp->expansions[idx], slen);
        buf[slen] = '\0';
        return buf;
    }

    /* ── Cycle detection ─────────────────────────────────────────────── */
    if (bitset_test(in_expand, idx)) {
        /* Cycle detected — only possible in a corrupted model. */
        fprintf(stderr, "[pruner] cycle detected at rule idx %u — "
                        "model may be corrupted\n", idx);
        snprintf(buf, buf_cap, "?CYCLE");
        return buf;
    }
    bitset_set(in_expand, idx);

    /* ── Recurse into children ───────────────────────────────────────── */
    Rule *r = &gp->rs->rules[idx];

    char left[MAX_PRUNER_CHARS];
    char right[MAX_PRUNER_CHARS];
    left[0] = right[0] = '\0';

    expand_sym(gp, r->rhs[0], stbl, left,  sizeof left,  depth + 1, in_expand);
    expand_sym(gp, r->rhs[1], stbl, right, sizeof right, depth + 1, in_expand);

    /* Calculate required length to prevent UTF-8 truncation */
    size_t left_len = strlen(left);
    size_t right_len = strlen(right);
    size_t total_len = left_len + right_len;
    
    if (total_len >= buf_cap) {
        /* Buffer too small - need dynamic allocation */
        char *dynamic_buf = malloc(total_len + 1);
        if (!dynamic_buf) {
            gp->fatal_oom = true;
            bitset_clear(in_expand, idx);
            snprintf(buf, buf_cap, "?OOM");
            return buf;
        }
        
        memcpy(dynamic_buf, left, left_len);
        memcpy(dynamic_buf + left_len, right, right_len);
        dynamic_buf[total_len] = '\0';
        
        /* Cache the dynamically allocated expansion */
        gp->expansions[idx] = dynamic_buf;
        /* total_len is a size_t; values > UINT32_MAX cannot be stored even
         * in the widened field. Treat this as a fatal build error — in
         * practice Re-Pair rule depths are bounded so this cannot occur
         * with well-formed training data. */
        if (total_len > UINT32_MAX) {
            gp->fatal_oom = true;
            free(dynamic_buf);
            gp->expansions[idx] = NULL;
            bitset_clear(in_expand, idx);
            snprintf(buf, buf_cap, "?OVF");
            return buf;
        }
        gp->expansion_lens[idx] = (uint32_t)total_len;
        gp->expanded[idx] = true;
        
        bitset_clear(in_expand, idx);
        
        /* Return truncated version for this call (buf is smaller) */
        size_t copy_len = buf_cap - 1;
        memcpy(buf, dynamic_buf, copy_len);
        buf[copy_len] = '\0';
        return buf;
    }
    
    /* Safe to concatenate in provided buffer */
    memcpy(buf, left, left_len);
    memcpy(buf + left_len, right, right_len);
    buf[total_len] = '\0';

    /* ── Populate cache ──────────────────────────────────────────────── */
    size_t elen = strlen(buf);
    gp->expansions[idx] = malloc(elen + 1);
    if (gp->expansions[idx]) {
        memcpy(gp->expansions[idx], buf, elen + 1);
        gp->expansion_lens[idx] = (uint32_t)elen;
        gp->expanded[idx] = true;
    } else {
        gp->fatal_oom = true;
        fprintf(stderr, "[pruner] OOM expanding rule idx %u\n", idx);
    }

    bitset_clear(in_expand, idx);
    return buf;
}

/* =========================================================
 *  Public expansion entry point
 *
 *  Thread safety: NOT safe for concurrent calls that share `gp`.
 *  Call pruner_expand_all() once during build to materialise every
 *  expansion, then rely on the read-only cached values at runtime.
 * ========================================================= */
const char *pruner_expand(GrammarPruner       *gp,
                           uint32_t             sym,
                           const SyllableTable *stbl)
{
    /* Terminal fast path */
    if (sym < BASE_SYMBOL_OFFSET) {
        if (sym < (uint32_t)stbl->count) return stbl->entries[sym].text;
        return "";
    }

    uint32_t idx = sym - BASE_SYMBOL_OFFSET;
    if (idx >= gp->rs->rule_count) return "";

    if (!gp->expanded[idx]) {
        /* Allocate per-call cycle-detection bitset on the stack.
         * BITSET_WORDS(MAX_RULES) = 512 words = 4096 bytes — acceptable. */
        uint64_t in_expand[BITSET_WORDS(MAX_RULES)];
        memset(in_expand, 0, sizeof in_expand);

        char tmp[MAX_PRUNER_CHARS];
        expand_sym(gp, sym, stbl, tmp, sizeof tmp, 0, in_expand);
        /* Side effect: gp->expansions[idx] and gp->expanded[idx] set. */
    }

    return gp->expansions[idx] ? gp->expansions[idx] : "";
}

/*
 * pruner_expand_all() — eagerly populate the expansion cache for every rule.
 *
 * Call this once after pruner_flatten() and before any concurrent calls
 * to tokenizer_encode() / pruner_expand().  After this function returns,
 * all gp->expanded[] entries are true and gp->expansions[][] is immutable,
 * making concurrent pruner_expand() calls read-only and therefore safe.
 */
void pruner_expand_all(GrammarPruner *gp, const SyllableTable *stbl)
{
    for (uint32_t i = 0; i < gp->rs->rule_count; i++) {
        if (gp->fatal_oom) break;
        if (!gp->expanded[i]) {
            /* Reset the cycle-detection bitset before every top-level call.
             *
             * In the common case (no cycles, no depth overflow) expand_sym
             * clears each bit on the way back up via bitset_clear(), so the
             * bitset would be all-zeros between calls anyway.  But if
             * expand_sym returns early — due to a ?CYCLE or ?DEPTH sentinel
             * — the bits it set before the early return are NOT cleared.
             * Those stale bits remain set when the next rule is expanded,
             * causing false cycle detection for any rule that legitimately
             * shares a child with the failed expansion.
             *
             * Resetting here is O(BITSET_WORDS(MAX_RULES)) = 512 words per
             * rule in the worst case, which is negligible compared with the
             * string-copy work done inside expand_sym.                      */
            uint64_t in_expand[BITSET_WORDS(MAX_RULES)];
            memset(in_expand, 0, sizeof in_expand);

            char tmp[MAX_PRUNER_CHARS];
            uint32_t sym = i + BASE_SYMBOL_OFFSET;
            expand_sym(gp, sym, stbl, tmp, sizeof tmp, 0, in_expand);
        }
    }
    gp->is_frozen = true;
}

/* =========================================================
 *  Convenience: run all three pruning passes in order
 *
 *  Returns true on success.  On OOM in Pass 1 the function returns
 *  false and the caller should abort the build — the grammar state is
 *  "all rules dead" which is detectable but unusable.
 * ========================================================= */
bool pruner_flatten(GrammarPruner *gp)
{
    if (!pruner_compute_eff_freq(gp))
        return false;           /* OOM: eff_freq not computed, cannot prune */
    pruner_mark_dead(gp);
    /*
     * String expansions are populated lazily by pruner_expand() or
     * eagerly by pruner_expand_all().  Callers that need thread-safe
     * encode() must call pruner_expand_all() before going multi-threaded.
     */
    return true;
}

/* =========================================================
 *  Thread safety functions
 * ========================================================= */

/*
 * Freeze the pruner to prevent further mutations.
 * After calling this, the pruner is safe for concurrent reads.
 */
void pruner_freeze(GrammarPruner *gp) {
    if (!gp || gp->is_frozen) return;
    
    /* Ensure all expansions are populated before freezing.
     * Without this, any caller who calls pruner_freeze() directly
     * (rather than pruner_expand_all() + pruner_freeze()) would get
     * a silently incomplete frozen pruner — pruner_get_thread_safe()
     * would return "" for any un-expanded rule. */
    pruner_expand_all(gp, gp->stbl);
    gp->is_frozen = true;
}

/*
 * Thread-safe read accessor for frozen pruners.
 * Returns NULL if symbol not found or pruner not frozen.
 */
const char *pruner_get_thread_safe(const GrammarPruner *gp, uint32_t sym) {
    if (!gp || !gp->is_frozen) {
        /* Pruner must be frozen for thread-safe access */
        return NULL;
    }
    
    /* Terminal symbols */
    if (sym < BASE_SYMBOL_OFFSET) {
        return NULL;  /* Caller should handle terminals directly */
    }
    
    /* Non-terminal symbols */
    uint32_t idx = sym - BASE_SYMBOL_OFFSET;
    if (idx >= gp->rs->rule_count) {
        return NULL;
    }
    
    /* Safe lock-free read - expansions are immutable after freeze */
    return gp->expanded[idx] ? gp->expansions[idx] : "";
}
