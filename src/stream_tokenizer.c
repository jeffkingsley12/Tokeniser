#include "tokenizer.h"
#include <string.h>
#include <unistd.h>
#include <fcntl.h>
#include <errno.h>

#define STREAM_BUF_SIZE     65536
#define STREAM_MIN_OUTPUT   64

void stream_tokenizer_init(StreamTokenizer *st, const Tokenizer *t) {
    if (!st || !t) return;
    memset(st, 0, sizeof *st);
    st->t = t;
}

static ssize_t stream_refill(StreamTokenizer *st, int fd) {
    if (st->head > 0 && st->tail > st->head) {
        size_t remaining = st->tail - st->head;
        memmove(st->buf, st->buf + st->head, remaining);
        if (st->best_pos != NULL) {
            if (st->best_pos >= st->buf + st->head) {
                st->best_pos -= st->head;
            } else {
                st->best_pos = NULL;
                st->best_token = 0;
                st->in_token = false;
            }
        }
        st->tail = remaining;
        st->head = 0;
    }

    if (!st->eof && st->tail < STREAM_BUF_SIZE) {
        ssize_t n = read(fd, st->buf + st->tail, STREAM_BUF_SIZE - st->tail);
        if (n < 0) {
            if (errno == EAGAIN || errno == EINTR) return 0;
            return -1;
        }
        if (n == 0) st->eof = true;
        else st->tail += (size_t)n;
    }
    
    /* Safely null-terminate since st->buf has a 16-byte overflow sentinel */
    st->buf[st->tail] = '\0';
    
    return (ssize_t)(st->tail - st->head);
}



static size_t find_last_safe_boundary(const uint8_t *buf, size_t len) {
    if (len < 4) return 0; // Ensure a healthy margin for Luganda lookahead
    size_t i = len - 4;    // Back off by max Luganda onset/vowel length
    while (i > 0 && (buf[i] & 0xC0) == 0x80) i--;
    return i;
}

TokenizerStatus stream_tokenizer_process(StreamTokenizer *st, uint32_t *out, 
                                         size_t out_cap, size_t *out_used) {
    if (!st || !out || !out_used) return TOKENIZER_MORE;
    *out_used = 0;

    const Tokenizer *t = st->t;
    const LOUDS *l = t->louds;
    if (!l || !l->has_csr) return TOKENIZER_MORE;

    size_t old_head = st->head;
    size_t available = st->tail - st->head;

    if (available == 0 && st->eof) {
        if (st->in_token && st->best_token != 0) {
            if (*out_used >= out_cap) return TOKENIZER_MORE;
            out[(*out_used)++] = st->best_token - 1;
            st->best_token = 0;
            st->in_token = false;
        }
        return TOKENIZER_EOF;
    }

    size_t safe_len = st->eof ? available : find_last_safe_boundary(st->buf + st->head, available);
    const uint8_t *p = st->buf + st->head;
    const uint8_t *limit = p + safe_len;

    while (p < limit && *out_used < out_cap) {
        uint16_t syl_id;
        const uint8_t *next_p = p;
        int consumed = consume_syllable_fast(t, &next_p, &syl_id);

        if (consumed <= 0) {
            if (consumed == 0) {
                if (st->eof) {
                    /* At EOF, force progress on unrecognized bytes */
                    consumed = -1;
                } else {
                    break;
                }
            }
            size_t skip = utf8_seq_len(p);
            if (skip == 0) skip = 1;
            if (skip > (size_t)(limit - p)) skip = (size_t)(limit - p);

            if (st->in_token && st->best_token != 0) {
                if (*out_used >= out_cap) break;
                out[(*out_used)++] = st->best_token - 1;
            }
            st->trie_node = 0;
            st->best_token = 0;
            st->best_pos = NULL;
            st->in_token = false;
            p += skip;
            continue;
        }

        /* Fast-path: 1-syllable terminal token at root */
        if (st->trie_node == 0) {
            uint32_t ft = l->fast_token[syl_id];
            if (ft != 0 && !l->can_extend[syl_id]) {
                out[(*out_used)++] = ft - 1;
                p = next_p;
                continue;
            }
        }

        /* CSR Traversal */
        uint32_t start = l->row_ptr[st->trie_node];
        uint32_t end   = l->row_ptr[st->trie_node + 1];
        uint32_t next_node = 0xFFFFFFFF;

        uint32_t lo = start, hi = end;
        while (lo < hi) {
            uint32_t mid = (lo + hi) >> 1;
            if (l->labels[mid] < syl_id) lo = mid + 1;
            else hi = mid;
        }
        if (lo < end && l->labels[lo] == syl_id) next_node = l->next_node[lo];

        if (next_node != 0xFFFFFFFF) {
            st->trie_node = next_node;
            if (l->terminals[next_node] != 0) {
                st->best_token = l->terminals[next_node];
                st->best_pos = next_p;
            }
            st->in_token = true;
            p = next_p;
            continue;
        } else {
            /* Match ended */
            if (st->best_token != 0) {
                out[(*out_used)++] = st->best_token - 1;
                p = st->best_pos ? st->best_pos : next_p;
            } else {
                /* Fallback: emit syllable ID */
                out[(*out_used)++] = (uint32_t)syl_id;
                p = next_p;
            }
            st->trie_node = 0;
            st->best_token = 0;
            st->best_pos = NULL;
            st->in_token = false;
        }
    }

    st->head = (size_t)(p - st->buf);
    st->bytes_processed += (uint64_t)(st->head - old_head);

    if (*out_used >= out_cap || (*out_used >= STREAM_MIN_OUTPUT && !st->eof)) {
        return TOKENIZER_MORE;
    }
    return st->eof ? TOKENIZER_EOF : TOKENIZER_OK;
}

TokenizerStatus stream_tokenizer_file(StreamTokenizer *st, int fd, uint32_t *out, 
                                      size_t out_cap, size_t *out_used) {
    ssize_t n = stream_refill(st, fd);
    if (n < 0) return TOKENIZER_ERROR;
    return stream_tokenizer_process(st, out, out_cap, out_used);
}

TokenizerStatus stream_tokenizer_mem(StreamTokenizer *st,
                                     const uint8_t *data, size_t len,
                                     bool is_final,
                                     uint32_t *out, size_t out_cap,
                                     size_t *out_used,
                                     size_t *bytes_consumed) {
    if (!st || !data || !out || !out_used || !bytes_consumed) return TOKENIZER_ERROR;
    
    *bytes_consumed = 0;
    size_t space = STREAM_BUF_SIZE - st->tail;
    size_t to_copy = (len < space) ? len : space;
    
    if (to_copy > 0) {
        memcpy(st->buf + st->tail, data, to_copy);
        st->tail += to_copy;
        st->buf[st->tail] = '\0';
        *bytes_consumed = to_copy;
    }
    
    if (is_final && *bytes_consumed == len) st->eof = true;
    
    return stream_tokenizer_process(st, out, out_cap, out_used);
}
