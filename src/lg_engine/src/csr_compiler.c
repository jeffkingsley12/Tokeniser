/*
===============================================================================
CSR HYPERGRAPH COMPILER
Offline compiler for deterministic grammatical rule overrides.

Compiles grammar rules into a memory-mappable Compressed Sparse Row (CSR)
matrix format for zero-copy loading and cache-efficient rule evaluation.
===============================================================================
*/

#include "csr_hypergraph.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <errno.h>

/* ============================================================================
 * JSON PARSER (Minimal implementation for grammar rules)
 * ============================================================================ */

typedef struct {
    const char* data;
    size_t len;
    size_t pos;
} JSONParser;

static char json_peek(JSONParser* p) {
    if (p->pos < p->len) return p->data[p->pos];
    return '\0';
}

static char json_next(JSONParser* p) {
    if (p->pos < p->len) return p->data[p->pos++];
    return '\0';
}

static void json_skip_whitespace(JSONParser* p) {
    while (p->pos < p->len) {
        char c = p->data[p->pos];
        if (c == ' ' || c == '\t' || c == '\n' || c == '\r') {
            p->pos++;
        } else {
            break;
        }
    }
}

static int json_expect(JSONParser* p, char expected) {
    json_skip_whitespace(p);
    if (json_peek(p) == expected) {
        json_next(p);
        return 0;
    }
    fprintf(stderr, "JSON parse error: expected '%c' at position %zu\n", expected, p->pos);
    return -1;
}

static int json_parse_string(JSONParser* p, char* out_buf, size_t buf_len) {
    json_skip_whitespace(p);
    if (json_peek(p) != '"') return -1;
    json_next(p);  /* Skip opening quote */
    
    size_t i = 0;
    while (p->pos < p->len && i < buf_len - 1) {
        char c = json_next(p);
        if (c == '"') {
            out_buf[i] = '\0';
            return 0;
        }
        if (c == '\\' && p->pos < p->len) {
            /* Handle escape sequences */
            c = json_next(p);
            switch (c) {
                case 'n': c = '\n'; break;
                case 't': c = '\t'; break;
                case 'r': c = '\r'; break;
                case '\\': c = '\\'; break;
                case '"': c = '"'; break;
                case 'b': c = '\b'; break;  /* M-7 FIX: Add backspace */
                case 'f': c = '\f'; break;  /* M-7 FIX: Add form feed */
                case '/': c = '/'; break;   /* M-7 FIX: Add forward slash */
                case 'u': {                 /* M-7 FIX: Add Unicode escape */
                    /* Parse 4 hex digits for Unicode escape */
                    uint16_t unicode_val = 0;
                    for (int j = 0; j < 4 && p->pos < p->len; j++) {
                        char hex_c = json_next(p);
                        uint16_t nibble = 0;
                        if (hex_c >= '0' && hex_c <= '9') nibble = hex_c - '0';
                        else if (hex_c >= 'a' && hex_c <= 'f') nibble = hex_c - 'a' + 10;
                        else if (hex_c >= 'A' && hex_c <= 'F') nibble = hex_c - 'A' + 10;
                        else break;
                        unicode_val = (unicode_val << 4) | nibble;
                    }
                    /* For simplicity, just store as '?' if not ASCII */
                    c = (unicode_val < 128) ? (char)unicode_val : '?';
                    break;
                }
                default: break;
            }
        }
        out_buf[i++] = c;
    }
    out_buf[i] = '\0';
    return -1;  /* Unterminated string */
}

static int json_parse_uint32(JSONParser* p, uint32_t* out_val) {
    json_skip_whitespace(p);
    uint32_t val = 0;
    int digits = 0;
    
    while (p->pos < p->len) {
        char c = p->data[p->pos];
        if (c >= '0' && c <= '9') {
            /* HIGH FIX H-5: Check for overflow before multiplication and addition */
            if (val > UINT32_MAX / 10) return -1;  /* Would overflow on multiply */
            uint32_t new_val = val * 10;
            uint32_t digit = c - '0';
            if (new_val > UINT32_MAX - digit) return -1;  /* Would overflow on add */
            val = new_val + digit;
            p->pos++;
            digits++;
        } else {
            break;
        }
    }
    
    if (digits == 0) return -1;
    *out_val = val;
    return 0;
}

static int __attribute__((unused)) json_parse_uint64(JSONParser* p, uint64_t* out_val) {
    json_skip_whitespace(p);
    uint64_t val = 0;
    int digits = 0;
    
    while (p->pos < p->len) {
        char c = p->data[p->pos];
        if (c >= '0' && c <= '9') {
            /* HIGH FIX H-5: Check for overflow before multiplication and addition */
            if (val > UINT64_MAX / 10) return -1;  /* Would overflow on multiply */
            uint64_t new_val = val * 10;
            uint64_t digit = c - '0';
            if (new_val > UINT64_MAX - digit) return -1;  /* Would overflow on add */
            val = new_val + digit;
            p->pos++;
            digits++;
        } else {
            break;
        }
    }
    
    if (digits == 0) return -1;
    *out_val = val;
    return 0;
}

static int json_parse_float(JSONParser* p, float* out_val) {
    json_skip_whitespace(p);
    char buf[64];
    size_t i = 0;
    
    /* Handle optional sign */
    if (json_peek(p) == '-' || json_peek(p) == '+') {
        buf[i++] = json_next(p);
    }
    
    /* Parse integer part */
    while (p->pos < p->len && i < 63) {
        char c = p->data[p->pos];
        if (c >= '0' && c <= '9') {
            buf[i++] = json_next(p);
        } else {
            break;
        }
    }
    
    /* Parse fractional part */
    if (json_peek(p) == '.' && i < 63) {
        buf[i++] = json_next(p);
        while (p->pos < p->len && i < 63) {
            char c = p->data[p->pos];
            if (c >= '0' && c <= '9') {
                buf[i++] = json_next(p);
            } else {
                break;
            }
        }
    }
    
    /* CRITICAL FIX: Parse scientific notation (e.g., 1.2e-2, 3.5E+10) */
    if ((json_peek(p) == 'e' || json_peek(p) == 'E') && i < 63) {
        buf[i++] = json_next(p);
        /* Optional sign in exponent */
        if (json_peek(p) == '-' || json_peek(p) == '+') {
            buf[i++] = json_next(p);
        }
        /* Parse exponent digits */
        while (p->pos < p->len && i < 63) {
            char c = p->data[p->pos];
            if (c >= '0' && c <= '9') {
                buf[i++] = json_next(p);
            } else {
                break;
            }
        }
    }
    
    if (i == 0) return -1;
    buf[i] = '\0';
    *out_val = (float)atof(buf);
    return 0;
}

/* HIGH FIX H-4: Skip JSON value including nested objects/arrays.
 * The previous implementation only skipped until the next comma or closing brace,
 * which breaks on nested structures. This function tracks nesting depth to
 * properly skip entire values including nested objects and arrays. */
static void json_skip_value(JSONParser* p) {
    json_skip_whitespace(p);
    char c = json_peek(p);
    
    if (c == '"') {
        /* Skip string */
        json_next(p);  /* Skip opening quote */
        while (json_peek(p) != '"' && json_peek(p) != '\0') {
            if (json_peek(p) == '\\') json_next(p);  /* Skip escaped char */
            json_next(p);
        }
        if (json_peek(p) == '"') json_next(p);  /* Skip closing quote */
    } else if (c == '{' || c == '[') {
        /* Skip object or array with nesting depth tracking */
        int depth = 1;
        json_next(p);  /* Skip opening brace/bracket */
        
        while (depth > 0 && json_peek(p) != '\0') {
            char cur = json_peek(p);
            if (cur == '{' || cur == '[') {
                depth++;
            } else if (cur == '}' || cur == ']') {
                depth--;
            } else if (cur == '"') {
                /* Skip strings inside the structure */
                json_next(p);  /* Skip opening quote */
                while (json_peek(p) != '"' && json_peek(p) != '\0') {
                    if (json_peek(p) == '\\') json_next(p);
                    json_next(p);
                }
                if (json_peek(p) == '"') json_next(p);
            }
            json_next(p);
        }
    } else {
        /* Skip primitive (number, true, false, null) */
        while (json_peek(p) != ',' && json_peek(p) != '}' && 
               json_peek(p) != ']' && json_peek(p) != '\0') {
            json_next(p);
        }
    }
}

static int json_parse_hex_uint64(JSONParser* p, uint64_t* out_val) {
    json_skip_whitespace(p);
    
    /* Skip "0x" prefix if present */
    if (p->pos + 2 < p->len && p->data[p->pos] == '0' && 
        (p->data[p->pos + 1] == 'x' || p->data[p->pos + 1] == 'X')) {
        p->pos += 2;
    }
    
    uint64_t val = 0;
    int digits = 0;
    
    while (p->pos < p->len) {
        char c = p->data[p->pos];
        uint64_t nibble;
        if (c >= '0' && c <= '9')      nibble = (uint64_t)(c - '0');
        else if (c >= 'a' && c <= 'f') nibble = (uint64_t)(c - 'a' + 10);
        else if (c >= 'A' && c <= 'F') nibble = (uint64_t)(c - 'A' + 10);
        else break;

        /* FIX: Check for overflow before shifting.
         * A 64-bit hex value has at most 16 nibbles.  If val already has bits
         * in the top 4 positions, shifting left by 4 would overflow. */
        if (val > (UINT64_MAX >> 4)) return -1;
        val = (val << 4) | nibble;
        p->pos++;
        digits++;
    }
    
    if (digits == 0) return -1;
    *out_val = val;
    return 0;
}

int csr_parse_json_rules(const char* json_path, GrammarRule** out_rules,
                         size_t* out_num_rules, uint32_t* out_vocab_size) {
    FILE* f = fopen(json_path, "r");
    if (!f) {
        fprintf(stderr, "Failed to open JSON file: %s\n", json_path);
        return -1;
    }
    
    fseek(f, 0, SEEK_END);
    long file_size = ftell(f);
    fseek(f, 0, SEEK_SET);
    
    if (file_size <= 0) {
        fprintf(stderr, "Invalid file size: %ld\n", file_size);
        fclose(f);
        return -1;
    }
    
    char* json_data = (char*)malloc(file_size + 1);
    if (!json_data) {
        fprintf(stderr, "Failed to allocate JSON buffer\n");
        fclose(f);
        return -1;
    }
    
    size_t bytes_read = fread(json_data, 1, file_size, f);
    fclose(f);
    json_data[bytes_read] = '\0';
    
    JSONParser parser = { json_data, bytes_read, 0 };
    
    /* Parse top-level object */
    if (json_expect(&parser, '{') != 0) {
        free(json_data);
        return -1;
    }
    
    uint32_t vocab_size = 0;
    GrammarRule* rules = NULL;
    size_t rules_capacity = 0;
    size_t rules_count = 0;
    
    while (json_peek(&parser) != '}') {
        char key[64];
        int rc = json_parse_string(&parser, key, sizeof(key));
        if (rc != 0) {
            /* CORR-2 FIX: Provide explicit error message for long keys */
            fprintf(stderr, "ERROR: Failed to parse JSON key (key too long or invalid) at position %zu\n", parser.pos);
            free(json_data);
            if (rules) free(rules);
            return -1;
        }
        
        if (json_expect(&parser, ':') != 0) {
            free(json_data);
            if (rules) free(rules);
            return -1;
        }
        
        if (strcmp(key, "vocab_size") == 0) {
            if (json_parse_uint32(&parser, &vocab_size) != 0) {
                free(json_data);
                if (rules) free(rules);
                return -1;
            }
        } else if (strcmp(key, "rules") == 0) {
            if (json_expect(&parser, '[') != 0) {
                free(json_data);
                if (rules) free(rules);
                return -1;
            }
            
            while (json_peek(&parser) != ']') {
                if (json_expect(&parser, '{') != 0) {
                    free(json_data);
                    if (rules) free(rules);
                    return -1;
                }
                
                GrammarRule rule = {0};
                
                while (json_peek(&parser) != '}') {
                    char rule_key[64];
                    if (json_parse_string(&parser, rule_key, sizeof(rule_key)) != 0) {
                        free(json_data);
                        if (rules) free(rules);
                        return -1;
                    }
                    
                    if (json_expect(&parser, ':') != 0) {
                        free(json_data);
                        if (rules) free(rules);
                        return -1;
                    }
                    
                    if (strcmp(rule_key, "candidate_token") == 0) {
                        if (json_parse_uint32(&parser, &rule.candidate_token) != 0) {
                            free(json_data);
                            if (rules) free(rules);
                            return -1;
                        }
                    } else if (strcmp(rule_key, "context_token") == 0) {
                        if (json_parse_uint32(&parser, &rule.context_token) != 0) {
                            free(json_data);
                            if (rules) free(rules);
                            return -1;
                        }
                    } else if (strcmp(rule_key, "modifier") == 0) {
                        if (json_parse_float(&parser, &rule.modifier) != 0) {
                            free(json_data);
                            if (rules) free(rules);
                            return -1;
                        }
                    } else if (strcmp(rule_key, "rule_mask") == 0) {
                        if (json_parse_hex_uint64(&parser, &rule.rule_mask) != 0) {
                            free(json_data);
                            if (rules) free(rules);
                            return -1;
                        }
                    } else {
                        /* HIGH FIX H-4: Use json_skip_value to properly handle nested structures */
                        json_skip_value(&parser);
                    }
                    
                    if (json_peek(&parser) == ',') json_next(&parser);
                }
                
                json_expect(&parser, '}');
                
                /* Add rule to array */
                if (rules_count >= rules_capacity) {
                    rules_capacity = rules_capacity == 0 ? 64 : rules_capacity * 2;
                    GrammarRule* new_rules = (GrammarRule*)realloc(rules, 
                        rules_capacity * sizeof(GrammarRule));
                    if (!new_rules) {
                        free(json_data);
                        if (rules) free(rules);
                        return -1;
                    }
                    rules = new_rules;
                }
                
                rules[rules_count++] = rule;
                
                if (json_peek(&parser) == ',') json_next(&parser);
            }
            
            json_expect(&parser, ']');
        } else {
            /* FIX: Use json_skip_value() to correctly skip nested objects/arrays.
             * The previous byte-scan loop only consumed until ',' or '}', which
             * stalls the parser on any nested structure (e.g. "metadata": {...}).
             * json_skip_value() tracks nesting depth and handles strings, arrays,
             * and objects correctly. */
            json_skip_value(&parser);
        }
        
        if (json_peek(&parser) == ',') json_next(&parser);
    }
    
    json_expect(&parser, '}');
    free(json_data);
    
    *out_rules = rules;
    *out_num_rules = rules_count;
    *out_vocab_size = vocab_size;
    
    return 0;
}

void csr_free_rules(GrammarRule* rules) {
    if (rules) free(rules);
}

/* ============================================================================
 * CSR MATRIX CONSTRUCTION
 * ============================================================================ */

static int compare_rules_by_candidate(const void* a, const void* b) {
    const GrammarRule* ra = (const GrammarRule*)a;
    const GrammarRule* rb = (const GrammarRule*)b;
    if (ra->candidate_token != rb->candidate_token) {
        /* CRITICAL FIX C-6: Branchless three-way compare, no overflow */
        return (ra->candidate_token > rb->candidate_token) -
               (ra->candidate_token < rb->candidate_token);
    }
    return 0;
}

int csr_compile_from_rules(const GrammarRule* rules, size_t num_rules,
                          uint32_t vocab_size, const char* output_path) {
    if (!rules || num_rules == 0 || vocab_size == 0 || !output_path) {
        fprintf(stderr, "Invalid parameters for CSR compilation\n");
        return -1;
    }
    
    /* Allocate and sort rules by candidate token */
    GrammarRule* sorted_rules = (GrammarRule*)malloc(num_rules * sizeof(GrammarRule));
    if (!sorted_rules) {
        fprintf(stderr, "Failed to allocate sorted rules array\n");
        return -1;
    }
    memcpy(sorted_rules, rules, num_rules * sizeof(GrammarRule));
    qsort(sorted_rules, num_rules, sizeof(GrammarRule), compare_rules_by_candidate);
    
    /* Count edges per candidate (row) */
    uint32_t* row_counts = (uint32_t*)calloc(vocab_size, sizeof(uint32_t));
    if (!row_counts) {
        fprintf(stderr, "Failed to allocate row counts\n");
        free(sorted_rules);
        return -1;
    }
    
    for (size_t i = 0; i < num_rules; i++) {
        if (sorted_rules[i].candidate_token < vocab_size) {
            row_counts[sorted_rules[i].candidate_token]++;
        }
    }
    
    /* Compute row offsets (CSR format) */
    uint32_t* row_ptr = (uint32_t*)malloc((vocab_size + 1) * sizeof(uint32_t));
    if (!row_ptr) {
        fprintf(stderr, "Failed to allocate row_ptr\n");
        free(row_counts);
        free(sorted_rules);
        return -1;
    }
    
    row_ptr[0] = 0;
    for (uint32_t i = 0; i < vocab_size; i++) {
        row_ptr[i + 1] = row_ptr[i] + row_counts[i];
    }
    
    /* Allocate CSR arrays */
    uint32_t* ctx_tokens = (uint32_t*)malloc(num_rules * sizeof(uint32_t));
    uint16_t* modifiers = (uint16_t*)malloc(num_rules * sizeof(uint16_t));
    uint64_t* rule_masks = (uint64_t*)malloc(num_rules * sizeof(uint64_t));
    
    if (!ctx_tokens || !modifiers || !rule_masks) {
        fprintf(stderr, "Failed to allocate CSR arrays\n");
        free(row_ptr);
        free(row_counts);
        free(sorted_rules);
        if (ctx_tokens) free(ctx_tokens);
        if (modifiers) free(modifiers);
        if (rule_masks) free(rule_masks);
        return -1;
    }
    
    /* Fill CSR arrays from sorted rules */
    uint32_t* current_pos = (uint32_t*)calloc(vocab_size, sizeof(uint32_t));
    if (!current_pos) {
        fprintf(stderr, "Failed to allocate current_pos\n");
        free(row_ptr);
        free(row_counts);
        free(sorted_rules);
        free(ctx_tokens);
        free(modifiers);
        free(rule_masks);
        return -1;
    }
    
    for (size_t i = 0; i < num_rules; i++) {
        uint32_t candidate = sorted_rules[i].candidate_token;
        if (candidate >= vocab_size) continue;
        
        uint32_t idx = row_ptr[candidate] + current_pos[candidate];
        ctx_tokens[idx] = sorted_rules[i].context_token;
        modifiers[idx] = float_to_q88(sorted_rules[i].modifier);
        rule_masks[idx] = sorted_rules[i].rule_mask;
        current_pos[candidate]++;
    }
    
    free(current_pos);
    free(row_counts);
    free(sorted_rules);
    
    /* Calculate file offsets with 64-byte alignment for cache lines.
     * FIX: All intermediate offset variables are size_t to prevent silent
     * uint32_t overflow on large rule sets (e.g. 100 M rules → multi-GB file).
     * The CSRHeader stores offsets as uint32_t, so we guard against truncation
     * before casting. */
    size_t header_size = sizeof(CSRHeader);
    size_t row_ptr_size = (size_t)(vocab_size + 1) * sizeof(uint32_t);
    size_t ctx_tokens_size = (size_t)num_rules * sizeof(uint32_t);
    size_t modifiers_size = (size_t)num_rules * sizeof(uint16_t);
    size_t rule_masks_size = (size_t)num_rules * sizeof(uint64_t);

    /* Guard: reject impossibly large inputs early */
    if (num_rules > (size_t)UINT32_MAX / sizeof(uint64_t)) {
        fprintf(stderr, "ERROR: num_rules %zu too large for CSR format\n", num_rules);
        free(row_ptr);
        free(ctx_tokens);
        free(modifiers);
        free(rule_masks);
        return -1;
    }

    /* Align each section to 64 bytes — keep all intermediates in size_t */
    const size_t align = 64;
    size_t row_ptr_offset     = ((header_size + align - 1) / align) * align;
    size_t ctx_tokens_offset  = ((row_ptr_offset    + row_ptr_size    + align - 1) / align) * align;
    size_t modifiers_offset   = ((ctx_tokens_offset + ctx_tokens_size + align - 1) / align) * align;
    size_t rule_masks_offset  = ((modifiers_offset  + modifiers_size  + align - 1) / align) * align;
    size_t total_size         = rule_masks_offset + rule_masks_size;

    /* FIX: Verify that all offsets fit in uint32_t before writing the header.
     * If the computed layout exceeds 4 GB the CSRHeader fields would truncate
     * silently, producing an unloadable file. Fail loudly instead. */
    if (row_ptr_offset    > (size_t)UINT32_MAX ||
        ctx_tokens_offset > (size_t)UINT32_MAX ||
        modifiers_offset  > (size_t)UINT32_MAX ||
        rule_masks_offset > (size_t)UINT32_MAX ||
        total_size        > (size_t)UINT32_MAX) {
        fprintf(stderr, "ERROR: CSR file layout would exceed 4 GB (total=%zu bytes). "
                "Reduce num_rules or increase offset width in CSRHeader.\n", total_size);
        free(row_ptr);
        free(ctx_tokens);
        free(modifiers);
        free(rule_masks);
        return -1;
    }
    
    /* Write binary file */
    FILE* f = fopen(output_path, "wb");
    if (!f) {
        fprintf(stderr, "Failed to open output file: %s\n", output_path);
        free(row_ptr);
        free(ctx_tokens);
        free(modifiers);
        free(rule_masks);
        return -1;
    }
    
    /* Write header */
    CSRHeader header = {
        .magic = CSR_MAGIC,
        .version = CSR_VERSION,
        .vocab_size = vocab_size,
        .num_edges = (uint32_t)num_rules,
        .row_ptr_offset    = (uint32_t)row_ptr_offset,
        .ctx_tokens_offset = (uint32_t)ctx_tokens_offset,
        .modifiers_offset  = (uint32_t)modifiers_offset,
        .rule_masks_offset = (uint32_t)rule_masks_offset,
        .reserved = {0, 0}
    };
    
    if (fwrite(&header, sizeof(CSRHeader), 1, f) != 1) {
        fprintf(stderr, "Failed to write header\n");
        fclose(f);
        free(row_ptr);
        free(ctx_tokens);
        free(modifiers);
        free(rule_masks);
        return -1;
    }
    
    /* Write padding to align row_ptr */
    uint8_t padding[64] = {0};
    /* F-4 FIX: Compile-time assertion prevents out-of-bounds read if someone
     * increases 'align' beyond the hardcoded padding buffer size.
     * VLA is avoided because it has no stack overflow protection. */
    _Static_assert(sizeof(padding) >= 64, "padding buffer must be >= align");
    /* padding_size is always < align (64) due to the ceiling-align formula */
    size_t padding_size = row_ptr_offset - header_size;
    if (padding_size > 0 && fwrite(padding, padding_size, 1, f) != 1) {
        fprintf(stderr, "Failed to write padding\n");
        fclose(f);
        free(row_ptr);
        free(ctx_tokens);
        free(modifiers);
        free(rule_masks);
        return -1;
    }
    
    /* Write row_ptr */
    if (fwrite(row_ptr, row_ptr_size, 1, f) != 1) {
        fprintf(stderr, "Failed to write row_ptr\n");
        fclose(f);
        free(row_ptr);
        free(ctx_tokens);
        free(modifiers);
        free(rule_masks);
        return -1;
    }
    
    /* Write padding to align ctx_tokens */
    padding_size = ctx_tokens_offset - (row_ptr_offset + row_ptr_size);
    if (padding_size > 0 && fwrite(padding, padding_size, 1, f) != 1) {
        fprintf(stderr, "Failed to write padding\n");
        fclose(f);
        free(row_ptr);
        free(ctx_tokens);
        free(modifiers);
        free(rule_masks);
        return -1;
    }
    
    /* Write ctx_tokens */
    if (fwrite(ctx_tokens, ctx_tokens_size, 1, f) != 1) {
        fprintf(stderr, "Failed to write ctx_tokens\n");
        fclose(f);
        free(row_ptr);
        free(ctx_tokens);
        free(modifiers);
        free(rule_masks);
        return -1;
    }
    
    /* Write padding to align modifiers */
    padding_size = modifiers_offset - (ctx_tokens_offset + ctx_tokens_size);
    if (padding_size > 0 && fwrite(padding, padding_size, 1, f) != 1) {
        fprintf(stderr, "Failed to write padding\n");
        fclose(f);
        free(row_ptr);
        free(ctx_tokens);
        free(modifiers);
        free(rule_masks);
        return -1;
    }
    
    /* Write modifiers */
    if (fwrite(modifiers, modifiers_size, 1, f) != 1) {
        fprintf(stderr, "Failed to write modifiers\n");
        fclose(f);
        free(row_ptr);
        free(ctx_tokens);
        free(modifiers);
        free(rule_masks);
        return -1;
    }
    
    /* Write padding to align rule_masks */
    padding_size = rule_masks_offset - (modifiers_offset + modifiers_size);
    if (padding_size > 0 && fwrite(padding, padding_size, 1, f) != 1) {
        fprintf(stderr, "Failed to write padding\n");
        fclose(f);
        free(row_ptr);
        free(ctx_tokens);
        free(modifiers);
        free(rule_masks);
        return -1;
    }
    
    /* Write rule_masks */
    if (fwrite(rule_masks, rule_masks_size, 1, f) != 1) {
        fprintf(stderr, "Failed to write rule_masks\n");
        fclose(f);
        free(row_ptr);
        free(ctx_tokens);
        free(modifiers);
        free(rule_masks);
        return -1;
    }
    
    fclose(f);
    
    free(row_ptr);
    free(ctx_tokens);
    free(modifiers);
    free(rule_masks);
    
    printf("CSR hypergraph compiled successfully: %s\n", output_path);
    printf("  Vocab size: %u\n", vocab_size);
    printf("  Rules: %zu\n", num_rules);
    printf("  File size: %zu bytes\n", total_size);
    
    return 0;
}

/* ============================================================================
 * MEMORY-MAPPED FILE LOADING
 * ============================================================================ */

int csr_load_from_file(const char* path, CSRHypergraph* out_hypergraph) {
    if (!path || !out_hypergraph) {
        fprintf(stderr, "Invalid parameters for CSR loading\n");
        return -1;
    }
    
    int fd = open(path, O_RDONLY);
    if (fd == -1) {
        fprintf(stderr, "Failed to open CSR file: %s (errno=%d)\n", path, errno);
        return -1;
    }
    
    struct stat st;
    if (fstat(fd, &st) == -1) {
        fprintf(stderr, "Failed to stat CSR file: %s\n", path);
        close(fd);
        return -1;
    }
    
    void* mapped = mmap(NULL, st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (mapped == MAP_FAILED) {
        fprintf(stderr, "Failed to mmap CSR file: %s\n", path);
        close(fd);
        return -1;
    }
    
    CSRHeader* header = (CSRHeader*)mapped;
    if (header->magic != CSR_MAGIC) {
        fprintf(stderr, "Invalid CSR magic number: 0x%x\n", header->magic);
        munmap(mapped, st.st_size);
        close(fd);
        return -1;
    }
    
    if (header->version != CSR_VERSION) {
        fprintf(stderr, "Unsupported CSR version: %u\n", header->version);
        munmap(mapped, st.st_size);
        close(fd);
        return -1;
    }

    /* HIGH FIX H-6: Validate all offsets are within mapped file bounds.
     * Corrupt or adversarial files could have offsets pointing outside the
     * mapped region, causing out-of-bounds access. */
    if (header->row_ptr_offset >= (size_t)st.st_size) {
        fprintf(stderr, "Invalid row_ptr_offset: %u (file size: %ld)\n",
                header->row_ptr_offset, st.st_size);
        munmap(mapped, st.st_size);
        close(fd);
        return -1;
    }
    if (header->ctx_tokens_offset >= (size_t)st.st_size) {
        fprintf(stderr, "Invalid ctx_tokens_offset: %u (file size: %ld)\n",
                header->ctx_tokens_offset, st.st_size);
        munmap(mapped, st.st_size);
        close(fd);
        return -1;
    }
    if (header->modifiers_offset >= (size_t)st.st_size) {
        fprintf(stderr, "Invalid modifiers_offset: %u (file size: %ld)\n",
                header->modifiers_offset, st.st_size);
        munmap(mapped, st.st_size);
        close(fd);
        return -1;
    }
    if (header->rule_masks_offset >= (size_t)st.st_size) {
        fprintf(stderr, "Invalid rule_masks_offset: %u (file size: %ld)\n",
                header->rule_masks_offset, st.st_size);
        munmap(mapped, st.st_size);
        close(fd);
        return -1;
    }

    /* Validate that data sections don't overflow the mapped region */
    /* INT-3 FIX: Cast to size_t before addition to prevent overflow on large vocab_size */
    size_t row_ptr_size = ((size_t)header->vocab_size + 1) * sizeof(uint32_t);
    if (header->row_ptr_offset + row_ptr_size > (size_t)st.st_size) {
        fprintf(stderr, "row_ptr section overflows file bounds\n");
        munmap(mapped, st.st_size);
        close(fd);
        return -1;
    }
    size_t ctx_tokens_size = header->num_edges * sizeof(uint32_t);
    if (header->ctx_tokens_offset + ctx_tokens_size > (size_t)st.st_size) {
        fprintf(stderr, "ctx_tokens section overflows file bounds\n");
        munmap(mapped, st.st_size);
        close(fd);
        return -1;
    }
    size_t modifiers_size = header->num_edges * sizeof(uint16_t);
    if (header->modifiers_offset + modifiers_size > (size_t)st.st_size) {
        fprintf(stderr, "modifiers section overflows file bounds\n");
        munmap(mapped, st.st_size);
        close(fd);
        return -1;
    }
    size_t rule_masks_size = header->num_edges * sizeof(uint64_t);
    if (header->rule_masks_offset + rule_masks_size > (size_t)st.st_size) {
        fprintf(stderr, "rule_masks section overflows file bounds\n");
        munmap(mapped, st.st_size);
        close(fd);
        return -1;
    }

    /* Set up hypergraph pointers */
    out_hypergraph->vocab_size = header->vocab_size;
    out_hypergraph->num_edges = header->num_edges;
    out_hypergraph->row_ptr = (const uint32_t*)((const char*)mapped + header->row_ptr_offset);
    out_hypergraph->ctx_tokens = (const uint32_t*)((const char*)mapped + header->ctx_tokens_offset);
    out_hypergraph->modifiers = (const uint16_t*)((const char*)mapped + header->modifiers_offset);
    out_hypergraph->rule_masks = (const uint64_t*)((const char*)mapped + header->rule_masks_offset);
    
    /* Store mapping metadata for proper cleanup */
    out_hypergraph->mapped_base = mapped;
    out_hypergraph->mapped_size = st.st_size;
    
    /* CRITICAL FIX: Close file descriptor after mmap succeeds
     * POSIX mmap adds a reference to the underlying file object, so the fd
     * can be safely closed. Failing to close this causes file descriptor leaks
     * in long-running processes. */
    close(fd);
    
    printf("CSR hypergraph loaded successfully: %s\n", path);
    printf("  Vocab size: %u\n", out_hypergraph->vocab_size);
    printf("  Rules: %u\n", out_hypergraph->num_edges);
    
    return 0;
}

void csr_unload(CSRHypergraph* hypergraph) {
    if (hypergraph && hypergraph->mapped_base && hypergraph->mapped_size > 0) {
        munmap(hypergraph->mapped_base, hypergraph->mapped_size);
        hypergraph->mapped_base = NULL;
        hypergraph->mapped_size = 0;
    }
}
