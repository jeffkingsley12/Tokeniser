#ifndef CSR_HYPERGRAPH_H
#define CSR_HYPERGRAPH_H

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>

/* ============================================================================
 * CSR HYPERGRAPH STRUCTURES
 * Zero-copy, memory-mappable compressed sparse row matrix for deterministic
 * grammatical rule overrides. Designed for cache-line efficiency.
 * ============================================================================ */

#define CSR_MAGIC 0x43535248  /* "CSRH" in hex */
#define CSR_VERSION 1

/* Cache-aligned parallel arrays for the CSR Hypergraph.
 * Designed to be memory-mapped directly from disk (zero-copy). */
typedef struct {
    uint32_t magic;              /* File format identifier */
    uint32_t version;            /* Format version */
    uint32_t vocab_size;         /* Size of vocabulary (number of rows) */
    uint32_t num_edges;          /* Total number of rule edges */
    uint32_t row_ptr_offset;     /* Byte offset to row_ptr array */
    uint32_t ctx_tokens_offset;  /* Byte offset to ctx_tokens array */
    uint32_t modifiers_offset;   /* Byte offset to modifiers array */
    uint32_t rule_masks_offset;  /* Byte offset to rule_masks array */
    uint32_t reserved[2];        /* Future expansion */
} CSRHeader;

/* Runtime CSR structure (pointers into memory-mapped data) */
typedef struct {
    uint32_t vocab_size;
    uint32_t num_edges;
    
    /* Row offsets: Size = vocab_size + 1. Indexed by Candidate Token ID. */
    const uint32_t *restrict row_ptr;      
    
    /* Column indices: Target context Token IDs that trigger a rule. */
    const uint32_t *restrict ctx_tokens;   
    
    /* Multipliers stored as Q8.8 fixed-point to avoid FPU bloat in memory. */
    const uint16_t *restrict modifiers;    
    
    /* Bitmasks for strict multi-word constraints (e.g., Luganda noun-class agreement) */
    const uint64_t *restrict rule_masks;   
    
    /* Memory mapping metadata for proper cleanup */
    void* mapped_base;      /* Original mmap base address */
    size_t mapped_size;     /* Size of mapped region */
} CSRHypergraph;

/* Multi-word carryover state tracker */
typedef struct {
    uint32_t tokens[5];           /* Up to 5-gram history window */
    uint64_t active_class_mask;   /* Rolling bitmask of active morphological classes */
    uint8_t  history_len;
} MultiWordContext;

/* ============================================================================
 * GRAMMAR RULE INTERMEDIATE REPRESENTATION
 * Used during compilation before CSR construction
 * ============================================================================ */

typedef struct {
    uint32_t candidate_token;     /* The token being scored (row index) */
    uint32_t context_token;       /* Token in history that triggers this rule */
    float    modifier;            /* Probability multiplier (e.g., 2.0f for boost) */
    uint64_t rule_mask;           /* Bitmask for multi-word class constraints */
} GrammarRule;

/* ============================================================================
 * COMPILER API
 * ============================================================================ */

/**
 * csr_compile_from_rules - Compile grammar rules into CSR binary format
 * 
 * @param rules: Array of grammar rules
 * @param num_rules: Number of rules in the array
 * @param vocab_size: Size of the vocabulary (determines row_ptr size)
 * @param output_path: Path to write the binary CSR file
 * @return: 0 on success, -1 on error
 */
int csr_compile_from_rules(const GrammarRule* rules, size_t num_rules,
                          uint32_t vocab_size, const char* output_path);

/**
 * csr_load_from_file - Memory-map a CSR binary file
 * 
 * @param path: Path to the CSR binary file
 * @param out_hypergraph: Output parameter for the loaded hypergraph
 * @return: 0 on success, -1 on error
 */
int csr_load_from_file(const char* path, CSRHypergraph* out_hypergraph);

/**
 * csr_unload - Unmap a CSR binary file
 * 
 * @param hypergraph: The hypergraph to unload
 */
void csr_unload(CSRHypergraph* hypergraph);

/**
 * csr_parse_json_rules - Parse grammar rules from JSON file
 * 
 * Expected JSON format:
 * {
 *   "vocab_size": 65536,
 *   "rules": [
 *     {
 *       "candidate_token": 1234,
 *       "context_token": 5678,
 *       "modifier": 2.0,
 *       "rule_mask": 0x8
 *     },
 *     ...
 *   ]
 * }
 * 
 * @param json_path: Path to the JSON rules file
 * @param out_rules: Output array of parsed rules
 * @param out_num_rules: Output number of rules
 * @param out_vocab_size: Output vocabulary size
 * @return: 0 on success, -1 on error
 */
int csr_parse_json_rules(const char* json_path, GrammarRule** out_rules,
                         size_t* out_num_rules, uint32_t* out_vocab_size);

/**
 * csr_free_rules - Free memory allocated for parsed rules
 * 
 * @param rules: Array of rules to free
 */
void csr_free_rules(GrammarRule* rules);

/* ============================================================================
 * UTILITY FUNCTIONS
 * ============================================================================ */

/**
 * float_to_q88 - Convert float to Q8.8 fixed-point
 * 
 * @param f: Float value to convert
 * @return: Q8.8 representation (0-65535, where 256 = 1.0)
 */
static inline uint16_t float_to_q88(float f) {
    if (f < 0.0f) return 0;
    if (f > 255.996f) return 65535;  /* Max Q8.8 value */
    return (uint16_t)(f * 256.0f + 0.5f);
}

/**
 * q88_to_float - Convert Q8.8 fixed-point to float
 * 
 * @param q: Q8.8 value
 * @return: Float representation
 */
static inline float q88_to_float(uint16_t q) {
    return (float)q / 256.0f;
}

#endif /* CSR_HYPERGRAPH_H */
