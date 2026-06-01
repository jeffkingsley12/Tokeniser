/*
===============================================================================
CSR HYPERGRAPH COMPILER - MAIN DRIVER
Command-line tool for compiling grammar rules into CSR binary format.

Usage:
    ./csr_compile <input_json> <output_bin>

Example:
    ./csr_compile luganda_grammar_rules.json luganda_hypergraph.bin
===============================================================================
*/

#include "csr_hypergraph.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void print_usage(const char* program_name) {
    fprintf(stderr, "Usage: %s <input_json> <output_bin>\n", program_name);
    fprintf(stderr, "\n");
    fprintf(stderr, "Arguments:\n");
    fprintf(stderr, "  input_json   Path to JSON file containing grammar rules\n");
    fprintf(stderr, "  output_bin   Path to output CSR binary file\n");
    fprintf(stderr, "\n");
    fprintf(stderr, "JSON Format:\n");
    fprintf(stderr, "  {\n");
    fprintf(stderr, "    \"vocab_size\": 65536,\n");
    fprintf(stderr, "    \"rules\": [\n");
    fprintf(stderr, "      {\n");
    fprintf(stderr, "        \"candidate_token\": 1234,\n");
    fprintf(stderr, "        \"context_token\": 5678,\n");
    fprintf(stderr, "        \"modifier\": 2.0,\n");
    fprintf(stderr, "        \"rule_mask\": 0x8\n");
    fprintf(stderr, "      },\n");
    fprintf(stderr, "      ...\n");
    fprintf(stderr, "    ]\n");
    fprintf(stderr, "  }\n");
    fprintf(stderr, "\n");
    fprintf(stderr, "Rule Fields:\n");
    fprintf(stderr, "  candidate_token  Token ID being scored (row index in CSR)\n");
    fprintf(stderr, "  context_token    Token in history that triggers the rule\n");
    fprintf(stderr, "  modifier         Probability multiplier (Q8.8 fixed-point)\n");
    fprintf(stderr, "  rule_mask        Bitmask for multi-word class constraints\n");
}

int main(int argc, char* argv[]) {
    if (argc != 3) {
        print_usage(argv[0]);
        return 1;
    }
    
    const char* input_json = argv[1];
    const char* output_bin = argv[2];
    
    printf("CSR Hypergraph Compiler\n");
    printf("=======================\n");
    printf("Input:  %s\n", input_json);
    printf("Output: %s\n", output_bin);
    printf("\n");
    
    /* Parse JSON rules */
    GrammarRule* rules = NULL;
    size_t num_rules = 0;
    uint32_t vocab_size = 0;
    
    printf("Parsing JSON rules...\n");
    if (csr_parse_json_rules(input_json, &rules, &num_rules, &vocab_size) != 0) {
        fprintf(stderr, "ERROR: Failed to parse JSON rules\n");
        return 1;
    }
    
    printf("  Parsed %zu rules\n", num_rules);
    printf("  Vocab size: %u\n", vocab_size);
    printf("\n");
    
    /* Compile to CSR binary */
    printf("Compiling CSR matrix...\n");
    if (csr_compile_from_rules(rules, num_rules, vocab_size, output_bin) != 0) {
        fprintf(stderr, "ERROR: Failed to compile CSR matrix\n");
        csr_free_rules(rules);
        return 1;
    }
    
    printf("\n");
    printf("Compilation successful!\n");
    printf("\n");
    
    /* Verify by loading the compiled file */
    printf("Verifying compiled file...\n");
    CSRHypergraph hypergraph;
    if (csr_load_from_file(output_bin, &hypergraph) != 0) {
        fprintf(stderr, "WARNING: Failed to load compiled file for verification\n");
    } else {
        printf("  Verification passed\n");
        /* LOW FIX M-2: Unmap the region to prevent memory leak */
        csr_unload(&hypergraph);
    }
    
    /* Cleanup */
    csr_free_rules(rules);
    
    return 0;
}
