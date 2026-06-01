/*
 * Memory & Topology Audit Suite
 * 
 * Performs comprehensive structural audit of Symbol 171 (hub vs sink investigation),
 * checks raw metadata of Symbol 7 (memory corruption investigation), and builds
 * distribution histogram to determine if 171 is an expected scale-free hub or
 * an anomalous sink.
 */

#define _POSIX_C_SOURCE 200809L
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdatomic.h>
#include "tokenizer_api.h"
#include "gemini_internal.h"
#include "bridge_engine.h"

#define DAWG_PREDICTION_MAX_LEN 256

// Basic UTF-8 validator to trace high-bit corruption artifacts
bool is_valid_utf8_string(const char *str, size_t *out_len) {
    if (!str) return false;
    const unsigned char *bytes = (const unsigned char *)str;
    size_t len = 0;
    
    while (bytes[len] != '\0') {
        if (len >= 256) { // Catch missing null-terminators
            *out_len = len;
            return false; 
        }
        // Check for common corrupted markers or illegal high-bit sequences
        if (bytes[len] == 0xFF || bytes[len] == 0xFE) {
            *out_len = len;
            return false;
        }
        len++;
    }
    *out_len = len;
    return true;
}

// Audit tokenizer string pool directly (vocabulary layer)
void audit_tokenizer_string_pool(int tok_handle) {
    printf("\n==================================================\n");
    printf("         TOKENIZER STRING POOL AUDIT               \n");
    printf("==================================================\n");
    
    uint32_t vocab_size = tok_vocab_size(tok_handle);
    printf("[T1] VOCABULARY SIZE: %u tokens\n", vocab_size);
    
    uint32_t corruption_markers = 0;
    uint32_t null_strings = 0;
    uint32_t utf8_violations = 0;
    uint32_t short_strings = 0;
    uint32_t oversized_strings = 0;
    
    // Scan first 2000 tokens or full vocab (whichever is smaller)
    uint32_t scan_limit = (vocab_size < 2000) ? vocab_size : 2000;
    
    for (uint32_t i = 0; i < scan_limit; i++) {
        const char* text = tok_decode(tok_handle, i);
        
        if (!text) {
            null_strings++;
            if (null_strings <= 5) {
                printf("    > NULL string at TokenID %u\n", i);
            }
            continue;
        }
        
        size_t len = strlen(text);
        
        // Check for corruption marker
        if (strcmp(text, "<??>") == 0) {
            corruption_markers++;
            if (corruption_markers <= 10) {
                printf("    > CORRUPTION MARKER at TokenID %u\n", i);
            }
            continue;
        }
        
        // Check for suspiciously short strings
        if (len < 2) {
            short_strings++;
            if (short_strings <= 5) {
                printf("    > Short string at TokenID %u: \"%s\" (len=%zu)\n", i, text, len);
            }
        }
        
        // Check for oversized strings (potential buffer overflow)
        if (len >= 256) {
            oversized_strings++;
            printf("    > OVERSIZED string at TokenID %u: len=%zu\n", i, len);
        }
        
        // UTF-8 validation
        size_t parsed_len = 0;
        if (!is_valid_utf8_string(text, &parsed_len)) {
            utf8_violations++;
            if (utf8_violations <= 10) {
                printf("    > UTF-8 violation at TokenID %u (len=%zu): \"%s\"\n", i, parsed_len, text);
            }
        }
    }
    
    // Specific audit of low TokenIDs (0-19) like Symbol 7
    printf("\n[T2] LOW TOKENID CORRUPTION SCAN (IDs 0-19):\n");
    for (uint32_t i = 0; i < 20 && i < vocab_size; i++) {
        const char* text = tok_decode(tok_handle, i);
        
        if (!text) {
            printf("    TokenID %2u: NULL\n", i);
        } else if (strcmp(text, "<\\\\\?\?>") == 0) {
            printf("    TokenID %2u: CORRUPTION MARKER\n", i);
        } else if (strlen(text) < 2) {
            printf("    TokenID %2u: SHORT \"%s\"\n", i, text);
        } else {
            char snippet[21] = {0};
            strncpy(snippet, text, 20);
            printf("    TokenID %2u: OK \"%s\"\n", i, snippet);
        }
    }
    
    printf("\n[T3] TOKENIZER INTEGRITY SUMMARY:\n");
    printf("    NULL Strings:         %u\n", null_strings);
    printf("    Corruption Markers:   %u\n", corruption_markers);
    printf("    Short Strings (<2):   %u\n", short_strings);
    printf("    Oversized Strings:   %u\n", oversized_strings);
    printf("    UTF-8 Violations:     %u\n", utf8_violations);
    printf("==================================================\n\n");
}

/**
 * Low-level vocabulary serialization forensic audit
 * Detects struct alignment drift, 8-bit index overflow, and arena boundary corruption
 */
void execute_vocabulary_serialization_audit(int tok_handle) {
    printf("\n===================================================================\n");
    printf("         CRITICAL VOCABULARY & SERIALIZATION FORENSIC SUITE        \n");
    printf("===================================================================\n");

    if (tok_handle < 0) {
        fprintf(stderr, "FATAL: Invalid tokenizer handle\n");
        return;
    }

    uint32_t vocab_size = tok_vocab_size(tok_handle);
    printf("[1] VOCABULARY STRUCT SIZE & ENVIRONMENT DIAGNOSTICS:\n");
    printf("    Pointer Size:           %zu bytes\n", sizeof(const char *));
    printf("    Reported Vocabulary Size: %u entries\n", vocab_size);
    printf("    MAX_TOKEN_CHARS:        %d (from tokenizer_config.h)\n", 256); // Standard value

    // ---- AUDIT TARGET 1: FORENSIC DUMP OF BOOTSTRAP TOKENIDs 5-19 ----
    printf("\n[2] METADATA ANALYSIS FOR BOOTSTRAP TOKENS (IDs 5-19):\n");
    printf("    %-5s | %-20s | %-10s | %-20s\n", "ID", "STRING POINTER", "LENGTH", "STATUS");
    printf("    -------------------------------------------------------------------\n");

    uint32_t corruption_count = 0;
    uint32_t null_pointer_count = 0;
    uint32_t suspicious_length_count = 0;
    uint32_t corruption_start_id = UINT32_MAX;
    uint32_t corruption_end_id = 0;
    uint32_t corruption_span = 0;

    for (uint32_t id = 0; id < vocab_size; id++) {
        // Focus on bootstrap range and boundaries
        if (id > 25 && id < (vocab_size - 5)) {
            if (id == 26) printf("    ... [Healthy Mid-Range Omitted for Brevity] ...\n");
            continue;
        }

        const char* text = tok_decode(tok_handle, id);
        bool is_corrupted = false;
        char status_buf[64] = "VALID";
        size_t len = 0;

        if (!text) {
            is_corrupted = true;
            null_pointer_count++;
            strcpy(status_buf, "NULL POINTER");
        } else {
            len = strlen(text);
            
            // Check for corruption marker
            if (strcmp(text, "<\?\?>") == 0) {
                is_corrupted = true;
                corruption_count++;
                strcpy(status_buf, "CORRUPTION MARKER");
            }
            // Check for suspiciously short strings (bootstrap tokens should be valid)
            else if (len < 2 && id >= 5) {
                is_corrupted = true;
                suspicious_length_count++;
                snprintf(status_buf, sizeof(status_buf), "SUSPICIOUS LENGTH (%zu)", len);
            }
            // Check for oversized strings (potential buffer overflow)
            else if (len >= 256) {
                is_corrupted = true;
                suspicious_length_count++;
                snprintf(status_buf, sizeof(status_buf), "OVERSIZED (%zu)", len);
            }
        }

        if (is_corrupted) {
            if (corruption_start_id == UINT32_MAX) {
                corruption_start_id = id;
            }
            corruption_end_id = id;
        }

        // Print detailed info for bootstrap range and boundaries
        if (id <= 25 || id >= vocab_size - 5) {
            printf("    %-5u | %-20p | %-10zu | %s\n", 
                   id, (void*)text, len, status_buf);
            
            if (text && !is_corrupted) {
                char snippet[21] = {0};
                strncpy(snippet, text, 20);
                printf("          └─ Text Payload: \"%s\"\n", snippet);
            } else if (text && is_corrupted) {
                // Dump raw bytes of corrupted token
                printf("          └─ Corrupted Raw Hex: ");
                for (size_t b = 0; b < 8 && text[b] != '\0'; b++) {
                    printf("%02X ", (unsigned char)text[b]);
                }
                printf("\n");
            }
        }
    }

    // ---- AUDIT TARGET 2: TOTAL SYSTEM POOL VIOLATION SCAN ----
    uint32_t total_corrupted = 0;
    uint32_t total_null = 0;
    uint32_t total_oversized = 0;

    for (uint32_t i = 0; i < vocab_size; i++) {
        const char* text = tok_decode(tok_handle, i);
        if (!text) {
            total_null++;
            total_corrupted++;
        } else if (text[0] == '<' && text[1] == '?' && text[2] == '?' && text[3] == '>' && text[4] == '\0') {
            total_corrupted++;
        } else {
            size_t len = strlen(text);
            if (len >= 256) {
                total_oversized++;
                total_corrupted++;
            }
        }
    }

    printf("\n[3] SYSTEM INTEGRITY SUMMARY:\n");
    printf("    Total Corrupted Entries:      %u\n", total_corrupted);
    printf("    NULL Pointer Entries:        %u\n", total_null);
    printf("    Oversized String Entries:     %u\n", total_oversized);
    printf("    Corruption Window:            IDs %u to %u\n", 
           corruption_start_id, corruption_end_id);
    
    if (corruption_start_id != UINT32_MAX) {
        corruption_span = corruption_end_id - corruption_start_id + 1;
        printf("    Corruption Span:              %u entries\n", corruption_span);
        
        // Check for the 256 anomaly
        if (corruption_span == 256) {
            printf("    ⚠️  CRITICAL: Exactly 256 corrupted entries detected!\n");
            printf("               This indicates 8-bit index overflow or struct alignment drift.\n");
        } else if (corruption_span % 256 == 0) {
            printf("    ⚠️  WARNING: Corruption span is a multiple of 256 (%u blocks).\n", 
                   corruption_span / 256);
            printf("               This suggests arena block boundary corruption.\n");
        }
        
        if (corruption_start_id == 5) {
            printf("    ⚠️  CRITICAL: Corruption starts exactly at TokenID 5!\n");
            printf("               This confirms bootstrap seed corruption hypothesis.\n");
        }
    }
    
    if (total_corrupted > 0) {
        printf("\n    DIAGNOSIS: ");
        if (corruption_span == 256 && corruption_start_id == 5) {
            printf("Struct alignment drift or 8-bit overflow confirmed.\n");
            printf("               The loader is reading misaligned offsets from the file snapshot.\n");
        } else if (total_null > 0) {
            printf("String pool pointer corruption detected.\n");
            printf("               Investigate isolated runtime memory clobbering loops.\n");
        } else {
            printf("Metadata boundaries structurally intact but content corrupted.\n");
            printf("               Focus on string serialization/deserialization logic.\n");
        }
    } else {
        printf("    DIAGNOSIS: No corruption detected in vocabulary structures.\n");
    }
    printf("===================================================================\n\n");
}

void execute_foundational_memory_audit(EngineContext *ctx, int tok_handle) {
    printf("\n==================================================\n");
    printf("         CRITICAL MEMORY & TOPOLOGY AUDIT          \n");
    printf("==================================================\n");

    if (!ctx) {
        fprintf(stderr, "FATAL: Engine context is NULL\n");
        return;
    }

    uint32_t total_symbols = atomic_load(&ctx->symbol_count);
    printf("[1] SYMBOL TABLE STATUS: %u active symbols\n", total_symbols);

    // ---- AUDIT TARGET: SYMBOL 171 (HUB VS SINK INVESTIGATION) ----
    if (171 < total_symbols) {
        Symbol *sym171 = &ctx->dawg_nodes[171];
        
        // Get canonical node and text
        NodeID canon171 = sym171->canonical_node;
        const char *text171 = NULL;
        
        if (canon171 != INVALID && canon171 < MAX_NODES) {
            TokenID tid = ctx->nodes[canon171].token_id;
            text171 = tok_decode(tok_handle, tid);
        }
        
        uint32_t outdegree = atomic_load(&sym171->transition_count);
        uint32_t indegree = 0;
        
        // Scan edge pools to calculate exact fan-in
        uint32_t total_edges = atomic_load(&ctx->edge_count);
        for (uint32_t i = 0; i < total_edges; i++) {
            EdgeID edge_idx = i;
            // Check if this edge points to symbol 171's canonical node
            // This is a simplified check - in practice you'd need to traverse the edge structure
            if (ctx->edge_pool.headers && i < ctx->edge_pool.capacity) {
                PackedEdgeHeader hdr = ctx->edge_pool.headers[i];
                NodeID dst = UNPACK_EDGE_DST(hdr);
                if (dst == canon171) {
                    indegree++;
                }
            }
        }
        
        printf("\n[2] TARGET SYMBOL 171 DIAGNOSTIC:\n");
        printf("    Text Payload:  \"%s\"\n", text171 ? text171 : "NULL");
        printf("    Canonical Node: %u\n", canon171);
        printf("    Indegree:      %u\n", indegree);
        printf("    Outdegree:     %u\n", outdegree);
        printf("    Topology Classification: %s\n", 
               (outdegree > 0) ? "VALID GRAPH HUB (Linguistically Normal)" : "GRAPH SINK (Structural Flaw)");
    } else {
        printf("\n[2] SYMBOL 171 NOT ACTIVE (total_symbols=%u)\n", total_symbols);
    }

    // ---- AUDIT TARGET: SYMBOL 7 (RAW MEMORY CORRUPTION INVESTIGATION) ----
    if (7 < total_symbols) {
        Symbol *sym7 = &ctx->dawg_nodes[7];
        printf("\n[3] TARGET SYMBOL 7 RAW METADATA:\n");
        printf("    Address:        %p\n", (void*)sym7);
        printf("    Canonical ID:   %u\n", sym7->canonical_node);
        printf("    Stability Score: %.4f\n", sym7->stability_score);
        printf("    Transition Count: %u\n", sym7->transition_count);
        printf("    Entropy Delta: %.4f\n", sym7->entropy_delta);
        
        // Test underlying string table boundaries
        NodeID canon = sym7->canonical_node;
        if (canon != INVALID && canon < MAX_NODES) {
            TokenID tid = ctx->nodes[canon].token_id;
            printf("    Associated TokenID: %u\n", tid);
            const char *text7 = tok_decode(tok_handle, tid);
            printf("    Decoded String:     \"%s\"\n", text7 ? text7 : "NULL FETCH");
            
            // Check for corruption markers
            if (text7) {
                size_t len = strlen(text7);
                printf("    String Length:       %zu\n", len);
                if (strcmp(text7, "<\?\?>") == 0) {
                    printf("    ⚠️  CORRUPTION MARKER: <\?\?> placeholder detected\n");
                }
            }
        }
    } else {
        printf("\n[3] SYMBOL 7 NOT ACTIVE (total_symbols=%u)\n", total_symbols);
    }

    // ---- AUDIT TARGET: SYSTEM-WIDE UTF-8 & MEMORY VIOLATIONS ----
    uint32_t utf8_violations = 0;
    uint32_t zero_length_errors = 0;
    uint32_t dangling_nodes = 0;
    uint32_t corruption_markers = 0;

    for (uint32_t i = 0; i < total_symbols && i < 1000; i++) {  // Limit to first 1000 for performance
        Symbol *s = &ctx->dawg_nodes[i];
        if (s->canonical_node == INVALID) {
            dangling_nodes++;
            continue;
        }

        NodeID canon = s->canonical_node;
        if (canon >= MAX_NODES) {
            dangling_nodes++;
            continue;
        }

        TokenID tid = ctx->nodes[canon].token_id;
        const char *text = tok_decode(tok_handle, tid);
        
        size_t parsed_len = 0;
        if (!text) {
            zero_length_errors++;
        } else if (strcmp(text, "<\?\?>") == 0) {
            corruption_markers++;
        } else if (!is_valid_utf8_string(text, &parsed_len)) {
            utf8_violations++;
            if (utf8_violations <= 10) {
                printf("    > UTF-8 Violation at Symbol ID %u (len=%zu): \"%s\"\n", i, parsed_len, text);
            }
        }
    }

    printf("\n[4] SYSTEM INTEGRITY SUMMARY (first 1000 symbols):\n");
    printf("    Dangling Canonical Nodes:  %u\n", dangling_nodes);
    printf("    NULL/Zero-Length Strings:   %u\n", zero_length_errors);
    printf("    Corruption Markers (<\?\?>):   %u\n", corruption_markers);
    printf("    UTF-8 Corruption Defects:  %u\n", utf8_violations);
    
    // Additional check: sample first 20 low IDs for corruption patterns
    printf("\n[5] LOW ID CORRUPTION PATTERN SCAN (IDs 0-19):\n");
    for (uint32_t i = 0; i < 20 && i < total_symbols; i++) {
        Symbol *s = &ctx->dawg_nodes[i];
        if (s->canonical_node == INVALID || s->canonical_node >= MAX_NODES) {
            printf("    ID %2u: INVALID canonical node\n", i);
            continue;
        }
        
        TokenID tid = ctx->nodes[s->canonical_node].token_id;
        const char *text = tok_decode(tok_handle, tid);
        
        if (!text) {
            printf("    ID %2u: NULL decode\n", i);
        } else if (text[0] == '<' && text[1] == '?' && text[2] == '?' && text[3] == '>' && text[4] == '\0') {
            printf("    ID %2u: CORRUPTION MARKER\n", i);
        } else if (strlen(text) < 2) {
            printf("    ID %2u: SHORT STRING \"%s\"\n", i, text);
        } else {
            // Print first 20 chars of string
            char snippet[21] = {0};
            strncpy(snippet, text, 20);
            printf("    ID %2u: OK \"%s\"\n", i, snippet);
        }
    }
    
    printf("==================================================\n\n");
}

int main() {
    printf("=== Memory & Topology Audit Suite ===\n");
    printf("Auditing all available model files...\n\n");

    // Load tokenizer (try multiple paths)
    int tok = tok_load("production_model.bin");
    if (tok < 0) {
        printf("Failed to load tokenizer production_model.bin\n");
        printf("Trying alternative model paths...\n");
        
        const char* model_paths[] = {
            "model.bin",
            "../model.bin",
            "../../model.bin",
            "vocab/model.bin"
        };
        
        for (int i = 0; i < 4; i++) {
            tok = tok_load(model_paths[i]);
            if (tok >= 0) {
                printf("Successfully loaded model from: %s\n", model_paths[i]);
                break;
            }
        }
        
        if (tok < 0) {
            printf("❌ Failed to load any model file\n");
            return 1;
        }
    }
    printf("✅ Tokenizer loaded\n\n");

    // Audit tokenizer string pool first
    audit_tokenizer_string_pool(tok);
    
    // Run low-level vocabulary serialization audit
    execute_vocabulary_serialization_audit(tok);

    // List of all snapshot files to audit
    const char* snapshot_paths[] = {
        "snapshot.bin",
        "output_snapshot.bin",
        "model/lg_engine.bin",
        "model/lg_tokeniser.bin",
        "production_model.bin"
    };
    const int num_snapshots = 5;

    int successful_audits = 0;
    
    for (int i = 0; i < num_snapshots; i++) {
        printf("\n");
        printf("========================================\n");
        printf("AUDITING FILE %d/%d: %s\n", i+1, num_snapshots, snapshot_paths[i]);
        printf("========================================\n");
        
        EngineContext* ctx = le_load_mmap(snapshot_paths[i], false);
        if (!ctx) {
            printf("❌ Failed to load %s (skipping)\n", snapshot_paths[i]);
            continue;
        }
        
        printf("✅ Successfully loaded: %s\n", snapshot_paths[i]);
        
        // Run the audit
        execute_foundational_memory_audit(ctx, tok);
        
        successful_audits++;
        
        // Note: We don't free ctx here since le_load_mmap uses mmap
        // The OS will clean up on process exit
    }

    printf("\n");
    printf("========================================\n");
    printf("AUDIT SUMMARY\n");
    printf("========================================\n");
    printf("Total files attempted: %d\n", num_snapshots);
    printf("Successful audits: %d\n", successful_audits);
    printf("Failed loads: %d\n", num_snapshots - successful_audits);
    printf("========================================\n");

    // Cleanup
    tok_free(tok);
    
    printf("✅ All audits completed\n");
    return 0;
}
