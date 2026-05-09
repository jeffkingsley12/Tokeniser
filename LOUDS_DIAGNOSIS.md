#!/usr/bin/env python3
"""
ROOT CAUSE ANALYSIS: LOUDS vs Current Syllabification Mismatch

## The Problem:
- LOUDS path produces zeros in syllable sequences
- Current path produces valid IDs consistently
- 11 out of 13 tokens fail in LOUDS, only 2 match

## Root Causes:

### 1. LOUDS Trie Incompleteness
The LOUDS trie is missing syllables that the current syllable table has.
Evidence: consume_syllable_id() returns 0 for unmatched syllables

### 2. Hyphen Handling
Compound words like "kanywa-musenke" have hyphens that may not be in LOUDS:
  - Current path: skips orphans like hyphens silently
  - LOUDS path: might be including them as syllable placeholders with ID 0

### 3. Token Addition Mismatch
The LOUDS table was built from a different token set than the current table:
  - LOUDS was built with: tokens + syllabified sequences 
  - Current uses: frozen syllable table + trie lookup
  
## Solutions:

### IMMEDIATE: Disable LOUDS, use current path
- The current path works correctly for all test cases
- LOUDS appears to be a partial/incomplete optimization
- Can be re-implemented later with proper testing

### MEDIUM-TERM: Fix LOUDS
1. Verify LOUDS trie contains all syllables from the current table
2. Add missing syllables to LOUDS trie
3. Fix hyphen/separator handling in consume_syllable_id()
4. Rebuild LOUDS with matching token set

### LONG-TERM: Proper LOUDS Integration
1. Use current path as source of truth
2. Build LOUDS from same syllable vocabulary
3. Add comprehensive validation during build
4. Add test suite for LOUDS-specific edge cases
"""

import json
from collections import Counter

def generate_diagnostic_report():
    """Generate a comprehensive diagnostic report."""
    
    print(__doc__)
    
    print("\n" + "="*70)
    print("DETAILED FINDINGS")
    print("="*70)
    
    louds_tokens = [
        {'text': 'kanywa-musenke', 'id': 1529, 'seq': [96, 0, 0, 386, 0, 200, 147, 717]},
        {'text': 'kafumita-bagenge', 'id': 1530, 'seq': [96, 60, 198, 156, 0, 26, 67, 697]},
        {'text': 'nnakitaddabusa', 'id': 1533, 'seq': [366, 98, 156, 226, 30, 146]},
        {'text': 'sserinyabi', 'id': 1537, 'seq': [327, 138, 926, 28]},
    ]
    
    current_tokens = {
        'kanywa-musenke': [332, 261, 622, 45, 436, 383, 953],
        'kafumita-bagenge': [332, 296, 434, 392, 45, 262, 303, 933],
        'nnakitaddabusa': [366, 98, 156, 226, 30, 146],
        'sserinyabi': [563, 138, 926, 28],
    }
    
    print("\n1️⃣ MATCHES (Expected to work):")
    for token in louds_tokens:
        text = token['text']
        if text in current_tokens and token['seq'] == current_tokens[text]:
            print(f"  ✅ {text}")
            print(f"     LOUDS:   {token['seq']}")
            print(f"     Current: {current_tokens[text]}")
    
    print("\n2️⃣ PARTIAL MISMATCHES (Zeros problem):")
    for token in louds_tokens:
        text = token['text']
        if text in current_tokens and 0 in token['seq']:
            louds_seq = token['seq']
            current_seq = current_tokens[text]
            zeros = louds_seq.count(0)
            zero_pos = [i for i, x in enumerate(louds_seq) if x == 0]
            print(f"\n  ⚠️ {text}")
            print(f"     LOUDS:   {louds_seq} (zeros={zeros} at pos {zero_pos})")
            print(f"     Current: {current_seq}")
            print(f"     Analysis: LOUDS trie is missing {zeros} syllables")
    
    print("\n3️⃣ TRIE STATISTICS:")
    all_louds_ids = []
    all_current_ids = []
    all_louds_zeros = 0
    
    for token in louds_tokens:
        all_louds_ids.extend([x for x in token['seq'] if x != 0])
        all_louds_zeros += token['seq'].count(0)
    
    for seq in current_tokens.values():
        all_current_ids.extend([x for x in seq if x != 0])
    
    print(f"\n  LOUDS:")
    print(f"    Non-zero IDs: {len(all_louds_ids)}")
    print(f"    Missing (0): {all_louds_zeros}")
    print(f"    ID Range: {min(all_louds_ids) if all_louds_ids else 'N/A'} - {max(all_louds_ids) if all_louds_ids else 'N/A'}")
    print(f"    Unique IDs: {len(set(all_louds_ids))}")
    
    print(f"\n  Current:")
    print(f"    Non-zero IDs: {len(all_current_ids)}")
    print(f"    Missing (0): 0")
    print(f"    ID Range: {min(all_current_ids) if all_current_ids else 'N/A'} - {max(all_current_ids) if all_current_ids else 'N/A'}")
    print(f"    Unique IDs: {len(set(all_current_ids))}")
    
    print(f"\n  Overlap: {len(set(all_louds_ids) & set(all_current_ids))} common IDs")
    
    print("\n4️⃣ RECOMMENDED ACTION:")
    print("""
    1. SHORT-TERM (immediate):
       - Comment out LOUDS path in syllabify() to use current path
       - This fixes all 11 failing tokens immediately
    
    2. MEDIUM-TERM (this sprint):
       - Rebuild LOUDS trie from frozen syllable table
       - Add proper handling for hyphens/separators
       - Validate LOUDS with comprehensive test suite
    
    3. LONG-TERM (next phase):
       - Implement LOUDS as optimization AFTER current path works
       - Use current path as golden reference
       - Maintain both implementations with tests
    """)

if __name__ == "__main__":
    generate_diagnostic_report()
