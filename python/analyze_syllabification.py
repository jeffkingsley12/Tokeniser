#!/usr/bin/env python3
"""
Analyze the syllabification discrepancy between LOUDS and current paths.
Key question: Why does LOUDS produce zeros while current doesn't?
"""

import subprocess
import json
import re

# Test words with issues
test_words = [
    'kanywa-musenke',   # LOUDS: [96, 0, 0, 386, 0, 200, 147, 717]
    'kafumita-bagenge',  # LOUDS: [96, 60, 198, 156, 0, 26, 67, 697]
    'kalumanywere',      # LOUDS: [96, 110, 196, 0, 0, 387, 137]
    'nnakitaddabusa',    # LOUDS: [366, 98, 156, 226, 30, 146] (only match!)
    'sserinyabi',        # LOUDS: [327, 138, 926, 28] (partial match)
    'ssere',             # LOUDS: [327, 137]
    'kanywa',            # Test individual component
    'musenke',           # Test individual component
]

def syllabify_word(word):
    """Use tokenizer_demo to syllabify a word and extract debug info."""
    try:
        result = subprocess.run(
            ['./tokenizer_demo', word],
            capture_output=True,
            text=True,
            timeout=5
        )
        # Parse output for syllable IDs
        output = result.stderr + result.stdout
        
        # Look for patterns like "syllable_id: X" or similar
        ids = re.findall(r'syllable_id:\s*(\d+)', output)
        if ids:
            return [int(x) for x in ids]
        
        # Try parsing from other formats
        ids = re.findall(r'id[=:\s]+(\d+)', output)
        if ids:
            return [int(x) for x in ids]
            
        return None
    except Exception as e:
        print(f"Error syllabifying '{word}': {e}")
        return None

def analyze_discrepancies():
    print("🔍 Syllabification Path Analysis")
    print("=" * 60)
    print("\nTesting if LOUDS path returns zeros due to trie mismatches\n")
    
    for word in test_words:
        print(f"🔤 '{word}'")
        print("-" * 40)
        
        # Check if word contains problematic characters
        has_hyphen = '-' in word
        has_double_consonant = any(c*2 in word for c in 'kgmnpstbdlnjrw')
        
        if has_hyphen:
            print(f"  ⚠️  Contains hyphen: {has_hyphen}")
            parts = word.split('-')
            print(f"     Parts: {parts}")
            
        if has_double_consonant:
            doubles = [c for c in 'kgmnpstbdlnjrw' if c*2 in word]
            print(f"  ⚠️  Contains double consonants: {doubles}")
        
        # Try to get syllabification
        ids = syllabify_word(word)
        if ids:
            print(f"  Syllable IDs: {ids}")
            if 0 in ids:
                zero_positions = [i for i, x in enumerate(ids) if x == 0]
                print(f"  ❌ Contains zeros at positions: {zero_positions}")
        else:
            print(f"  ❌ Could not extract syllable IDs")
        
        print()

def check_trie_mismatch():
    """Check if LOUDS trie might be missing syllables."""
    print("\n📊 Potential Trie Mismatch Analysis")
    print("=" * 60)
    print("\nHypothesis: LOUDS trie is missing certain syllables")
    print("- Single consonants (m, n, etc.) map to ID 0 in debug")
    print("- Hyphen (-) maps to ID 0 in debug")
    print("- Hyphens in compound words (kanywa-musenke) may not be handled")
    print("\nPossible root causes:")
    print("1. Trie doesn't include syllables for hyphens/separators")
    print("2. LOUDS syllabifier doesn't handle multi-word tokens correctly")
    print("3. consume_syllable_id returns 0 for unrecognized syllables")
    print("4. Mismatch between how tokens are added to LOUDS vs current table")

def extract_louds_patterns():
    """Extract patterns from LOUDS debug output."""
    print("\n📈 LOUDS Sequence Patterns")
    print("=" * 60)
    
    louds_data = {
        'kanywa-musenke': [96, 0, 0, 386, 0, 200, 147, 717],
        'kafumita-bagenge': [96, 60, 198, 156, 0, 26, 67, 697],
        'kalumanywere': [96, 110, 196, 0, 0, 387, 137],
        'nnakitaddabusa': [366, 98, 156, 226, 30, 146],
        'sserinyabi': [327, 138, 926, 28],
    }
    
    print("\nTokens with zeros (trie misses):")
    for word, seq in louds_data.items():
        if 0 in seq:
            zero_count = seq.count(0)
            zero_positions = [i for i, x in enumerate(seq) if x == 0]
            parts = word.split('-')
            print(f"  {word:20} → zeros={zero_count} at pos {zero_positions}")
            if len(parts) > 1:
                print(f"                        compound: {parts}")
    
    print("\nTokens without zeros (trie matches):")
    for word, seq in louds_data.items():
        if 0 not in seq:
            print(f"  {word:20} → {seq}")

if __name__ == "__main__":
    # analyze_discrepancies()
    check_trie_mismatch()
    extract_louds_patterns()
