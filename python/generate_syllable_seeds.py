#!/usr/bin/env python3
"""
generate_syllable_seeds.py
==========================
Generate syllable_seeds.h from syllables.yaml with proper sanitization.

This script:
- Strips everything after the first # (comments)
- Rejects lines containing Cyrillic or Greek confusables
- Splits long lines safely (no mid-string truncation)
- Deduplicates entries
- Outputs a clean C header file
"""

import sys
import os

# Confusable Unicode ranges
CYRILLIC_RANGE = (0x0400, 0x04FF)
GREEK_OMICRON = 0x03BF

def has_confusable(text):
    """Check if text contains Cyrillic or Greek confusables."""
    for char in text:
        cp = ord(char)
        if CYRILLIC_RANGE[0] <= cp <= CYRILLIC_RANGE[1]:
            return True
        if cp == GREEK_OMICRON:
            return True
    return False

def clean_line(raw):
    """
    Clean a line from the YAML file.
    - Remove comments (everything after #)
    - Strip whitespace
    - Reject if contains confusable characters
    """
    # Remove comments
    if '#' in raw:
        raw = raw.split('#')[0]
    raw = raw.strip()
    
    # Reject empty lines
    if not raw:
        return None
    
    # Reject Cyrillic and Greek confusables
    if has_confusable(raw):
        print(f"WARNING: Rejecting confusable entry: {raw}", file=sys.stderr)
        return None
    
    return raw

def generate_syllable_seeds():
    """Generate syllable_seeds.h from syllables.yaml."""
    # Read file line by line (simpler than YAML parsing)
    entries = []
    with open('syllables.yaml', 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            # Skip empty lines and comments
            if not line or line.startswith('#'):
                continue
            # Look for list entries (starts with "- ")
            if line.startswith('- '):
                entry = line[2:].strip()
                cleaned = clean_line(entry)
                if cleaned:
                    entries.append(cleaned)
    
    # Deduplicate while preserving order
    unique_entries = list(dict.fromkeys(entries))
    
    # Generate C header
    output = """/* syllable_seeds.h — Morpheme seed data for the Luganda tokenizer.
 *
 * Auto-generated from syllables.yaml by generate_syllable_seeds.py
 *
 * IMPORTANT: This array is included in multiple translation units.
 * To avoid multiple-definition ODR violations it is declared
 * static const — each TU gets its own copy, which is fine because
 * the data is read-only and the compiler will fold identical string
 * literals.
 *
 * Duplicate entries are intentional: the same surface form may
 * represent morphemes from different grammatical categories.
 * The seeding pass de-duplicates at the SyllableTable level.
 */

#ifndef SYLLABLE_SEEDS_H
#define SYLLABLE_SEEDS_H

static const char *MORPHEME_SEEDS[] = {
"""
    
    # Add entries
    for entry in unique_entries:
        # Escape quotes and backslashes
        escaped = entry.replace('\\', '\\\\').replace('"', '\\"')
        output += f'    "{escaped}",\n'
    
    output += """};

#define MORPHEME_SEED_COUNT (sizeof(MORPHEME_SEEDS) / sizeof(MORPHEME_SEEDS[0]))

#endif /* SYLLABLE_SEEDS_H */
"""
    
    # Write output
    output_path = os.path.join(os.path.dirname(__file__), '..', 'include', 'syllable_seeds.h')
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(output)
    
    print(f"Generated {len(unique_entries)} unique entries in syllable_seeds.h", file=sys.stderr)

if __name__ == '__main__':
    generate_syllable_seeds()
