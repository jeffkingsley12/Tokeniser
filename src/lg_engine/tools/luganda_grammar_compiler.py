#!/usr/bin/env python3
"""
Luganda Grammar Rule Compiler

Converts human-readable YAML grammar rules into packed CSR JSON format.
Maps linguistic tags to 64-bit packed bitmasks for the CSR hypergraph.

Usage:
    python luganda_grammar_compiler.py rules.yaml vocab.txt output.json
"""

import yaml
import json
import sys
from pathlib import Path

# ============================================================================
# Luganda Morphological Bit Allocation (20-bit blocks)
# ============================================================================

# Bits 0-9: Noun Classes (MU-BA, KI-BI, LI-MA, etc.)
NOUN_CLASSES = {
    "CLASS_1": 0x001,   # MU - singular human
    "CLASS_2": 0x002,   # BA - plural human
    "CLASS_3": 0x004,   # N - singular plant/animal
    "CLASS_4": 0x008,   # N - plural plant/animal
    "CLASS_5": 0x010,   # RI/KA - singular object
    "CLASS_6": 0x020,   # ZI/MA - plural object
    "CLASS_7": 0x040,   # KI - singular object
    "CLASS_8": 0x080,   # BI - plural object
    "CLASS_9": 0x100,   # N - singular abstract
    "CLASS_10": 0x200,  # N - plural abstract
}

# Bits 10-13: Locatives
LOCATIVES = {
    "LOC_CLASS_16": 0x400,   # WA - specific location
    "LOC_CLASS_17": 0x800,   # KU - upon/surface
    "LOC_CLASS_18": 0x1000,  # MU - inside
    "LOC_CLASS_19": 0x2000,  # KU - general location
}

# Bits 14-17: Tense/Aspect
TENSE_ASPECT = {
    "TENSE_PAST": 0x4000,
    "TENSE_PRESENT": 0x8000,
    "TENSE_FUTURE": 0x10000,
    "ASPECT_PERFECTIVE": 0x20000,
}

# Bits 18-19: Negation & Modality
NEGATION_MODALITY = {
    "NEGATION_ACTIVE": 0x40000,    # te-, si-
    "MODALITY_SUBJUNCTIVE": 0x80000,
}

# Combined tag mapping
TAG_MAP = {
    **NOUN_CLASSES,
    **LOCATIVES,
    **TENSE_ASPECT,
    **NEGATION_MODALITY,
}

# Special context markers
CONTEXT_MARKERS = {
    "[START_VERB]": 0xFFFFFFFF,  # Special marker for verb start
}


def pack_rule_mask(require_tags, add_tags, clear_tags):
    """
    Pack three tag lists into a single 64-bit rule_mask.
    
    Layout:
    - Bits 0-19:   Require mask (constraints candidate must see)
    - Bits 20-39:  Add mask (constraints candidate injects)
    - Bits 40-59:  Clear mask (constraints candidate consumes/resolves)
    - Bits 60-63:  Reserved
    
    Args:
        require_tags: List of tag strings (e.g., ["CLASS_2"])
        add_tags: List of tag strings (e.g., ["CLASS_7"])
        clear_tags: List of tag strings (e.g., ["CLASS_2"])
    
    Returns:
        uint64_t packed mask value
    """
    require_mask = 0
    for tag in require_tags:
        if tag in TAG_MAP:
            require_mask |= TAG_MAP[tag]
        else:
            print(f"Warning: Unknown require tag '{tag}'", file=sys.stderr)
    
    add_mask = 0
    for tag in add_tags:
        if tag in TAG_MAP:
            add_mask |= TAG_MAP[tag]
        else:
            print(f"Warning: Unknown add tag '{tag}'", file=sys.stderr)
    
    clear_mask = 0
    for tag in clear_tags:
        if tag in TAG_MAP:
            clear_mask |= TAG_MAP[tag]
        else:
            print(f"Warning: Unknown clear tag '{tag}'", file=sys.stderr)
    
    # Pack into 64-bit: (clear << 40) | (add << 20) | require
    packed = (clear_mask << 40) | (add_mask << 20) | require_mask
    return packed


def resolve_token_id(token_str, vocab_dict, context_markers):
    """
    Resolve a token string to a Token ID.
    
    Args:
        token_str: Token string (e.g., "ba" or "[START_VERB]")
        vocab_dict: Dictionary mapping token strings to IDs
        context_markers: Dictionary of special context markers
    
    Returns:
        Token ID (uint32_t)
    """
    if token_str in context_markers:
        return context_markers[token_str]
    elif token_str in vocab_dict:
        return vocab_dict[token_str]
    else:
        print(f"Warning: Token '{token_str}' not found in vocabulary", file=sys.stderr)
        return 0  # Use 0 as sentinel for unknown tokens


def load_vocab(vocab_path):
    """
    Load vocabulary file (format: token<space>id per line).
    
    Args:
        vocab_path: Path to vocabulary file
    
    Returns:
        Dictionary mapping token strings to IDs
    """
    vocab = {}
    with open(vocab_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                token = ' '.join(parts[:-1])  # Handle multi-word tokens
                token_id = int(parts[-1])
                vocab[token] = token_id
    return vocab


def compile_yaml_to_json(yaml_path, vocab_path, output_path):
    """
    Compile YAML grammar rules to JSON CSR format.
    
    Args:
        yaml_path: Path to input YAML file
        vocab_path: Path to vocabulary file
        output_path: Path to output JSON file
    """
    # Load vocabulary
    print(f"Loading vocabulary from {vocab_path}...")
    vocab = load_vocab(vocab_path)
    print(f"  Loaded {len(vocab)} tokens")
    
    # Load YAML rules
    print(f"Loading grammar rules from {yaml_path}...")
    with open(yaml_path, 'r', encoding='utf-8') as f:
        rules_data = yaml.safe_load(f)
    
    rules = rules_data.get('rules', [])
    print(f"  Loaded {len(rules)} rules")
    
    # Compile rules
    compiled_rules = []
    for rule in rules:
        candidate_str = rule.get('candidate', '')
        context_str = rule.get('context', '')
        require_tags = rule.get('require', [])
        add_tags = rule.get('add', [])
        clear_tags = rule.get('clear', [])
        modifier = rule.get('modifier', 1.0)
        
        # Resolve token IDs
        candidate_id = resolve_token_id(candidate_str, vocab, CONTEXT_MARKERS)
        context_id = resolve_token_id(context_str, vocab, CONTEXT_MARKERS)
        
        # Pack rule mask
        packed_mask = pack_rule_mask(require_tags, add_tags, clear_tags)
        
        compiled_rule = {
            "candidate_token": candidate_id,
            "context_token": context_id,
            "modifier": modifier,
            "rule_mask": packed_mask
        }
        compiled_rules.append(compiled_rule)
        
        print(f"  Rule: {candidate_str} <- {context_str} "
              f"(require={require_tags}, add={add_tags}, clear={clear_tags}) "
              f"-> mask=0x{packed_mask:016x}")
    
    # Determine vocab size (max token ID + 1)
    max_token_id = max([r['candidate_token'] for r in compiled_rules] +
                       [r['context_token'] for r in compiled_rules if r['context_token'] != 0xFFFFFFFF])
    vocab_size = max(max_token_id + 1, len(vocab))
    
    # Build output JSON
    output = {
        "vocab_size": vocab_size,
        "rules": compiled_rules
    }
    
    # Write output
    print(f"Writing compiled rules to {output_path}...")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2)
    
    print(f"  Compiled {len(compiled_rules)} rules")
    print(f"  Vocab size: {vocab_size}")
    print("Done!")


def main():
    if len(sys.argv) != 4:
        print("Usage: python luganda_grammar_compiler.py <rules.yaml> <vocab.txt> <output.json>")
        print("\nExample:")
        print("  python luganda_grammar_compiler.py luganda_rules.yaml vocab.txt luganda_grammar_rules.json")
        sys.exit(1)
    
    yaml_path = sys.argv[1]
    vocab_path = sys.argv[2]
    output_path = sys.argv[3]
    
    compile_yaml_to_json(yaml_path, vocab_path, output_path)


if __name__ == "__main__":
    main()
