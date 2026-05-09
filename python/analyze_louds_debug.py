#!/usr/bin/env python3
"""
Analyze LOUDS debug output to identify patterns and potential issues
"""

import re
from collections import defaultdict

# Parse the debug output
debug_lines = [
    '[DEBUG-LOUDS-ADD] text="kanywa-musenke" id=1529 seq=[96, 0, 0, 386, 0, 200, 147, 717]',
    '[DEBUG-LOUDS-ADD] text="kafumita-bagenge" id=1530 seq=[96, 60, 198, 156, 0, 26, 67, 697]',
    '[DEBUG-LOUDS-ADD] text="ddoodo" id=1531 seq=[234, 49]',
    '[DEBUG-LOUDS-ADD] text="mmulukulu" id=1532 seq=[380, 110, 100, 110]',
    '[DEBUG-LOUDS-ADD] text="nnakitaddabusa" id=1533 seq=[366, 98, 156, 226, 30, 146]',
    '[DEBUG-LOUDS-ADD] text="nnakati" id=1534 seq=[366, 96, 158]',
    '[DEBUG-LOUDS-ADD] text="nnakasuga" id=1535 seq=[366, 96, 150, 66]',
    '[DEBUG-LOUDS-ADD] text="sseziwundu" id=1536 seq=[327, 178, 390, 690]',
    '[DEBUG-LOUDS-ADD] text="sserinyabi" id=1537 seq=[327, 138, 926, 28]',
    '[DEBUG-LOUDS-ADD] text="ssere" id=1538 seq=[327, 137]',
    '[DEBUG-LOUDS-ADD] text="nnasamba" id=1539 seq=[366, 146, 756]',
    '[DEBUG-LOUDS-ADD] text="ggiri" id=1540 seq=[248, 138]',
    '[DEBUG-LOUDS-ADD] text="kalumanywere" id=1541 seq=[96, 110, 196, 0, 0, 387, 137]'
]

def parse_debug_line(line):
    """Parse a single LOUDS debug line"""
    match = re.match(r'\[DEBUG-LOUDS-ADD\] text="([^"]+)" id=(\d+) seq=\[(.*?)\]', line)
    if match:
        text, token_id, seq_str = match.groups()
        seq = [int(x) for x in seq_str.split(', ')]
        return {
            'text': text,
            'id': int(token_id),
            'seq': seq,
            'length': len(seq)
        }
    return None

def analyze_patterns(tokens):
    """Analyze patterns in the token data"""
    print("🔍 LOUDS Token Addition Analysis")
    print("=" * 50)
    
    # Basic statistics
    print(f"Total tokens analyzed: {len(tokens)}")
    print(f"Token ID range: {tokens[0]['id']} - {tokens[-1]['id']}")
    
    # Sequence length distribution
    length_dist = defaultdict(list)
    for token in tokens:
        length_dist[token['length']].append(token)
    
    print(f"\n📊 Sequence Length Distribution:")
    for length in sorted(length_dist.keys()):
        count = len(length_dist[length])
        examples = [t['text'] for t in length_dist[length][:3]]
        print(f"  {length} syllables: {count} tokens (e.g., {examples})")
    
    # Analyze zero syllables (potential issues)
    zero_syllable_tokens = [t for t in tokens if 0 in t['seq']]
    print(f"\n⚠️  Tokens with zero syllables: {len(zero_syllable_tokens)}")
    for token in zero_syllable_tokens:
        zero_positions = [i for i, syl in enumerate(token['seq']) if syl == 0]
        print(f"  {token['text']} (id={token['id']}): zeros at positions {zero_positions}")
    
    # Prefix analysis
    print(f"\n🔤 Prefix Patterns:")
    prefixes = defaultdict(list)
    for token in tokens:
        if len(token['seq']) >= 2:
            prefix = tuple(token['seq'][:2])
            prefixes[prefix].append(token)
    
    common_prefixes = sorted(prefixes.items(), key=lambda x: len(x[1]), reverse=True)[:5]
    for prefix, tokens_list in common_prefixes:
        if len(tokens_list) > 1:
            examples = [t['text'] for t in tokens_list[:3]]
            print(f"  Prefix {prefix}: {len(tokens_list)} tokens (e.g., {examples})")
    
    # Character patterns
    print(f"\n🔤 Character Patterns in Text:")
    double_consonants = [t for t in tokens if any(c*2 in t['text'] for c in 'kgmnps')]
    print(f"  Double consonants: {len(double_consonants)} tokens")
    for token in double_consonants:
        doubles = [c*2 for c in 'kgmnps' if c*2 in token['text']]
        print(f"    {token['text']}: {doubles}")
    
    hyphenated = [t for t in tokens if '-' in t['text']]
    print(f"  Hyphenated tokens: {len(hyphenated)}")
    for token in hyphenated:
        parts = token['text'].split('-')
        print(f"    {token['text']}: {parts}")

# Parse and analyze
tokens = []
for line in debug_lines:
    parsed = parse_debug_line(line)
    if parsed:
        tokens.append(parsed)

if tokens:
    analyze_patterns(tokens)
else:
    print("No valid debug lines parsed")
