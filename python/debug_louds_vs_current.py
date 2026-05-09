#!/usr/bin/env python3
"""
Compare LOUDS debug output with current syllable table to identify discrepancies
"""

# Original LOUDS debug data
louds_tokens = [
    {'text': 'kanywa-musenke', 'id': 1529, 'seq': [96, 0, 0, 386, 0, 200, 147, 717]},
    {'text': 'kafumita-bagenge', 'id': 1530, 'seq': [96, 60, 198, 156, 0, 26, 67, 697]},
    {'text': 'ddoodo', 'id': 1531, 'seq': [234, 49]},
    {'text': 'mmulukulu', 'id': 1532, 'seq': [380, 110, 100, 110]},
    {'text': 'nnakitaddabusa', 'id': 1533, 'seq': [366, 98, 156, 226, 30, 146]},
    {'text': 'nnakati', 'id': 1534, 'seq': [366, 96, 158]},
    {'text': 'nnakasuga', 'id': 1535, 'seq': [366, 96, 150, 66]},
    {'text': 'sseziwundu', 'id': 1536, 'seq': [327, 178, 390, 690]},
    {'text': 'sserinyabi', 'id': 1537, 'seq': [327, 138, 926, 28]},
    {'text': 'ssere', 'id': 1538, 'seq': [327, 137]},
    {'text': 'nnasamba', 'id': 1539, 'seq': [366, 146, 756]},
    {'text': 'ggiri', 'id': 1540, 'seq': [248, 138]},
    {'text': 'kalumanywere', 'id': 1541, 'seq': [96, 110, 196, 0, 0, 387, 137]}
]

# Current syllabification results (from debug_syllable_ids output)
current_tokens = {
    'kanywa-musenke': [332, 261, 622, 45, 436, 383, 953],
    'kafumita-bagenge': [332, 296, 434, 392, 45, 262, 303, 933],
    'kalumanywere': [332, 346, 432, 261, 623, 373],
    'ddoodo': [470, 285],
    'mmulukulu': [616, 346, 336, 346],
    'nnakitaddabusa': [366, 98, 156, 226, 30, 146],
    'nnakati': [602, 332, 394],
    'nnakasuga': [602, 332, 386, 302],
    'sseziwundu': [563, 414, 626, 926],
    'sserinyabi': [563, 138, 926, 28],
    'ssere': [563, 373],
    'nnasamba': [602, 382, 992],
    'ggiri': [484, 374]
}

def analyze_discrepancies():
    print("🔍 LOUDS vs Current Syllable ID Analysis")
    print("=" * 50)
    
    print("\n📊 Token-by-Token Comparison:")
    print("-" * 40)
    
    for louds_token in louds_tokens:
        text = louds_token['text']
        louds_seq = louds_token['seq']
        
        if text in current_tokens:
            current_seq = current_tokens[text]
            
            print(f"\n🔤 {text} (ID: {louds_token['id']})")
            print(f"  LOUDS seq:     {louds_seq}")
            print(f"  Current seq:   {current_seq}")
            
            # Check for differences
            if len(louds_seq) != len(current_seq):
                print(f"  ⚠️  Length mismatch: LOUDS={len(louds_seq)}, Current={len(current_seq)}")
            
            # Count zeros in LOUDS
            louds_zeros = louds_seq.count(0)
            current_zeros = current_seq.count(0)
            
            if louds_zeros > 0:
                zero_positions = [i for i, x in enumerate(louds_seq) if x == 0]
                print(f"  ❌ LOUDS has {louds_zeros} zeros at positions {zero_positions}")
            
            if current_zeros > 0:
                zero_positions = [i for i, x in enumerate(current_seq) if x == 0]
                print(f"  ❌ Current has {current_zeros} zeros at positions {zero_positions}")
            
            # Check for matching elements
            matches = 0
            min_len = min(len(louds_seq), len(current_seq))
            for i in range(min_len):
                if louds_seq[i] == current_seq[i] and louds_seq[i] != 0:
                    matches += 1
            
            if matches > 0:
                print(f"  ✅ {matches} matching non-zero syllable IDs")
            else:
                print(f"  ❌ No matching syllable IDs")
        else:
            print(f"\n🔤 {text} (ID: {louds_token['id']})")
            print(f"  ❌ Not found in current tokens")
    
    print(f"\n📈 Summary Statistics:")
    print("-" * 30)
    
    total_louds_tokens = len(louds_tokens)
    total_current_tokens = len(current_tokens)
    
    louds_zero_count = sum(seq.count(0) for seq in [t['seq'] for t in louds_tokens])
    current_zero_count = sum(seq.count(0) for seq in current_tokens.values())
    
    print(f"LOUDS tokens analyzed: {total_louds_tokens}")
    print(f"Current tokens: {total_current_tokens}")
    print(f"LOUDS zero syllables: {louds_zero_count}")
    print(f"Current zero syllables: {current_zero_count}")
    
    # Analyze ID ranges
    all_louds_ids = []
    all_current_ids = []
    
    for token in louds_tokens:
        all_louds_ids.extend([x for x in token['seq'] if x != 0])
    
    for seq in current_tokens.values():
        all_current_ids.extend([x for x in seq if x != 0])
    
    if all_louds_ids:
        print(f"LOUDS ID range: {min(all_louds_ids)} - {max(all_louds_ids)}")
    
    if all_current_ids:
        print(f"Current ID range: {min(all_current_ids)} - {max(all_current_ids)}")
    
    # Check for ID overlap
    if all_louds_ids and all_current_ids:
        overlap = set(all_louds_ids) & set(all_current_ids)
        print(f"ID overlap: {len(overlap)} syllables")
        
        if overlap:
            print(f"Common IDs: {sorted(list(overlap))[:10]}{'...' if len(overlap) > 10 else ''}")

if __name__ == "__main__":
    analyze_discrepancies()
