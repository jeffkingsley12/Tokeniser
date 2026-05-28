import os
import re

# Keywords for formal and colloquial scoring
FORMAL_KEYWORDS = {
    "katikkiro", "kabaka", "owek", "minisita", "gavumenti", "palamenti", "supreme", 
    "lutikko", "ssaabasajja", "ebitongole", "enkulaakulana", "ebiwandiiko", 
    "amatendekero", "ssentendekero", "pookino", "ssekabaka", "omulangira", 
    "omubaka", "mmeeya", "bulange", "mmengo", "ennukuta", "ennyingo", "ebigambo", 
    "ensirifu", "empeeerezi", "yebazizza", "akakasizza", "kikuzizza", "ategeezezza", 
    "bwayinza", "asabye", "yategeezezza", "kitongole", "kyasaliddwawo", "bakolagana", 
    "byobwannakyeewa", "obwakabaka", "omuterrega", "ssemasonga", "kamalabyonna", 
    "diocese", "matikkira"
}

COLLOQUIAL_KEYWORDS = {
    "bambi", "mbu", "dala", "dalala", "banange", "boobo", "love", "baby", "selfie", 
    "bro", "whatsapp", "oga", "legend", "betting", "kazino", "jokers", "circus", 
    "haters", "smoking", "drugs", "tiktok", "likes", "post", "page", "jackpot", 
    "bet", "urgent", "2k", "comedy", "store", "program", "photo", "vest", "neiba",
    "selfie", "biffa", "aliira", "nsiko", "mbu", "clown", "joker", "facebook",
    "bwenkanya", "kiwaana", "balaaba", "bannayuganda", "okwesiga", "selfie", "hijab"
}

def clean_words(text):
    # Split by whitespace and remove punctuation
    words = re.findall(r'\b\w+\b', text.lower())
    return words

def partition_corpus(input_path):
    print(f"Reading corpus from {input_path}...")
    
    formal_lines = []
    colloquial_lines = []
    neutral_lines = []
    
    with open(input_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            line_str = line.strip()
            if not line_str:
                continue
            
            words = clean_words(line_str)
            if not words:
                continue
            
            # Count word lengths
            word_count = len(words)
            
            # Scores
            f_score = sum(1 for w in words if w in FORMAL_KEYWORDS)
            c_score = sum(1 for w in words if w in COLLOQUIAL_KEYWORDS)
            
            # Punctuation styling score (colloquial uses lots of ?, !, emojis)
            punctuation_noise = line_str.count('!') + line_str.count('?')
            c_score += min(punctuation_noise, 5)
            
            # Calculate final rating
            rating = f_score - c_score
            
            if rating > 0:
                formal_lines.append((rating, word_count, line_str))
            elif rating < 0:
                colloquial_lines.append((rating, word_count, line_str))
            else:
                neutral_lines.append((0, word_count, line_str))

    # Sort formal lines descending (highest rating first)
    formal_lines.sort(key=lambda x: x[0], reverse=True)
    # Sort colloquial lines ascending (most negative rating first)
    colloquial_lines.sort(key=lambda x: x[0])

    print(f"Initial split: {len(formal_lines)} formal, {len(colloquial_lines)} colloquial, {len(neutral_lines)} neutral lines.")

    # Target 500,000 words
    TARGET_WORDS = 500000
    
    # Extract formal text
    selected_formal = []
    formal_words = 0
    for rating, wc, text in formal_lines:
        selected_formal.append(text)
        formal_words += wc
        if formal_words >= TARGET_WORDS:
            break
            
    # If not enough formal lines, add from neutral
    if formal_words < TARGET_WORDS:
        for rating, wc, text in neutral_lines:
            selected_formal.append(text)
            formal_words += wc
            if formal_words >= TARGET_WORDS:
                break
                
    # Extract colloquial text
    selected_colloquial = []
    colloquial_words = 0
    for rating, wc, text in colloquial_lines:
        selected_colloquial.append(text)
        colloquial_words += wc
        if colloquial_words >= TARGET_WORDS:
            break
            
    # If not enough colloquial lines, add from neutral
    if colloquial_words < TARGET_WORDS:
        for rating, wc, text in neutral_lines:
            selected_colloquial.append(text)
            colloquial_words += wc
            if colloquial_words >= TARGET_WORDS:
                break
                
    print(f"Formal corpus size: {len(selected_formal)} lines, {formal_words} words.")
    print(f"Colloquial corpus size: {len(selected_colloquial)} lines, {colloquial_words} words.")
    
    with open("formal_corpus.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(selected_formal) + "\n")
        
    with open("colloquial_corpus.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(selected_colloquial) + "\n")

    print("Corpus files formal_corpus.txt and colloquial_corpus.txt written successfully.")

if __name__ == "__main__":
    corpus_file = "luganda_corpus.txt"
    if not os.path.exists(corpus_file):
        corpus_file = "luganda_corpus_sub.txt"
    partition_corpus(corpus_file)
