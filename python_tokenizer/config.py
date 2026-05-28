"""
Configuration constants for the Luganda Tokenizer.
Ported from include/tokenizer_config.h and include/tokenizer.h
"""

# Special token IDs (must match C implementation)
TOK_UNK = 0   # <unk> - Unknown token
TOK_BOS = 1   # <s> - Beginning of sequence
TOK_EOS = 2   # </s> - End of sequence
TOK_PAD = 3   # <pad> - Padding token
TOK_MASK = 4  # [MASK] - Mask token for MLM

SPECIAL_TOKENS_COUNT = 5
SPECIAL_TOKENS = ["<unk>", "<s>", "</s>", "<pad>", "[MASK]"]

# Syllable table configuration
BASE_SYMBOL_OFFSET = 4096  # Max syllables before byte range
MAX_SYLLABLES = BASE_SYMBOL_OFFSET
MAX_TOKEN_CHARS = 64
# NOTE: The following constants are kept for reference but should not be
# relied upon until the C API exposes them properly:
MAX_SEQ_LEN = 1 << 24        # 16,777,216 - matches C MAX_SEQ_LEN
MAX_RULE_DEPTH = 128

# File format magic numbers - matches C tokenizer_mmap.c format
MODEL_MAGIC = b"LGMMAPv1"     # 8-byte header magic (not a uint32!)
MODEL_VERSION_MIN = 14        # Supported model format versions
MODEL_VERSION_MAX = 15

# Luganda phonotactic constraints
VOWELS = set('aeiou')
CONSONANTS = set('bcdfghjklmnpqrstvwxyz')
VALID_ONSETS = {
    # Single consonants
    'b', 'c', 'd', 'f', 'g', 'j', 'k', 'l', 'm', 'n', 
    'p', 'r', 's', 't', 'v', 'w', 'y', 'z',
    # Common clusters in Luganda
    'bw', 'by', 'cw', 'cy', 'dw', 'dy', 'fw', 'fy',
    'gw', 'gy', 'jw', 'jy', 'kw', 'ky', 'lw', 'ly',
    'mw', 'my', 'nw', 'ny', 'pw', 'py', 'rw', 'ry',
    'sw', 'sy', 'tw', 'ty', 'vw', 'vy', 'zw', 'zy',
    # Prenasalized clusters
    'mb', 'mbw', 'mby', 'mf', 'mv',
    'nd', 'ndw', 'ndy', 'ns', 'nz', 'nj',
    'ng', 'ngw', 'ngy', 'nk', 'nkw', 'nky', 'nt', 'nv',
    'nc', 'nny', 'nn', 'njw', 'njy', 'nsw', 'nsy',
    'nzw', 'nzy', 'nkwi', 'nkwu', 'nkwe', 'nkye',
}

# UTF-8 handling
REPLACEMENT_CHAR = '\ufffd'
MAX_UTF8_BYTES = 4

# Performance tuning
# NOTE: DEFAULT_NUM_WORKERS and BATCH_SIZE are implementation hints only
# (not enforced by the C library)
