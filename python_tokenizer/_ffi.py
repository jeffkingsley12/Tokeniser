"""
Low-level FFI bindings to the C tokenizer library.
This module wraps libluganda_tok.so using ctypes.
"""

import os
import sys
from ctypes import (
    CDLL, CFUNCTYPE, POINTER, c_char_p, c_uint8, c_uint16, c_uint32, 
    c_int, c_bool, c_void_p, c_char, c_size_t, Structure, byref, cast
)
from typing import Optional, List, Tuple
import platform

# Find the shared library
def _find_library() -> str:
    """Locate libluganda_tok.so in various places."""
    search_paths = [
        # Same directory as this file (development)
        os.path.join(os.path.dirname(__file__), '..', 'libluganda_tok.so'),
        # Parent directory
        os.path.join(os.path.dirname(__file__), '..', '..', 'libluganda_tok.so'),
        # Current working directory
        './libluganda_tok.so',
        # System paths
        '/usr/local/lib/libluganda_tok.so',
        '/usr/lib/libluganda_tok.so',
    ]
    
    # Add .dll for Windows, .dylib for macOS
    if platform.system() == 'Windows':
        lib_names = ['libluganda_tok.dll', 'luganda_tok.dll']
    elif platform.system() == 'Darwin':
        lib_names = ['libluganda_tok.dylib']
    else:
        lib_names = ['libluganda_tok.so']
    
    for path in search_paths:
        for name in lib_names:
            full_path = os.path.join(os.path.dirname(path), name) if os.path.dirname(path) else name
            if os.path.exists(full_path):
                return full_path
    
    # Try to build it
    raise RuntimeError(
        "Cannot find libluganda_tok.so. Please build it first with 'make libluganda_tok.so'"
    )

# Load the library
try:
    _lib = CDLL(_find_library())
except OSError as e:
    raise RuntimeError(f"Failed to load C library: {e}")

# ============================================================================
# C Structure Definitions
# ============================================================================

class Corpus(Structure):
    """C Corpus struct mirror."""
    _fields_ = [
        ("docs", POINTER(c_char_p)),
        ("n_docs", c_uint32),
        ("mmap_ptr", c_void_p),
        ("mmap_len", c_size_t),
    ]

class Tokenizer(Structure):
    """Opaque Tokenizer struct handle."""
    pass

# ============================================================================
# Function Signatures
# ============================================================================

# Corpus functions
_lib.corpus_new.argtypes = []
_lib.corpus_new.restype = POINTER(Corpus)

_lib.corpus_load.argtypes = [POINTER(Corpus), c_char_p]
_lib.corpus_load.restype = c_int

_lib.corpus_free.argtypes = [POINTER(Corpus)]
_lib.corpus_free.restype = None

# Tokenizer creation/destruction
_lib.tokenizer_new.argtypes = []
_lib.tokenizer_new.restype = POINTER(Tokenizer)

_lib.tokenizer_free.argtypes = [POINTER(Tokenizer)]
_lib.tokenizer_free.restype = None

_lib.tokenizer_build.argtypes = [POINTER(Tokenizer), POINTER(Corpus)]
_lib.tokenizer_build.restype = c_int

# Tokenization
_lib.tokenizer_encode.argtypes = [
    POINTER(Tokenizer), c_char_p, 
    POINTER(c_uint32), c_size_t
]
_lib.tokenizer_encode.restype = c_int

_lib.tokenizer_decode.argtypes = [
    POINTER(Tokenizer), POINTER(c_uint32), c_size_t,
    c_char_p, c_size_t
]
_lib.tokenizer_decode.restype = c_int

# Serialization
_lib.tokenizer_save.argtypes = [POINTER(Tokenizer), c_char_p]
_lib.tokenizer_save.restype = c_int

_lib.tokenizer_load.argtypes = [POINTER(Tokenizer), c_char_p]
_lib.tokenizer_load.restype = c_int

# Info
_lib.tokenizer_vocab_size.argtypes = [POINTER(Tokenizer)]
_lib.tokenizer_vocab_size.restype = c_uint32

_lib.tokenizer_get_token_str.argtypes = [POINTER(Tokenizer), c_uint32]
_lib.tokenizer_get_token_str.restype = c_char_p

# ============================================================================
# Pythonic Wrapper Classes
# ============================================================================

class CorpusHandle:
    """Python wrapper for C Corpus."""
    
    def __init__(self):
        self._ptr = _lib.corpus_new()
        if not self._ptr:
            raise MemoryError("Failed to create corpus")
    
    def load(self, path: str) -> None:
        """Load corpus from file."""
        result = _lib.corpus_load(self._ptr, path.encode('utf-8'))
        if result != 0:
            raise RuntimeError(f"Failed to load corpus from {path}")
    
    def __del__(self):
        if hasattr(self, '_ptr') and self._ptr:
            _lib.corpus_free(self._ptr)
            self._ptr = None


class TokenizerHandle:
    """
    Low-level handle to the C Tokenizer.
    Use LugandaTokenizer (in tokenizer.py) for the high-level API.
    """
    
    def __init__(self):
        self._ptr = _lib.tokenizer_new()
        if not self._ptr:
            raise MemoryError("Failed to create tokenizer")
    
    def build(self, corpus: CorpusHandle) -> None:
        """Build tokenizer from corpus."""
        result = _lib.tokenizer_build(self._ptr, corpus._ptr)
        if result != 0:
            raise RuntimeError("Failed to build tokenizer")
    
    def encode(self, text: str) -> List[int]:
        """Encode text to token IDs."""
        # Allocate buffer for output
        max_tokens = len(text) * 2 + 10  # Conservative estimate
        output = (c_uint32 * max_tokens)()
        
        # Call C function
        n_tokens = _lib.tokenizer_encode(
            self._ptr, 
            text.encode('utf-8'),
            output,
            max_tokens
        )
        
        if n_tokens < 0:
            raise RuntimeError("Encoding failed")
        
        return list(output[:n_tokens])
    
    def decode(self, tokens: List[int]) -> str:
        """Decode token IDs to text."""
        # Allocate output buffer
        max_len = len(tokens) * 16 + 100
        output = (c_char * max_len)()
        
        # Convert tokens to C array
        tokens_arr = (c_uint32 * len(tokens))(*tokens)
        
        # Call C function
        result = _lib.tokenizer_decode(
            self._ptr,
            tokens_arr,
            len(tokens),
            output,
            max_len
        )
        
        if result < 0:
            raise RuntimeError("Decoding failed")
        
        return output.value.decode('utf-8')
    
    def save(self, path: str) -> None:
        """Save tokenizer model to file."""
        result = _lib.tokenizer_save(self._ptr, path.encode('utf-8'))
        if result != 0:
            raise RuntimeError(f"Failed to save model to {path}")
    
    def load_model(self, path: str) -> None:
        """Load tokenizer model from file."""
        result = _lib.tokenizer_load(self._ptr, path.encode('utf-8'))
        if result != 0:
            raise RuntimeError(f"Failed to load model from {path}")
    
    @property
    def vocab_size(self) -> int:
        """Get vocabulary size."""
        return _lib.tokenizer_vocab_size(self._ptr)
    
    def get_token_str(self, token_id: int) -> Optional[str]:
        """Get string representation of a token."""
        result = _lib.tokenizer_get_token_str(self._ptr, token_id)
        if result:
            return result.decode('utf-8')
        return None
    
    def __del__(self):
        if hasattr(self, '_ptr') and self._ptr:
            _lib.tokenizer_free(self._ptr)
            self._ptr = None


# ============================================================================
# Module Exports
# ============================================================================

__all__ = [
    'CorpusHandle',
    'TokenizerHandle',
    '_lib',
]
