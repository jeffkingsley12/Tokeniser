"""
Luganda Tokenizer - Python FFI Bindings

High-performance Python bindings for the C tokenizer.
Uses FFI (ctypes) to wrap libluganda_tok.so with zero code duplication.

Basic Usage:
    >>> from python_tokenizer import LugandaTokenizer
    >>> tokenizer = LugandaTokenizer.load("model.bin")
    >>> tokens = tokenizer.encode("nze mbaagala")
    >>> text = tokenizer.decode(tokens)

Training (use C tools):
    $ make tokenizer_demo
    $ ./tokenizer_demo --train corpus.txt --save model.bin
"""

__version__ = "1.0.0"
__author__ = "AI Assisted Port"

from .luganda_tokenizer import LugandaTokenizer, load_tokenizer
from .closed_loop_agent import LugandaClosedLoopEngineAgent

__all__ = [
    "LugandaTokenizer",
    "load_tokenizer",
    "LugandaClosedLoopEngineAgent",
]
