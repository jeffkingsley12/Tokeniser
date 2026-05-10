"""
Luganda Tokenizer - Native Python Implementation

A pure Python rewrite of the C tokenizer with clean, Pythonic APIs.
Optimized for both readability and performance using modern Python features.

Basic Usage:
    >>> from python_tokenizer import LugandaTokenizer
    >>> tokenizer = LugandaTokenizer()
    >>> tokenizer.train(["oluganda lwe ggwanga"])
    >>> tokens = tokenizer.encode("nze mbaagala")
    >>> text = tokenizer.decode(tokens)

Advanced Usage with Pre-trained Model:
    >>> tokenizer = LugandaTokenizer.load("model.bin")
    >>> tokens = tokenizer.encode_batch(corpus, num_workers=4)
"""

__version__ = "1.0.0"
__author__ = "Jeff .K. Mukisa"

from .tokenizer import LugandaTokenizer
from .syllabifier import Syllabifier, SyllableTable
from .louds import LoudsTrie
from .grammar import Grammar

__all__ = [
    "LugandaTokenizer",
    "Syllabifier", 
    "SyllableTable",
    "LoudsTrie",
    "Grammar",
]
