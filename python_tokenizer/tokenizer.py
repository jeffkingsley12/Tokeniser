"""
High-level Pythonic API for the Luganda Tokenizer.
Wraps the FFI bindings with a clean, user-friendly interface.
"""

import os
import tempfile
from typing import List, Optional, Union, Iterator, BinaryIO
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import mmap

from ._ffi import TokenizerHandle, CorpusHandle
from .config import SPECIAL_TOKENS, TOK_UNK, TOK_BOS, TOK_EOS, TOK_PAD, TOK_MASK


class LugandaTokenizer:
    """
    Production-ready Luganda tokenizer with Pythonic API.
    
    This is a thin wrapper around the high-performance C implementation,
    providing a clean interface for training, encoding, and decoding.
    
    Example:
        >>> tokenizer = LugandaTokenizer()
        >>> tokenizer.train(["oluganda lwe ggwanga", "nze mbaagala"])
        >>> tokens = tokenizer.encode("bw'osoma")
        >>> text = tokenizer.decode(tokens)
        >>> tokenizer.save("my_model.bin")
        
        >>> # Load pre-trained
        >>> tokenizer = LugandaTokenizer.load("my_model.bin")
    """
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize tokenizer.
        
        Args:
            model_path: Path to pre-trained model file. If None, creates empty tokenizer.
        """
        self._handle = TokenizerHandle()
        self._trained = False
        
        if model_path:
            self.load(model_path)
    
    def train(self, corpus: Union[List[str], str], 
              num_workers: int = 1) -> 'LugandaTokenizer':
        """
        Train tokenizer on text corpus.
        
        Args:
            corpus: List of documents or single text string
            num_workers: Parallel workers (currently single-threaded via C)
            
        Returns:
            self for method chaining
            
        Example:
            >>> tokenizer.train(["oluganda lwe ggwanga", "nze mbaagala"])
        """
        if isinstance(corpus, str):
            corpus = [corpus]
        
        # Write corpus to temp file (C API expects file)
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            for doc in corpus:
                f.write(doc + '\n')
            corpus_path = f.name
        
        try:
            # Load and build
            corpus_handle = CorpusHandle()
            corpus_handle.load(corpus_path)
            self._handle.build(corpus_handle)
            self._trained = True
        finally:
            os.unlink(corpus_path)
        
        return self
    
    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        """
        Encode text to token IDs.
        
        Args:
            text: Input text to tokenize
            add_special_tokens: Whether to add BOS/EOS tokens
            
        Returns:
            List of token IDs
            
        Example:
            >>> tokens = tokenizer.encode("oluganda")
            >>> # [1, 234, 567, 89, 2]  # BOS + tokens + EOS
        """
        if not self._trained and self.vocab_size == 0:
            raise RuntimeError("Tokenizer not trained. Call train() or load() first.")
        
        tokens = self._handle.encode(text)
        
        if add_special_tokens:
            tokens = [TOK_BOS] + tokens + [TOK_EOS]
        
        return tokens
    
    def decode(self, tokens: List[int], skip_special_tokens: bool = True) -> str:
        """
        Decode token IDs to text.
        
        Args:
            tokens: List of token IDs
            skip_special_tokens: Whether to remove special tokens (BOS, EOS, etc.)
            
        Returns:
            Decoded text string
        """
        if skip_special_tokens:
            tokens = [t for t in tokens if t >= len(SPECIAL_TOKENS)]
        
        return self._handle.decode(tokens)
    
    def encode_batch(self, texts: List[str], 
                     num_workers: int = 4,
                     add_special_tokens: bool = True) -> List[List[int]]:
        """
        Encode multiple texts in parallel.
        
        Args:
            texts: List of texts to tokenize
            num_workers: Number of parallel workers
            add_special_tokens: Whether to add BOS/EOS tokens
            
        Returns:
            List of token ID lists
        """
        if num_workers == 1:
            return [self.encode(t, add_special_tokens) for t in texts]
        
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = {
                executor.submit(self.encode, t, add_special_tokens): i 
                for i, t in enumerate(texts)
            }
            
            results = [None] * len(texts)
            for future in as_completed(futures):
                idx = futures[future]
                results[idx] = future.result()
        
        return results
    
    def save(self, path: Union[str, Path]) -> None:
        """
        Save tokenizer model to file.
        
        Args:
            path: Output file path (e.g., "tokenizer.bin")
        """
        self._handle.save(str(path))
    
    @classmethod
    def load(cls, path: Union[str, Path]) -> 'LugandaTokenizer':
        """
        Load pre-trained tokenizer from file.
        
        Args:
            path: Model file path
            
        Returns:
            Loaded LugandaTokenizer instance
        """
        tokenizer = cls()
        tokenizer._handle.load_model(str(path))
        tokenizer._trained = True
        return tokenizer
    
    @property
    def vocab_size(self) -> int:
        """Get vocabulary size."""
        return self._handle.vocab_size
    
    def get_vocab(self) -> dict:
        """
        Get full vocabulary as {token: id} dictionary.
        
        Returns:
            Dictionary mapping token strings to IDs
        """
        vocab = {}
        for i in range(self.vocab_size):
            token_str = self._handle.get_token_str(i)
            if token_str:
                vocab[token_str] = i
        return vocab
    
    def id_to_token(self, token_id: int) -> Optional[str]:
        """Convert token ID to string."""
        return self._handle.get_token_str(token_id)
    
    def token_to_id(self, token: str) -> Optional[int]:
        """Convert token string to ID (requires vocab lookup)."""
        vocab = self.get_vocab()
        return vocab.get(token)
    
    def __repr__(self) -> str:
        status = "trained" if self._trained else "untrained"
        return f"LugandaTokenizer({status}, vocab_size={self.vocab_size})"
    
    def __len__(self) -> int:
        """Return vocabulary size."""
        return self.vocab_size


class StreamingTokenizer:
    """
    Memory-efficient tokenizer for large files using mmap.
    Processes files without loading entirely into memory.
    """
    
    def __init__(self, model_path: str):
        """
        Initialize with pre-trained model.
        
        Args:
            model_path: Path to trained model file
        """
        self._tokenizer = LugandaTokenizer.load(model_path)
    
    def tokenize_file(self, path: str, batch_size: int = 1000) -> Iterator[List[int]]:
        """
        Lazily tokenize a large file line by line.
        
        Args:
            path: File to tokenize
            batch_size: Lines to process per batch
            
        Yields:
            Lists of token IDs for each line
        """
        with open(path, 'r', encoding='utf-8') as f:
            # Use mmap for large files
            if os.path.getsize(path) > 10 * 1024 * 1024:  # 10MB
                with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
                    for line in iter(mm.readline, b''):
                        text = line.decode('utf-8').strip()
                        if text:
                            yield self._tokenizer.encode(text)
            else:
                for line in f:
                    text = line.strip()
                    if text:
                        yield self._tokenizer.encode(text)
    
    def encode_file(self, input_path: str, output_path: str,
                   format: str = 'json') -> None:
        """
        Tokenize entire file and save results.
        
        Args:
            input_path: Input text file
            output_path: Output file
            format: 'json', 'txt', or 'npy'
        """
        import json
        
        results = []
        for tokens in self.tokenize_file(input_path):
            results.append(tokens)
        
        if format == 'json':
            with open(output_path, 'w') as f:
                json.dump(results, f)
        elif format == 'txt':
            with open(output_path, 'w') as f:
                for tokens in results:
                    f.write(' '.join(map(str, tokens)) + '\n')
        elif format == 'npy':
            import numpy as np
            # Pad to equal length for numpy array
            max_len = max(len(t) for t in results)
            padded = [t + [TOK_PAD] * (max_len - len(t)) for t in results]
            np.save(output_path, np.array(padded, dtype=np.uint32))
        else:
            raise ValueError(f"Unknown format: {format}")


# Convenience functions
def train_tokenizer(corpus: List[str], save_path: Optional[str] = None) -> LugandaTokenizer:
    """Quick training helper."""
    tokenizer = LugandaTokenizer()
    tokenizer.train(corpus)
    if save_path:
        tokenizer.save(save_path)
    return tokenizer


def load_tokenizer(path: str) -> LugandaTokenizer:
    """Quick loading helper."""
    return LugandaTokenizer.load(path)
