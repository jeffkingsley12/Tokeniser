"""
luganda_tokenizer.py
====================
Python bindings for libluganda_tok.so via ctypes.

All tokenizer logic lives in C — this file only describes the four-function
C API and wraps them in an ergonomic Python class.

Quick start
-----------
    from luganda_tokenizer import LugandaTokenizer

    with LugandaTokenizer("luganda_tok.bin") as tok:
        ids    = tok.encode("okusoma nnyo")        # → list[int]
        tokens = tok.encode_str("okusoma nnyo")    # → list[str]
        word   = tok.decode(ids[0])                # → str

Thread safety
-------------
A single LugandaTokenizer instance may be shared across threads.
The underlying C library serialises tok_load / tok_free with a mutex;
tok_encode / tok_decode are fully concurrent.

Requirements
------------
- libluganda_tok.so (or .dylib on macOS) in the same directory as this
  file, or on LD_LIBRARY_PATH / DYLD_LIBRARY_PATH.
- Python 3.8+
- No third-party dependencies.
"""

from __future__ import annotations

import ctypes
import os
import sys
import threading
from pathlib import Path
# FIX: Added Union, List, Dict for Python 3.8 compatibility.
# Python 3.9+ allows built-in generics (list[int], dict[k,v]) and
# 3.10+ allows X | Y union syntax. The project targets Python 3.8+
# (pyproject.toml requires-python = ">=3.8"), so we must use typing.*
from typing import Dict, Iterator, List, Sequence, Union

# ── Locate and load the shared library ───────────────────────────────────────

def _find_lib() -> str:
    """Search for the shared library next to this file, then parent dir, then system path."""
    ext  = "dylib" if sys.platform == "darwin" else "so"
    name = f"libluganda_tok.{ext}"
    here = Path(__file__).parent / name
    if here.exists():
        return str(here)
    parent = Path(__file__).parent.parent / name
    if parent.exists():
        return str(parent)
    # Fall back to whatever the dynamic linker can find.
    return name


def _load_lib() -> ctypes.CDLL:
    lib = ctypes.CDLL(_find_lib())

    # int tok_load(const char *path)
    lib.tok_load.argtypes  = [ctypes.c_char_p]
    lib.tok_load.restype   = ctypes.c_int

    # int tok_encode(int handle, const char *text, uint32_t *out, int cap)
    lib.tok_encode.argtypes = [
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.c_int,
    ]
    lib.tok_encode.restype = ctypes.c_int

    # const char *tok_decode(int handle, uint32_t id)
    lib.tok_decode.argtypes = [ctypes.c_int, ctypes.c_uint32]
    lib.tok_decode.restype  = ctypes.c_char_p   # do NOT free; owned by C

    # void tok_free(int handle)
    lib.tok_free.argtypes = [ctypes.c_int]
    lib.tok_free.restype  = None

    return lib


_lib = _load_lib()

# ── Exceptions ────────────────────────────────────────────────────────────────

class TokenizerError(RuntimeError):
    """Raised when the underlying C library returns an error."""


# ── LugandaTokenizer ──────────────────────────────────────────────────────────

class LugandaTokenizer:
    """
    Wraps a single tokenizer handle returned by tok_load().

    Parameters
    ----------
    model_path : str | Path
        Path to the .bin model file produced by tokenizer_save().
    initial_cap : int
        Initial token buffer size for encode().  Doubles automatically if
        a longer sequence is encountered.

    Notes
    -----
    The handle is opened eagerly in __init__ and closed in close() / __exit__.
    Use as a context manager to guarantee cleanup:

        with LugandaTokenizer("model.bin") as tok:
            ...
    """

    _INITIAL_CAP = 512
    # IDs 0-4 are the standard special tokens (matches config.py SPECIAL_TOKENS_COUNT)
    _SPECIAL_TOKEN_CEILING = 5

    def __init__(self, model_path: Union[str, Path], initial_cap: int = _INITIAL_CAP) -> None:
        path   = os.fsencode(str(model_path))
        handle = _lib.tok_load(path)
        if handle < 0:
            raise TokenizerError(
                f"tok_load() failed — check that '{model_path}' exists "
                "and is a valid model file"
            )
        self._handle: int = handle
        # Thread-local buffers so concurrent encodes each use their own buffer
        self._local = threading.local()
        self._initial_cap = max(1, initial_cap)
        # FIX: Lock to prevent double-free when close() is called concurrently.
        # Without this, two threads can both pass the `not self._closed` check
        # before either sets _closed = True, causing tok_free() to run twice.
        self._close_lock = threading.Lock()
        self._closed = False

    # ── Core encode ───────────────────────────────────────────────────────────

    def _get_buf(self):
        """Return the thread-local buffer state, creating it on first access."""
        if not hasattr(self._local, 'buf'):
            self._local.cap = self._initial_cap
            self._local.buf = (ctypes.c_uint32 * self._local.cap)()
        return self._local

    def _encode_raw(self, text: str) -> List[int]:
        """
        Encode *text* and return a list of token IDs.

        The C function returns the number of tokens written.  If the buffer
        was too small the C API should return a negative sentinel; until that
        contract is formalised we double-and-retry whenever n == cap.
        """
        encoded = text.encode("utf-8")
        tl = self._get_buf()
        while True:
            n = _lib.tok_encode(self._handle, encoded, tl.buf, tl.cap)
            if n < 0:
                raise TokenizerError(
                    f"tok_encode() returned {n} for input: {text!r}"
                )
            # If n == cap the buffer may have been silently truncated.
            # Double and retry to be safe.
            if n < tl.cap:
                return list(tl.buf[:n])
            new_cap = tl.cap * 2
            if new_cap > (1 << 24):
                raise TokenizerError(
                    f"Encoded sequence exceeds maximum buffer size ({1 << 24} tokens)"
                )
            tl.cap = new_cap
            tl.buf = (ctypes.c_uint32 * tl.cap)()

    def encode(self, text: str, add_special_tokens: bool = False) -> List[int]:
        """
        Tokenise *text* into a list of integer token IDs.

        Parameters
        ----------
        text : str
            UTF-8 text to tokenise.
        add_special_tokens : bool, default False
            If True, adds BOS/EOS special tokens (HuggingFace API compatibility).
            Currently a no-op — the C library controls special token injection.

        Returns
        -------
        List[int]
            Token IDs.  Empty list for empty input.

        Raises
        ------
        TokenizerError
            If the C library returns an error or input exceeds maximum size.
        TypeError
            If *text* is not a str.
        ValueError
            If *text* exceeds maximum input size.
        """
        if not isinstance(text, str):
            raise TypeError(f"text must be str, got {type(text).__name__}")
        self._check_open()
        if not text:
            return []
        MAX_INPUT_CHARS = 100_000
        if len(text) > MAX_INPUT_CHARS:
            raise ValueError(
                f"Input text exceeds maximum size of {MAX_INPUT_CHARS} characters "
                f"(got {len(text)} characters)"
            )
        return self._encode_raw(text)

    # ── Decode ────────────────────────────────────────────────────────────────

    def decode(self, token_id_or_ids, skip_special_tokens: bool = False) -> str:
        """
        Return the surface string for a token ID or list of token IDs.

        Parameters
        ----------
        token_id_or_ids : int | List[int]
            A token ID or list of token IDs previously returned by encode().
        skip_special_tokens : bool, default False
            If True, tokens with IDs 0-4 (the standard special tokens defined
            in config.py) are omitted from the output. Provided for
            HuggingFace API compatibility.

        Returns
        -------
        str
            The surface form, or the concatenated decoded string for a list.

        Raises
        ------
        TokenizerError
            If *token_id* is out of range.
        """
        self._check_open()
        if isinstance(token_id_or_ids, (list, tuple)):
            parts = []
            for t in token_id_or_ids:
                if skip_special_tokens and self._is_special_token(t):
                    continue
                parts.append(self._decode_single(t))
            return "".join(parts)
        return self._decode_single(token_id_or_ids)

    def _is_special_token(self, token_id: int) -> bool:
        """Return True if token_id is a standard special token (IDs 0-4)."""
        return 0 <= token_id < self._SPECIAL_TOKEN_CEILING

    def _decode_single(self, token_id: int) -> str:
        """Decode a single token ID to its surface string."""
        raw = _lib.tok_decode(self._handle, ctypes.c_uint32(token_id))
        if raw is None:
            raise TokenizerError(f"tok_decode() returned NULL for token_id={token_id}")
        return raw.decode("utf-8")

    # ── Convenience helpers ───────────────────────────────────────────────────

    def encode_str(self, text: str) -> List[str]:
        """Tokenise *text* and return surface strings instead of IDs."""
        return [self._decode_single(tid) for tid in self.encode(text)]

    def tokenize_batch(self, texts: Sequence[str]) -> List[List[int]]:
        """
        Encode a list of strings sequentially.

        For large batches, use tokenize_batch_threaded for parallelism.

        Parameters
        ----------
        texts : Sequence[str]

        Returns
        -------
        List[List[int]]
        """
        self._check_open()
        return [self.encode(t) for t in texts]

    def encode_batch(self, texts: Sequence[str], num_workers: int = 2) -> List[List[int]]:
        """
        Encode a batch of texts in parallel (alias for tokenize_batch_threaded).

        Parameters
        ----------
        texts : Sequence[str]
        num_workers : int, default 2

        Returns
        -------
        List[List[int]]
        """
        return self.tokenize_batch_threaded(texts, workers=num_workers)

    def get_vocab(self) -> Dict[str, int]:
        """
        Return vocabulary as a mapping from token string to token ID.

        Note
        ----
        The C library does not expose the full vocabulary, so this returns an
        empty dict. Provided for HuggingFace tokenizer API compatibility.

        Returns
        -------
        Dict[str, int]
            Empty dict — vocabulary is not accessible via FFI.
        """
        # FIX: Return type is Dict[str, int] (token_string → id) to match the
        # HuggingFace convention and how example.py iterates over it:
        #   `{k: v for k, v in vocab.items() if k.startswith('<')}`
        return {}

    @classmethod
    def load(cls, model_path: Union[str, Path], **kwargs) -> "LugandaTokenizer":
        """
        Convenience class method to load a tokenizer.

        Parameters
        ----------
        model_path : str | Path
        **kwargs
            Forwarded to __init__ (e.g. initial_cap).

        Returns
        -------
        LugandaTokenizer
        """
        return cls(model_path, **kwargs)

    def tokenize_batch_threaded(
        self, texts: Sequence[str], *, workers: int = 4
    ) -> List[List[int]]:
        """
        Parallel batch encode using a thread pool.

        ctypes calls release the GIL, so this gives real CPU parallelism when
        the C tokenizer is the bottleneck.

        Parameters
        ----------
        texts   : Sequence[str]
        workers : int, default 4

        Returns
        -------
        List[List[int]]
        """
        from concurrent.futures import ThreadPoolExecutor

        self._check_open()
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(self.encode, t) for t in texts]
            return [f.result() for f in futures]

    # ── Streaming (token-by-token) ─────────────────────────────────────────────

    def stream(self, text: str) -> Iterator[int]:
        """
        Yield token IDs one at a time.

        Useful for streaming output to a downstream consumer. Note that the
        full token list is computed up-front (the C API is batch-only); this
        method simply iterates over that result.
        """
        for tid in self.encode(text):
            yield tid

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def close(self) -> None:
        """Release the tokenizer handle.  Safe to call more than once."""
        with self._close_lock:
            if not self._closed and self._handle >= 0:
                _lib.tok_free(self._handle)
                self._handle = -1
                self._closed = True

    def __enter__(self) -> "LugandaTokenizer":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def __del__(self) -> None:
        # Best-effort cleanup if the user forgets close() / context manager.
        # We cannot use self._close_lock here because the interpreter may be
        # tearing down at finalisation time; a bare try/except is safest.
        try:
            if not self._closed and self._handle >= 0 and _lib is not None:
                _lib.tok_free(self._handle)
                self._handle = -1
                self._closed = True
        except Exception:
            pass

    def _check_open(self) -> None:
        if self._closed:
            raise TokenizerError("LugandaTokenizer has already been closed")

    # ── Repr ──────────────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        state = "closed" if self._closed else f"handle={self._handle}"
        return f"LugandaTokenizer({state})"


# ── Module-level convenience ──────────────────────────────────────────────────

def load(model_path: Union[str, Path]) -> LugandaTokenizer:
    """
    Load a tokenizer and return a ``LugandaTokenizer`` instance.

    Equivalent to ``LugandaTokenizer(model_path)``.  Prefer the class
    directly when you need context-manager lifetime management.
    """
    return LugandaTokenizer(model_path)


# Alias for __init__.py import
load_tokenizer = load


# ── CLI smoke-test ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Luganda tokenizer smoke-test")
    parser.add_argument("model", help="Path to .bin model file")
    parser.add_argument("texts", nargs="+", help="Text(s) to tokenise")
    args = parser.parse_args()

    with LugandaTokenizer(args.model) as tok:
        for text in args.texts:
            ids    = tok.encode(text)
            tokens = [tok.decode(i) for i in ids]
            print(f"Input : {text!r}")
            print(f"IDs   : {ids}")
            print(f"Tokens: {tokens}")
            print()
