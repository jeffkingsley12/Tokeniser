# Luganda Tokenizer - Python Bindings

High-performance Luganda tokenizer with Python bindings via FFI.

## Installation

### Prerequisites

You need the C library built first:

```bash
cd /home/jeff/Projects/c/Tokeniser
make libluganda_tok.so
```

### Install Python Package

```bash
cd python_tokenizer
pip install -e .
```

Or with development dependencies:
```bash
pip install -e ".[dev]"
```

## Quick Start

### 1. Train a model using C tools

```bash
cd /home/jeff/Projects/c/Tokeniser
make tokenizer_demo
./tokenizer_demo --train corpus.txt --save model.bin
```

### 2. Load and use in Python

```python
from python_tokenizer import LugandaTokenizer

# Load pre-trained model
tokenizer = LugandaTokenizer.load("model.bin")

# Encode/decode
tokens = tokenizer.encode("bw'osoma oluganda")
text = tokenizer.decode(tokens)

# Batch processing
texts = ["ebirabo by'omuzungu", "ssoma olupapula"]
token_lists = tokenizer.encode_batch(texts, num_workers=4)
```

## Architecture

This package uses **FFI bindings** to wrap the C tokenizer:

- `python_tokenizer/_ffi.py` - Low-level ctypes bindings
- `python_tokenizer/tokenizer.py` - High-level Pythonic API
- `libluganda_tok.so` - Compiled C library (single source of truth)

## API Reference

### LugandaTokenizer

Main tokenizer class.

#### Methods

- `train(corpus: List[str])` - Train on text corpus
- `encode(text: str) -> List[int]` - Text to token IDs
- `decode(tokens: List[int]) -> str` - Token IDs to text
- `encode_batch(texts: List[str], num_workers=4)` - Parallel encoding
- `save(path: str)` - Save model to file
- `load(path: str) -> LugandaTokenizer` - Load model from file

#### Properties

- `vocab_size: int` - Size of vocabulary
- `get_vocab(): Dict[str, int]` - Full vocabulary mapping

### StreamingTokenizer

Memory-efficient tokenizer for large files using mmap.

```python
from python_tokenizer import StreamingTokenizer

st = StreamingTokenizer("model.bin")
for tokens in st.tokenize_file("huge_corpus.txt"):
    process(tokens)
```

## Performance

The Python bindings have minimal overhead:

- **Encode**: ~4.5M ops/sec (same as C - Python just marshals data)
- **Batch encoding**: Scales linearly with num_workers
- **Memory**: Only token arrays cross the FFI boundary

## Comparison: FFI vs Native Python

| Approach | Lines of Code | Speed | Maintainability |
|----------|---------------|-------|-----------------|
| **FFI (this)** | ~500 Python | Native C speed | Single source of truth |
| Native Python rewrite | ~3000 Python | ~10x slower | Duplicated logic |
| Cython | ~500 .pyx | Near native | Requires compilation |

## Testing

```bash
pytest tests/
```

## DRY Principle

This package follows **Don't Repeat Yourself**:

1. **C code is the source of truth** - All tokenization logic lives in C
2. **Python is a thin wrapper** - Only marshals data and provides nice API
3. **Single build artifact** - One .so file used by all languages

If you need to modify tokenization logic, edit the C files and rebuild.
The Python bindings will automatically use the updated logic.
