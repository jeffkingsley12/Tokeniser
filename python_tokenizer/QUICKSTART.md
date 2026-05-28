# Quick Start - Python FFI Bindings

## 1. Build C Library

```bash
cd /home/jeff/Projects/c/Tokeniser
make lib
```

## 2. Install Python Package

```bash
cd python_tokenizer
pip install -e .
```

## 3. Use Tokenizer

```python
from python_tokenizer import LugandaTokenizer

# Load pre-trained model
tokenizer = LugandaTokenizer.load("production_model.bin")

# Encode text → tokens
tokens = tokenizer.encode("oluganda lwe ggwanga")
print(tokens)  # [1, 45, 67, 89, 2]

# Decode tokens → text
text = tokenizer.decode(tokens)
print(text)  # "oluganda lwe ggwanga"
```

## Training New Models

Use C tools (training not available via FFI):

```bash
make tokenizer_demo
./tokenizer_demo --train corpus.txt --save my_model.bin
```

Then load in Python:

```python
tokenizer = LugandaTokenizer.load("my_model.bin")
```

## Key API Methods

| Method | Description |
|--------|-------------|
| `LugandaTokenizer.load(path)` | Load pre-trained model |
| `encode(text)` → list[int] | Text to token IDs |
| `decode(tokens)` → str | Token IDs to text |
| `encode_batch(texts, num_workers=4)` | Parallel encoding |
| `get_vocab()` → dict[str, int] | Get vocabulary (slow, cached) |

## Architecture

```
Your Python Code
      ↓
python_tokenizer (ctypes FFI)
      ↓
libluganda_tok.so (C code - single source of truth)
```

Zero duplication. Native speed.
