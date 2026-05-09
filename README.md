# 🇺🇬 Luganda Morphological Tokenizer

A high-performance, production-grade tokenizer for the Luganda language, featuring a hybrid syllabification-based BPE approach, Re-Pair grammar compression, and a memory-optimized Split CSR LOUDS trie.

## 🚀 Key Features

- **Linguistic Syllabification**: Luganda-aware syllable boundary detection with support for geminates, pre-nasalized consonants, and long vowels.
- **Re-Pair Compression**: Iterative byte-pair encoding (BPE) variant optimized for morphology, delivering high compression ratios while preserving linguistic units.
- **Split CSR LOUDS Trie**: A state-of-the-art, memory-mapped trie layout using Compressed Sparse Row (CSR) for O(1) fast-path lookups and minimal memory footprint.
- **Aho-Corasick Truth Layer**: Integrated verification layer for identifying overlapping morphological matches and enforcing truth-set alignment.
- **Zero-Copy Streaming**: Production-ready streaming API with lookahead margins and robust EOF handling.
- **Memory Efficient**: mmap-based "page after page" corpus loading for multi-gigabyte training sets, supporting arbitrarily long lines (65KB+).

## 📊 Performance

| Configuration | Cycles/Token | Tokens/Sec (3GHz) | Cycles/Byte |
|---------------|-------------|-------------------|-------------|
| **Regular**   | 872         | 3.4M              | 425.99      |
| **Fused**     | 1,267       | 2.4M              | 431.02      |

*Benchmarks on 9,389 documents, 670KB corpus. Fused path overhead includes real-time Aho-Corasick morphological annotation.*

## 🏗️ Architecture

The tokenizer operates in a multi-stage pipeline:

1.  **Syllabification**: Raw text is decomposed into a stream of Luganda syllables.
2.  **Re-Pair Training**: Global frequency analysis identifies optimal merges to form subwords and tokens.
3.  **Trie Construction**: Tokens are compiled into a LOUDS (Level-Order Unary Degree Sequence) trie using a Split CSR layout.
4.  **Truth Layer Integration**: The Aho-Corasick automaton is built on top of the trie to provide morphological annotation.

### 🧠 Production Hardening

- **mmap Paging**: Corpus loading uses `MAP_PRIVATE` to scan data directly from disk, avoiding redundant allocations and supporting massive datasets.
- **OOM Safety**: All trie resizes include atomic rollback logic to maintain structural integrity under low-memory conditions.
- **Atomic Migrations**: Re-Pair hash table resizing uses a "validate-before-free" approach to prevent pointer corruption during state migration.

## 🛠️ Getting Started

### Prerequisites

- GCC or Clang
- Make
- Linux (for mmap/paging optimizations)

### Building the Project

```bash
# Build the main demo
make tokenizer_demo

# Build the performance benchmark suite
make bench

# Run the regression test suite
make test
```

### Running the Demo

```bash
./tokenizer_demo [path/to/corpus.txt]
```

## 📂 Directory Structure

- `src/`: Core implementation files.
- `include/`: API definitions and shared headers.
- `test/`: Regression and unit tests.
- `build_antigravity/`: (Generated) Object files and binaries.

## 🔌 Integration Guide

### 1. Use directly from C/C++

The API surface is small and designed for zero-copy production environments.

```c
#include "tokenizer.h"

// Load a trained model (zero-copy mmap version)
MmapTokenizer *mt = tokenizer_load_mmap("luganda_tok.bin");
Tokenizer     *tok = &mt->base;

// Encode
uint32_t ids[512];
int n = tokenizer_encode(tok, "okusoma", ids, 512);

// Decode
for (int i = 0; i < n; i++)
    printf("%s ", tokenizer_decode(tok, ids[i]));

// Cleanup
tokenizer_destroy_mmap(mt);
```

### 2. Call from Python via ctypes

For fast Python integration, compile the engine as a shared library:

```bash
gcc -O2 -shared -fPIC -Iinclude src/*.c -o libluganda_tok.so -lm
```

Then load it using `ctypes`:

```python
import ctypes
lib = ctypes.CDLL("./libluganda_tok.so")
# ... define restypes and argtypes ...
```

### 3. Porting to Other Runtimes

The `.bin` format is designed for straightforward deserialization in Rust, Go, or Mojo. It consists of a fixed-width header followed by aligned CSR arrays. See `include/tokenizer.h` for the `MmapHeader` structure and data layout.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

