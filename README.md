# Luganda Tokenizer (Tokeniser)

A high-performance, linguistically-aware BPE tokenizer specifically optimized for the Luganda language. It addresses common issues in standard tokenizers, such as non-deterministic output and morphological inaccuracy (e.g., preserving syllable boundaries like `okusoma` → `[oku][soma]`).

## Key Features

- **Linguistically Aware**: Uses a custom "Truth Layer" and syllabification pass to ensure tokens respect Luganda's morphological structure.
- **Deterministic**: Guarantees identical vocabulary IDs for the same corpus across different runs.
- **High Performance**: Optimized C implementation reaching ~9.6M tokens/sec (313 cycles/token) on modern x86 hardware.
- **SIMD Optimized**: Support for SSE4.1, AVX-512, and ARM NEON.
- **Memory Efficient**: Uses LOUDS (Level-Order Unary Degree Sequence) trie representation for a compact model footprint.
- **Two-Pass Architecture**: Optimized for branch prediction and cache locality.

## Performance Baseline

| Configuration | Cycles/Token | Throughput (3GHz) |
|---------------|-------------|-------------------|
| **Baseline**  | 313         | 9.6M tokens/sec   |
| **SSE4.1**    | 150         | 20.0M tokens/sec  |
| **AVX-512**   | 80-100      | 30-37M tokens/sec |

*Benchmarks measured on Intel Ice Lake/AMD Zen 4.*

## Getting Started

### Prerequisites

- GCC or Clang
- Make
- Python 3 (for analysis scripts)

### Building

```bash
# Clone the repository
git clone https://github.com/your-username/Tokeniser.git
cd Tokeniser

# Build with production defaults
make clean
make CFLAGS="-O3 -march=native -DENABLE_STATS=0"
```

### Usage

```bash
# Run the demo
./tokenizer_demo your_text_file.txt

# Run benchmarks
./benchmark_luganda
```

## Architecture

The tokenizer operates in two main passes:
1. **Syllabification**: Text is segmented into Luganda-appropriate syllables using a hybrid mode (byte-table + rule-based).
2. **Trie Traversal**: The segmented syllables are then processed through a LOUDS trie for final token assignment.

This design avoids the branch misprediction penalties associated with fused single-pass tokenizers while maintaining superior cache locality.

## Contributing

Contributions are welcome! Please feel free to submit Pull Requests or open Issues for bugs and feature requests.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
