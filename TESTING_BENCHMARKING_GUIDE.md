# Tokeniser Testing & Benchmarking Guide

## Overview

This guide provides a comprehensive approach to testing and benchmarking the Tokeniser project. The testing infrastructure includes:

- **Unit Tests**: Core functionality validation
- **Integration Tests**: Multi-component interaction testing  
- **Benchmarks**: Performance measurement under realistic workloads
- **Analysis Tools**: Result aggregation and visualization

## Quick Start

### Running Tests

```bash
# Run all unit and integration tests
bash run_tests.sh

# View test results
tail -50 test_results/test_report_*.txt
```

### Running Benchmarks

```bash
# Run all benchmarks
bash run_benchmarks.sh

# View benchmark results
ls -lh benchmark_results/
```

### Analyzing Results

```bash
# Generate comprehensive analysis report
python3 analyze_results.py .

# View analysis
cat analysis/report_*.md
```

## Test Infrastructure

### Available Test Suites

| Test | Binary | Type | Purpose |
|------|--------|------|---------|
| `test_simple_tokenizer` | `/test/test_simple_tokenizer` | Unit | Basic tokenization API |
| `test_benchmark_simple` | `/test/test_benchmark_simple` | Benchmark | Performance baseline |
| `test_edge_cases` | `/test/test_edge_cases_debug` | Unit | Edge case handling |
| `test_serialization_security` | `/test/test_serialization_security` | Integration | Data integrity & safety |
| `test_corruption_minimal` | `/test/test_corruption_minimal` | Robustness | Error recovery |
| `test_fixes` | `/test/test_fixes` | Regression | Previous bug fixes |
| `syllabifier_modes` | `/test/test_syllabifier_modes.sh` | Integration | Syllabifier modes |

### Test Suite Details

#### 1. Simple Tokenizer Test
**File:** `test/test_simple_tokenizer.c`

Tests basic tokenization with built-in or loaded corpus:

```c
// Load a tokenizer model
Tokenizer* tok = tokenizer_load("tokenizer_model.bin");

// Tokenize text
const char* text = "okusoma nnyo";
uint32_t tokens[100];
int count = tokenizer_encode(tok, text, tokens, 100);

// Decode back
for (int i = 0; i < count; i++) {
    const char* word = tokenizer_decode(tok, tokens[i]);
}
```

**Expected Output:**
- Model load confirmation
- Token count
- Decoded tokens
- Throughput (ops/sec)

#### 2. Edge Cases Test
**File:** `test/test_edge_cases.c`

Covers boundary conditions:
- Empty strings
- Single characters
- Multi-byte UTF-8 sequences
- Unicode normalization edge cases
- Grapheme cluster handling
- Whitespace variations (space, NBSP, ZWSP)
- Line ending variants (LF, CRLF, CR)

#### 3. Serialization & Security Test
**File:** `test/test_serialization_security.c`

Validates data integrity:
- Model save/load roundtrips
- Corruption detection
- Memory bounds checking
- Token validity verification

#### 4. Syllabifier Modes Test
**File:** `test/test_syllabifier_modes.sh`

Tests three syllabifier modes:
- **Louds-only**: Fast path via LOUDS trie
- **Viterbi**: Probabilistic path (if available)
- **Hybrid**: Adaptive mode selection

```bash
# Run individual modes
./test_syllabifier_modes.sh louds
./test_syllabifier_modes.sh viterbi
./test_syllabifier_modes.sh hybrid
```

## Benchmark Infrastructure

### Available Benchmarks

| Benchmark | File | Metrics |
|-----------|------|---------|
| Simple | `benchmark.c` | Throughput, iterations |
| Luganda-specific | `benchmark_luganda.c` | Fast-path %, compression ratio |
| Training | `src/train_bench.c` | Model generation time |

### Understanding Benchmark Metrics

#### Throughput (tokens/ms)

```
Throughput = Total Tokens Processed / Total Time (ms)
```

**Interpretation:**
- **>100 tokens/ms**: Excellent (well-optimized)
- **50-100 tokens/ms**: Good
- **<50 tokens/ms**: Needs optimization

**Example:**
```
Processed 50,000 tokens in 250ms
Throughput = 50,000 / 250 = 200 tokens/ms ✓
```

#### Fast-Path Ratio (%)

```
Fast-Path % = (Direct Lookups / Total Lookups) × 100
```

**Interpretation:**
- **70-90%**: Excellent (most text uses fast paths)
- **50-70%**: Good
- **<50%**: Suboptimal (many edge cases or mixed-language text)

**Components:**
1. **CV_TO_TOKEN**: Direct byte-to-token mapping (fastest)
2. **SPACE_PAIR**: Space + next syllable optimization
3. **SINGLE_SYL**: Single syllable lookup
4. **TRIE_FALLBACK**: Full trie traversal (slowest)

#### Cycles Per Token

```
CPT = (Total CPU Cycles) / (Total Tokens Processed)
```

**Interpretation:**
- **<500 cycles/token**: Excellent
- **500-1000 cycles/token**: Good
- **>1000 cycles/token**: Optimization opportunity

#### Compression Ratio (bytes/token)

```
Ratio = (Total Text Bytes) / (Total Token Count)
```

**Interpretation:**
- **2-3 bytes/token**: Excellent (aggressive compression)
- **3-4 bytes/token**: Good
- **>4 bytes/token**: Limited compression

### Running Custom Benchmarks

#### 1. Building and Running

```bash
# Build benchmark with optimization flags
gcc -O2 -march=native -o bench_custom src/benchmark.c \
    -I./include -lm

# Run benchmark
./bench_custom
```

#### 2. Adding Custom Test Cases

Edit `src/benchmark_luganda.c`:

```c
{
    .name = "my_test_case",
    .text = "text\xE2\x96\x81here",  // ▁ (U+2581) space marker
    .expected_tokens = 2,
    .flags = FLAG_EMOJI | FLAG_MIXED_LANG,
    .description = "Test description"
},
```

#### 3. Interpreting Results

```
=== Benchmark Results ===
Test Case: greeting_simple
  Input: "hello\xE2\x96\x81world"
  Tokens: 2
  Fast-path hits: 2/2 (100%)
  Throughput: 250 tokens/ms
  Avg token length: 2.5 bytes
```

## Performance Expectations by Input

| Input Type | Expected Fast-Path | Throughput | Avg Token Size |
|------------|-------------------|-----------|-----------------|
| Clean Luganda | 80-90% | 150-200 t/ms | 2-3 bytes |
| WhatsApp (mixed) | 40-60% | 100-150 t/ms | 3-4 bytes |
| Noisy (misspellings) | 20-40% | 50-100 t/ms | 3-5 bytes |
| English-heavy | 30-50% | 80-120 t/ms | 2-3 bytes |
| Emoji-heavy | 10-30% | 30-70 t/ms | 4-8 bytes |

## Test Results Analysis

### Using the Analysis Tool

```bash
# Generate automatic analysis
python3 analyze_results.py .

# View markdown report
cat analysis/report_*.md

# View JSON report (for automation)
cat analysis/report_*.json
```

### Key Metrics to Monitor

1. **Test Success Rate**: Target 100%
2. **Fast-Path Coverage**: Target 70%+ for Luganda
3. **Throughput**: Target >100 tokens/ms
4. **Latency**: Target <5μs per token
5. **Memory Usage**: Track for regressions

### Troubleshooting Common Issues

#### Low Test Pass Rate

**Symptoms:**
- Tests failing with corrupted output
- Segmentation faults
- Memory leaks

**Debug Steps:**
```bash
# Run under valgrind
valgrind --leak-check=full ./test/test_simple_tokenizer

# Run with debug symbols
gcc -g -O0 -o test_debug test/test_edge_cases.c
gdb ./test_debug
```

#### Low Fast-Path Ratio

**Symptoms:**
- Many tests using TRIE_FALLBACK
- Below-expected throughput

**Solutions:**
1. Check CV_TO_TOKEN table coverage
2. Add more space_pair combinations
3. Improve Unicode whitespace handling
4. Profile syllable frequency distribution

#### High Latency Variance

**Symptoms:**
- Inconsistent results between runs
- Variable throughput

**Investigation:**
```bash
# Run multiple times and compare
for i in {1..5}; do ./benchmark_luganda; done

# Check for memory pressure
free -h

# Monitor CPU frequency
watch -n1 'cat /proc/cpuinfo | grep MHz'
```

## Continuous Integration

### CI/CD Integration

Add to your CI pipeline:

```yaml
test:
  script:
    - bash run_tests.sh
    - python3 analyze_results.py .
  artifacts:
    - test_results/
    - analysis/

benchmark:
  script:
    - bash run_benchmarks.sh
    - python3 analyze_results.py .
  artifacts:
    - benchmark_results/
    - analysis/
  only:
    - main
```

### Regression Detection

```bash
# Store baseline
cp analysis/report_*.json baseline_metrics.json

# After changes, compare
python3 -c """
import json
with open('baseline_metrics.json') as f:
    baseline = json.load(f)['benchmarks']
# Compare against new results
"""
```

## Performance Profiling

### Using Linux perf

```bash
# Collect performance data
perf record -F 99 -g ./benchmark_luganda

# Analyze
perf report

# Flamegraph
perf script > out.perf-folded
# (requires FlameGraph tools)
```

### CPU Cache Analysis

```bash
# Monitor cache misses
perf stat -e cache-references,cache-misses ./benchmark_luganda

# Branch prediction
perf stat -e branch-misses ./benchmark_luganda
```

## Best Practices

### Test-Driven Development

1. **Write tests first** for new features
2. **Run tests frequently** during development
3. **Benchmark performance-critical code**
4. **Document test expectations**

### Benchmarking Guidelines

1. **Run multiple iterations** to reduce variance
2. **Warm up the CPU** before timing
3. **Disable CPU scaling** for consistent results
4. **Run on dedicated hardware** without background load
5. **Compare apples-to-apples** (same input, same conditions)

### Performance Optimization Strategy

1. **Measure first** before optimizing
2. **Identify bottlenecks** with profiling
3. **Target hot paths** (top 80% of time)
4. **Benchmark improvements** (verify they help)
5. **Avoid premature optimization** (correctness first)

## Resources

- [Fast-Path Optimization Details](analysis_reports/CODE_REVIEW_ANALYSIS.md)
- [Syllabifier Modes](analysis_reports/SYLLABIFIER_MODES.md)
- [Benchmark Guide](analysis_reports/BENCHMARK_GUIDE.md)
- [Developer Guide](analysis_reports/UGANDAN_DEVELOPER_GUIDE.md)

## Summary

The Tokeniser testing and benchmarking infrastructure provides:

✓ Comprehensive unit and integration tests  
✓ Real-world benchmark scenarios (WhatsApp, SMS, etc.)  
✓ Automated result analysis and reporting  
✓ Performance monitoring and regression detection  
✓ Syllabifier mode comparison  
✓ Security and serialization validation  

Use these tools regularly to maintain code quality and performance standards.
