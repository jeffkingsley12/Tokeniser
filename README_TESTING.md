#!/bin/bash
# README for Testing & Benchmarking
# Display overview on stdout

cat << 'EOF'

╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║               TOKENISER TESTING & BENCHMARKING INFRASTRUCTURE                ║
║                                                                              ║
║                        Comprehensive Guide & Quick Start                     ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝


TABLE OF CONTENTS
═════════════════════════════════════════════════════════════════════════════

1. QUICK START (30 seconds)
2. INFRASTRUCTURE OVERVIEW
3. TEST SUITE DETAILS
4. BENCHMARK FRAMEWORK
5. ANALYSIS & REPORTING
6. ADVANCED USAGE


═════════════════════════════════════════════════════════════════════════════
1. QUICK START (30 SECONDS)
═════════════════════════════════════════════════════════════════════════════

First Time Setup:
  1. bash setup_model.sh 1              # Set up test model
  2. bash run_tests.sh                  # Run all tests
  3. cat test_results/test_report_*.txt # View results

One-Command Testing:
  bash quick_test.sh                    # Show menu
  bash quick_test.sh 1                  # Run all tests
  bash quick_test.sh 7                  # Analyze results

Full Workflow:
  bash run_tests.sh                     # Unit tests
  bash run_benchmarks.sh                # Benchmarks
  python3 analyze_results.py .          # Analysis
  cat analysis/report_*.md              # View report


═════════════════════════════════════════════════════════════════════════════
2. INFRASTRUCTURE OVERVIEW
═════════════════════════════════════════════════════════════════════════════

Test & Benchmark Architecture:

    ┌─────────────────────────────────────────────────────────────────────┐
    │                      TESTING FRAMEWORK                              │
    ├─────────────────────────────────────────────────────────────────────┤
    │                                                                     │
    │  run_tests.sh ──┬─→ [Test Binaries] ──→ test_results/             │
    │                │   (7 unit tests)        ├─ test_report_*.txt      │
    │  run_benchmarks.sh ──┬─→ [Benchmarks] ──→ benchmark_results/      │
    │                     │   (multiple)        ├─ *.txt                 │
    │  quick_test.sh ──────┴─────────────────┤                          │
    │                                         │                          │
    │  analyze_results.py ──────────────────→ analysis/                 │
    │                                         ├─ report_*.md            │
    │                                         ├─ report_*.json          │
    │                                                                    │
    └────────────────────────────────────────────────────────────────────┘


Test Pipeline:

    Source Code             Compiled              Run                Results
        │                      │                  │                   │
    test_*.c ────→ [Compile] ─→ test_binary ─→ ./run_tests.sh ─→ test_results/
        │                      │                  │
        │                      └─→ (7 test binaries)
        │
    src/*.c ──────→ [Link] ────→ objects ──────────────────────→ Test Stats
        │
    include/*.h


Result Aggregation:

    test_results/                benchmark_results/           analysis/
    ├─ test_report_*.txt        ├─ benchmark_report_*.txt    ├─ report_*.md
    ├─ *.c_output_*.txt         ├─ *.txt                     └─ report_*.json
    └─ ...                      └─ ...
            │                           │
            └──────────────────┬────────┘
                               │
                        analyze_results.py
                               │
                        (Parse & Extract)
                               │
                               ▼
                        JSON Report
                               │
                               ▼
                        Markdown Summary


═════════════════════════════════════════════════════════════════════════════
3. TEST SUITE DETAILS
═════════════════════════════════════════════════════════════════════════════

Unit Tests (7 binaries):

  TEST                          SIZE    TYPE          COVERAGE
  ───────────────────────────────────────────────────────────────
  test_simple_tokenizer         387 KB  API/Functional  Core tokenization
  test_benchmark_simple         387 KB  Perf/Baseline   Basic performance
  test_edge_cases_debug         427 KB  Edge/Robustness Boundary conditions
  test_edge_cases_fixed         387 KB  Regression      Fixed issues
  test_serialization_security   414 KB  Integration     Data integrity
  test_corruption_minimal       395 KB  Robustness      Error handling
  test_fixes                    387 KB  Regression      Known fixes

Integration Tests:

  test_syllabifier_modes.sh ─→ Tests three modes:
                               ├─ Louds-only (fast path)
                               ├─ Viterbi (probabilistic)
                               └─ Hybrid (adaptive)

Running Individual Tests:

  # Direct execution
  ./test/test_simple_tokenizer
  ./test/test_edge_cases_debug
  
  # With analysis
  bash quick_test.sh 2              # Simple test
  bash quick_test.sh 3              # View test results


═════════════════════════════════════════════════════════════════════════════
4. BENCHMARK FRAMEWORK
═════════════════════════════════════════════════════════════════════════════

Performance Metrics:

  Metric              Formula                Good Range      Type
  ────────────────────────────────────────────────────────────────
  Throughput          tokens/ms               >100            Speed
  Fast-path %         direct lookups %        70-90%          Quality
  Cycles/token        CPU cycles/token        <1000           Efficiency
  Compression         bytes/token             2-4             Density
  Latency             time per token          <5μs            Responsiveness

Expected Performance by Input:

  Input Type           Fast-path    Throughput      Bytes/token
  ─────────────────────────────────────────────────────────────
  Clean Luganda        80-90%       150-200 t/ms    2-3
  WhatsApp mixed       40-60%       100-150 t/ms    3-4
  Noisy (typos)        20-40%       50-100 t/ms     3-5
  English-heavy        30-50%       80-120 t/ms     2-3
  Emoji-heavy          10-30%       30-70 t/ms      4-8

Benchmark Test Cases:

  greeting_simple         → Basic greetings with space markers
  whatsapp_mixed          → Real WhatsApp message patterns
  loanwords_french        → French loanwords in context
  long_message            → 100+ syllable messages
  unicode_whitespace      → NBSP, ZWSP variations
  emoji_sequence          → Complex emoji patterns
  ... and 9 more cases

Running Benchmarks:

  # All benchmarks
  bash run_benchmarks.sh
  
  # Individual benchmark (if available)
  ./benchmark_luganda
  ./bench
  
  # With timing
  time ./benchmark_luganda
  
  # With performance analysis
  perf record -F 99 -g ./benchmark_luganda
  perf report


═════════════════════════════════════════════════════════════════════════════
5. ANALYSIS & REPORTING
═════════════════════════════════════════════════════════════════════════════

Analysis Tool Features:

  python3 analyze_results.py
  │
  ├─→ Parse test outputs
  │   ├─ Count passed/failed tests
  │   ├─ Extract error messages
  │   └─ Calculate success rate
  │
  ├─→ Extract benchmark metrics
  │   ├─ Throughput (tokens/ms)
  │   ├─ Fast-path ratio (%)
  │   ├─ Cycles per token
  │   └─ Latency measurements
  │
  └─→ Generate reports
      ├─ Markdown (human readable)
      ├─ JSON (machine parseable)
      └─ Statistics/summaries

Report Contents:

  Markdown Report (report_*.md):
  ├─ Test Summary
  │  ├─ Total tests, passed, failed
  │  ├─ Success rate percentage
  │  └─ Individual test status
  ├─ Benchmark Results
  │  ├─ Performance metrics
  │  ├─ Metric comparison
  │  └─ Performance tiers
  └─ Recommendations

  JSON Report (report_*.json):
  ├─ Timestamp
  ├─ tests
  │  ├─ total, passed, failed
  │  └─ test_name → {passed, output_lines, error}
  └─ benchmarks
     ├─ benchmark_name → {metrics, file, timestamp}
     └─ metrics → {name, value, unit}

Viewing Reports:

  # View markdown report
  cat analysis/report_*.md | less
  
  # View JSON with formatting
  cat analysis/report_*.json | jq .
  
  # Extract specific metric
  jq '.benchmarks."benchmark_luganda".metrics[] | select(.name=="throughput")' *.json

Regression Detection:

  # Compare against baseline
  diff <(jq '.tests.failed' baseline.json) \
       <(jq '.tests.failed' current.json)


═════════════════════════════════════════════════════════════════════════════
6. ADVANCED USAGE
═════════════════════════════════════════════════════════════════════════════

6.1 CONTINUOUS INTEGRATION

Adding to CI/CD (GitHub Actions):

  name: Test & Benchmark
  on: [push, pull_request]
  jobs:
    test:
      runs-on: ubuntu-latest
      steps:
        - uses: actions/checkout@v2
        - name: Run tests
          run: bash run_tests.sh
        - name: Run benchmarks
          run: bash run_benchmarks.sh
        - name: Analyze results
          run: python3 analyze_results.py .
        - name: Upload reports
          uses: actions/upload-artifact@v2
          with:
            name: test-reports
            path: test_results/
            retention-days: 30

6.2 PERFORMANCE PROFILING

CPU Profiling:

  # Profile with perf
  perf record -F 99 -g ./benchmark_luganda
  perf report
  
  # Flame graph
  perf script > out.perf-folded
  (requires FlameGraph tools)

Memory Profiling:

  # Detect memory leaks
  valgrind --leak-check=full ./test_simple_tokenizer
  
  # Track allocations
  valgrind --tool=massif ./benchmark_luganda

Cache Analysis:

  # Monitor cache performance
  perf stat -e cache-references,cache-misses ./benchmark_luganda
  
  # Branch prediction
  perf stat -e branch-misses ./benchmark_luganda

6.3 CUSTOM TEST CASES

Adding Tests:

  1. Create test file: test/test_my_feature.c
  2. Implement test logic with assertions
  3. Compile: gcc -o test/test_my_feature test/test_my_feature.c ...
  4. Tests auto-discovered by run_tests.sh

Adding Benchmark Cases:

  Edit src/benchmark_luganda.c and add:
  
  {
      .name = "my_case",
      .text = "test\xE2\x96\x81text",
      .expected_tokens = 2,
      .flags = FLAG_EMOJI,
      .description = "Custom test"
  },

6.4 BASELINE COMPARISON

Create Baseline:

  # Run tests and save results
  bash run_tests.sh && bash run_benchmarks.sh
  mkdir baseline_$(date +%Y%m%d)
  cp -r test_results benchmark_results analysis baseline_$(date +%Y%m%d)/

Compare Current vs Baseline:

  # Extract metrics
  python3 << 'PYEOF'
  import json, sys
  from pathlib import Path
  
  baseline = json.load(open('baseline_20260501/analysis/report_*.json'))
  current = json.load(open('analysis/report_*.json'))
  
  # Compare test results
  if current['tests']['failed'] > baseline['tests']['failed']:
      print(f"REGRESSION: {current['tests']['failed']} vs {baseline['tests']['failed']}")
      sys.exit(1)
  
  print("All checks passed!")
  PYEOF


═════════════════════════════════════════════════════════════════════════════
QUICK COMMAND REFERENCE
═════════════════════════════════════════════════════════════════════════════

Setup & Configuration:
  bash setup_model.sh 1                    # Use test model
  bash setup_model.sh 2                    # Build from corpus

Testing:
  bash run_tests.sh                        # All tests
  ./test/test_simple_tokenizer             # Single test
  bash quick_test.sh 1                     # Menu option

Benchmarking:
  bash run_benchmarks.sh                   # All benchmarks
  ./benchmark_luganda                      # Luganda benchmark
  bash quick_test.sh 5                     # Menu option

Analysis:
  python3 analyze_results.py .             # Generate reports
  cat analysis/report_*.md                 # View markdown
  cat analysis/report_*.json | jq .        # View JSON

Profiling:
  perf record -F 99 ./benchmark_luganda    # Record profile
  perf report                              # View profile
  valgrind --leak-check=full ./test_*     # Check for leaks

Documentation:
  less TESTING_BENCHMARKING_GUIDE.md       # Main guide
  less TESTING_SETUP_SUMMARY.md            # Setup summary
  bash quick_test.sh 13                    # View guide


═════════════════════════════════════════════════════════════════════════════
FILES CREATED
═════════════════════════════════════════════════════════════════════════════

Scripts:
  ✓ run_tests.sh                   (5.9 KB)  Main test runner
  ✓ run_benchmarks.sh              (7.2 KB)  Benchmark runner
  ✓ analyze_results.py             (9.5 KB)  Analysis tool
  ✓ quick_test.sh                  (3.5 KB)  Quick menu
  ✓ setup_model.sh                 (2 KB)    Model setup

Documentation:
  ✓ TESTING_BENCHMARKING_GUIDE.md  (8 KB)    Comprehensive guide
  ✓ TESTING_SETUP_SUMMARY.md       (12 KB)   Setup summary
  ✓ README_TESTING.md              (this)    Quick reference


═════════════════════════════════════════════════════════════════════════════
NEXT STEPS
═════════════════════════════════════════════════════════════════════════════

1. First Run:
   bash setup_model.sh 1
   bash run_tests.sh

2. Explore Results:
   cat test_results/test_report_*.txt
   bash quick_test.sh 7

3. Read Documentation:
   less TESTING_BENCHMARKING_GUIDE.md

4. Integrate with Your Workflow:
   - Add scripts to version control
   - Configure CI/CD to run tests
   - Set performance baselines
   - Monitor for regressions

5. Customize for Your Needs:
   - Add custom test cases
   - Configure benchmark parameters
   - Set threshold alerts
   - Integrate with tools/dashboards


═════════════════════════════════════════════════════════════════════════════

For detailed information: cat TESTING_BENCHMARKING_GUIDE.md
For setup summary: cat TESTING_SETUP_SUMMARY.md
For quick commands: bash quick_test.sh

EOF
