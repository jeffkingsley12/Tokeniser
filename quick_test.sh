#!/bin/bash
##############################################################################
# Quick Test & Benchmark Commands
# ==============================
# Convenient shortcuts for common testing tasks
##############################################################################

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

# Color codes
BLUE='\033[0;34m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

print_menu() {
    echo ""
    echo -e "${BLUE}╔════════════════════════════════════════════════════════╗${NC}"
    echo -e "${BLUE}║  Tokeniser Testing & Benchmarking Quick Reference     ║${NC}"
    echo -e "${BLUE}╚════════════════════════════════════════════════════════╝${NC}"
    echo ""
    echo "Tests:"
    echo "  1. Run all tests          : bash run_tests.sh"
    echo "  2. Run single test        : ./test/test_simple_tokenizer"
    echo "  3. View test results      : tail -100 test_results/test_report_*.txt"
    echo ""
    echo "Benchmarks:"
    echo "  4. Run all benchmarks     : bash run_benchmarks.sh"
    echo "  5. Run Luganda benchmark  : ./benchmark_luganda (if available)"
    echo "  6. View benchmark results : ls -lh benchmark_results/"
    echo ""
    echo "Analysis:"
    echo "  7. Analyze results        : python3 analyze_results.py ."
    echo "  8. View analysis report   : cat analysis/report_*.md"
    echo "  9. View JSON report       : cat analysis/report_*.json | jq ."
    echo ""
    echo "Development:"
    echo "  10. Clean artifacts       : rm -rf test_results benchmark_results analysis"
    echo "  11. View test source      : cat test/test_simple_tokenizer.c"
    echo "  12. View benchmark source : cat src/benchmark_luganda.c"
    echo ""
    echo "Documentation:"
    echo "  13. Testing guide         : cat TESTING_BENCHMARKING_GUIDE.md"
    echo "  14. Benchmark guide       : cat analysis_reports/BENCHMARK_GUIDE.md"
    echo ""
}

case "$1" in
    1|all-tests)
        echo -e "${GREEN}Running all tests...${NC}"
        bash run_tests.sh
        ;;
    2|simple-test)
        echo -e "${GREEN}Running simple tokenizer test...${NC}"
        if [[ -x test/test_simple_tokenizer ]]; then
            ./test/test_simple_tokenizer
        else
            echo "test_simple_tokenizer not found or not executable"
        fi
        ;;
    3|view-tests)
        echo -e "${GREEN}Latest test results:${NC}"
        if [[ -f test_results/test_report_*.txt ]]; then
            tail -100 test_results/test_report_*.txt | sort -r | head -100
        else
            echo "No test results found. Run tests first."
        fi
        ;;
    4|all-bench)
        echo -e "${GREEN}Running all benchmarks...${NC}"
        bash run_benchmarks.sh
        ;;
    5|luganda-bench)
        echo -e "${GREEN}Running Luganda benchmark...${NC}"
        if [[ -x ./benchmark_luganda ]]; then
            ./benchmark_luganda
        else
            echo "benchmark_luganda binary not found"
        fi
        ;;
    6|view-bench)
        echo -e "${GREEN}Available benchmarks:${NC}"
        ls -lh benchmark_results/ 2>/dev/null || echo "No benchmarks run yet"
        ;;
    7|analyze)
        echo -e "${GREEN}Analyzing results...${NC}"
        python3 analyze_results.py .
        ;;
    8|report)
        echo -e "${GREEN}Latest analysis report:${NC}"
        if [[ -f analysis/report_*.md ]]; then
            less analysis/report_*.md
        else
            echo "No analysis reports found. Run analyze_results.py first."
        fi
        ;;
    9|json-report)
        echo -e "${GREEN}Latest JSON report:${NC}"
        if command -v jq &> /dev/null; then
            cat analysis/report_*.json 2>/dev/null | jq . || echo "No JSON reports found"
        else
            cat analysis/report_*.json 2>/dev/null || echo "No JSON reports found (install jq for pretty printing)"
        fi
        ;;
    10|clean)
        echo -e "${YELLOW}Cleaning test artifacts...${NC}"
        rm -rf test_results benchmark_results analysis
        echo "Cleaned."
        ;;
    11|view-test-src)
        echo -e "${GREEN}Simple tokenizer test source:${NC}"
        cat test/test_simple_tokenizer.c | head -50
        echo "... (see test/test_simple_tokenizer.c for full source)"
        ;;
    12|view-bench-src)
        echo -e "${GREEN}Benchmark source:${NC}"
        if [[ -f src/benchmark_luganda.c ]]; then
            head -50 src/benchmark_luganda.c
            echo "... (see src/benchmark_luganda.c for full source)"
        fi
        ;;
    13|guide)
        echo -e "${GREEN}Testing & Benchmarking Guide:${NC}"
        less TESTING_BENCHMARKING_GUIDE.md
        ;;
    14|bench-guide)
        echo -e "${GREEN}Benchmark Guide:${NC}"
        less analysis_reports/BENCHMARK_GUIDE.md
        ;;
    *)
        print_menu
        if [[ -n "$1" ]]; then
            echo -e "${YELLOW}Unknown command: $1${NC}"
        fi
        ;;
esac
