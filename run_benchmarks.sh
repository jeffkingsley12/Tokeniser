#!/bin/bash
##############################################################################
# Benchmark Runner for Tokeniser
# ==============================
# Runs benchmarks, collects performance metrics, and generates analysis
##############################################################################

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="$PROJECT_ROOT/build"
RESULTS_DIR="$PROJECT_ROOT/benchmark_results"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
REPORT="$RESULTS_DIR/benchmark_report_$TIMESTAMP.txt"

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

mkdir -p "$RESULTS_DIR" "$BUILD_DIR"

##############################################################################
# Logging
##############################################################################

log_header() {
    echo ""
    echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
    echo -e "${BLUE}  $1${NC}"
    echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
    echo "" | tee -a "$REPORT"
}

log_info() {
    echo "  $1" | tee -a "$REPORT"
}

log_result() {
    echo -e "${GREEN}✓ $1${NC}" | tee -a "$REPORT"
}

##############################################################################
# Benchmark Runners
##############################################################################

build_benchmark() {
    local source="$1"
    local binary="$2"
    local description="$3"

    echo -e "${YELLOW}Building: $description${NC}"
    
    if [[ ! -f "$source" ]]; then
        echo -e "${RED}Source not found: $source${NC}"
        return 1
    fi

    # Try to compile with different flags
    local compile_ok=0
    
    # Attempt 1: with available headers
    if gcc -O2 -march=native -o "$binary" "$source" \
        -I"$PROJECT_ROOT/include" \
        2>/dev/null; then
        compile_ok=1
    fi

    # Attempt 2: minimal compile
    if [[ $compile_ok -eq 0 ]] && gcc -O2 -o "$binary" "$source" \
        2>/dev/null; then
        compile_ok=1
    fi

    if [[ $compile_ok -eq 1 ]]; then
        echo -e "${GREEN}✓ Built successfully${NC}"
        return 0
    else
        echo -e "${YELLOW}⊘ Could not compile (may need additional dependencies)${NC}"
        return 1
    fi
}

run_benchmark() {
    local binary="$1"
    local description="$2"
    
    echo -e "${YELLOW}▶ $description${NC}" | tee -a "$REPORT"

    if [[ ! -x "$binary" ]]; then
        echo -e "${YELLOW}⊘ Benchmark binary not available${NC}" | tee -a "$REPORT"
        return 1
    fi

    local output_file="$RESULTS_DIR/$(basename "$binary")_$TIMESTAMP.txt"
    
    if "$binary" > "$output_file" 2>&1; then
        log_result "Completed: $description"
        echo "" | tee -a "$REPORT"
        cat "$output_file" | sed 's/^/    /' | tee -a "$REPORT"
        return 0
    else
        echo -e "${RED}✗ Failed to run: $description${NC}" | tee -a "$REPORT"
        return 1
    fi
}

##############################################################################
# System Information
##############################################################################

capture_system_info() {
    log_header "System Information"

    log_info "OS: $(uname -s)"
    log_info "Kernel: $(uname -r)"
    log_info "CPU: $(lscpu | grep 'Model name' | cut -d: -f2 | xargs)"
    log_info "Cores: $(nproc)"
    
    local mem_kb=$(grep MemTotal /proc/meminfo | awk '{print $2}')
    local mem_gb=$((mem_kb / 1024 / 1024))
    log_info "Memory: ${mem_gb} GB"
    
    log_info "Compiler: $(gcc --version | head -1)"
    log_info "CFLAGS: -O2 -march=native"
}

##############################################################################
# Benchmark Collection
##############################################################################

run_all_benchmarks() {
    log_header "Benchmark Suite"

    echo "Starting benchmarks at $(date)" | tee -a "$REPORT"
    echo ""

    # Check for pre-built benchmarks
    if [[ -x "$PROJECT_ROOT/bench" ]]; then
        run_benchmark "$PROJECT_ROOT/bench" "Pre-built benchmark"
    fi

    if [[ -x "$PROJECT_ROOT/benchmark_luganda" ]]; then
        run_benchmark "$PROJECT_ROOT/benchmark_luganda" "Luganda-specific benchmark"
    fi

    # Try building and running benchmark.c
    if [[ -f "$PROJECT_ROOT/src/benchmark.c" ]]; then
        local bench_binary="$BUILD_DIR/benchmark_main"
        if build_benchmark "$PROJECT_ROOT/src/benchmark.c" "$bench_binary" "benchmark.c"; then
            run_benchmark "$bench_binary" "Benchmark Main Suite"
        fi
    fi

    # Try building and running benchmark_luganda.c
    if [[ -f "$PROJECT_ROOT/src/benchmark_luganda.c" ]]; then
        local bench_lug_binary="$BUILD_DIR/benchmark_luganda_main"
        if build_benchmark "$PROJECT_ROOT/src/benchmark_luganda.c" "$bench_lug_binary" "benchmark_luganda.c"; then
            run_benchmark "$bench_lug_binary" "Luganda Benchmark Suite"
        fi
    fi
}

##############################################################################
# Performance Analysis
##############################################################################

analyze_results() {
    log_header "Performance Analysis"

    log_info "Benchmark files generated:"
    for f in "$RESULTS_DIR"/*_$TIMESTAMP.txt; do
        if [[ -f "$f" ]]; then
            local size=$(wc -l < "$f")
            log_info "  - $(basename "$f") ($size lines)"
        fi
    done

    log_info ""
    log_info "Key Performance Indicators:"
    log_info "  • Throughput (tokens/ms): Measure of tokenization speed"
    log_info "  • Fast-path ratio (%): Percentage of direct lookups vs trie traversal"
    log_info "  • Cycles/token: CPU cycles per token"
    log_info "  • Token compression (bytes/token): Memory efficiency"
}

##############################################################################
# Report Generation
##############################################################################

generate_final_report() {
    log_header "Benchmark Report"
    
    echo "Generated: $(date)" | tee -a "$REPORT"
    echo "Duration: Benchmark completed" | tee -a "$REPORT"
    echo ""
    echo "Next Steps:" | tee -a "$REPORT"
    echo "  1. Review detailed results in benchmark_results/" | tee -a "$REPORT"
    echo "  2. Compare against baseline metrics" | tee -a "$REPORT"
    echo "  3. Identify optimization opportunities" | tee -a "$REPORT"
    echo ""
    echo "Report: $REPORT" | tee -a "$REPORT"
}

##############################################################################
# Main
##############################################################################

main() {
    echo "Tokeniser Benchmark Suite" | tee "$REPORT"
    echo "Timestamp: $TIMESTAMP" | tee -a "$REPORT"
    echo ""

    capture_system_info
    run_all_benchmarks
    analyze_results
    generate_final_report

    echo ""
    echo -e "${GREEN}Benchmark suite completed!${NC}"
    echo "Results: $REPORT"
    echo "Details: $RESULTS_DIR/"
}

main "$@"
