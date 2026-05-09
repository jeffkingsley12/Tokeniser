#!/bin/bash
##############################################################################
# Tokeniser Testing Suite
# =======================
# Runs all available tests, collects results, and generates a summary report
##############################################################################

set +e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEST_DIR="$PROJECT_ROOT/test"
RESULTS_DIR="$PROJECT_ROOT/test_results"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
REPORT="$RESULTS_DIR/test_report_$TIMESTAMP.txt"

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Create results directory
mkdir -p "$RESULTS_DIR"

# Counters
PASSED=0
FAILED=0
SKIPPED=0

##############################################################################
# Logging functions
##############################################################################

log_header() {
    echo ""
    echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
    echo -e "${BLUE}  $1${NC}"
    echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
    echo "" | tee -a "$REPORT"
}

log_test() {
    echo -e "${YELLOW}▶ $1${NC}" | tee -a "$REPORT"
}

log_pass() {
    echo -e "${GREEN}✓ PASS: $1${NC}" | tee -a "$REPORT"
    ((PASSED++))
}

log_fail() {
    echo -e "${RED}✗ FAIL: $1${NC}" | tee -a "$REPORT"
    ((FAILED++))
}

log_skip() {
    echo -e "${YELLOW}⊘ SKIP: $1${NC}" | tee -a "$REPORT"
    ((SKIPPED++))
}

log_info() {
    echo "  $1" | tee -a "$REPORT"
}

##############################################################################
# Test functions
##############################################################################

run_unit_test() {
    local test_name="$1"
    local test_binary="$2"
    local test_description="$3"

    log_test "$test_description"

    if [[ ! -x "$test_binary" ]]; then
        log_skip "$test_name (binary not found or not executable)"
        return 0
    fi

    local output_file="$RESULTS_DIR/${test_name}_output_$TIMESTAMP.txt"

    # Run test and capture output
    if "$test_binary" > "$output_file" 2>&1; then
        local line_count=$(wc -l < "$output_file")
        log_pass "$test_name ($line_count lines of output)"
        head -20 "$output_file" | sed 's/^/    /' | tee -a "$REPORT"
        echo "    ..." | tee -a "$REPORT"
    else
        local exit_code=$?
        log_fail "$test_name (exit code: $exit_code)"
        head -20 "$output_file" | sed 's/^/    /' | tee -a "$REPORT"
        echo "    ... (see $output_file for full output)" | tee -a "$REPORT"
    fi
}

run_shell_test() {
    local test_name="$1"
    local test_script="$2"
    local test_description="$3"

    log_test "$test_description"

    if [[ ! -x "$test_script" ]]; then
        log_skip "$test_name (script not found or not executable)"
        return 0
    fi

    local output_file="$RESULTS_DIR/${test_name}_output_$TIMESTAMP.txt"

    if bash "$test_script" > "$output_file" 2>&1; then
        log_pass "$test_name"
        head -20 "$output_file" | sed 's/^/    /' | tee -a "$REPORT"
    else
        local exit_code=$?
        log_fail "$test_name (exit code: $exit_code)"
        head -20 "$output_file" | sed 's/^/    /' | tee -a "$REPORT"
    fi
}

##############################################################################
# Main Test Suite
##############################################################################

main() {
    echo "Tokeniser Test Suite" | tee "$REPORT"
    echo "Timestamp: $TIMESTAMP" | tee -a "$REPORT"
    echo "Project: $PROJECT_ROOT" | tee -a "$REPORT"
    echo ""

    log_header "Unit Tests"

    # Core functionality tests
    run_unit_test "simple_tokenizer" \
        "$TEST_DIR/test_simple_tokenizer" \
        "Simple Tokenizer (requires model file)"

    run_unit_test "benchmark_simple" \
        "$TEST_DIR/test_benchmark_simple" \
        "Benchmark Simple (performance baseline)"

    run_unit_test "edge_cases" \
        "$TEST_DIR/test_edge_cases_debug" \
        "Edge Cases Test"

    run_unit_test "serialization_security" \
        "$TEST_DIR/test_serialization_security" \
        "Serialization & Security"

    run_unit_test "corruption_minimal" \
        "$TEST_DIR/test_corruption_minimal" \
        "Corruption/Robustness Test"

    run_unit_test "fixes" \
        "$TEST_DIR/test_fixes" \
        "Regression Fixes Test"

    log_header "Integration Tests"

    run_shell_test "syllabifier_modes" \
        "$TEST_DIR/test_syllabifier_modes.sh" \
        "Syllabifier Modes Test (Louds/Viterbi/Hybrid)"

    log_header "Source Code Tests"

    # List available test source files
    log_info "Available test source files:"
    for f in "$TEST_DIR"/test_*.c; do
        if [[ -f "$f" ]]; then
            local fname=$(basename "$f")
            local lines=$(wc -l < "$f")
            log_info "  - $fname ($lines lines)"
        fi
    done

    log_header "Test Summary"

    echo ""
    echo "Test Results:" | tee -a "$REPORT"
    echo "  ${GREEN}Passed:${NC}  $PASSED" | tee -a "$REPORT"
    echo "  ${RED}Failed:${NC}  $FAILED" | tee -a "$REPORT"
    echo "  ${YELLOW}Skipped:${NC} $SKIPPED" | tee -a "$REPORT"
    local TOTAL=$((PASSED + FAILED + SKIPPED))
    echo "  Total:   $TOTAL" | tee -a "$REPORT"
    echo ""

    if [[ $FAILED -eq 0 ]]; then
        echo -e "${GREEN}All tests passed!${NC}" | tee -a "$REPORT"
    else
        echo -e "${RED}Some tests failed. Review the report for details.${NC}" | tee -a "$REPORT"
    fi

    echo ""
    echo "Report saved to: $REPORT"
    echo "Test outputs in: $RESULTS_DIR/"
}

main "$@"
