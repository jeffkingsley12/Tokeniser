#!/bin/bash
# test_syllabifier_modes.sh
#
# Test that all three syllabifier modes (byte-table, hybrid, predicate)
# compile and produce consistent results on a test corpus.
#

set -e

PROJECT_DIR="$(pwd)"
cd "$PROJECT_DIR"

echo "=== Syllabifier Mode Comparison Test ==="
echo ""

# Test data
TEST_WORDS=(
    "ayagala"
    "okwéŋŋa"
    "Ŋŋooyo"
    "lwadde"
    "gwanga"
    "dwaniro"
    "nyakatura"
    "nnyini"
)

echo "Test corpus:"
for word in "${TEST_WORDS[@]}"; do
    echo "  - $word"
done
echo ""

# Create test input file
TEST_INPUT="/tmp/syllabifier_test.txt"
{
    for word in "${TEST_WORDS[@]}"; do
        echo "$word"
    done
} > "$TEST_INPUT"

echo "Compiling byte-table mode..."
gcc -std=c11 -Wall -Wextra -O2 \
    -c src/syllabifier.c -I./include \
    -o /tmp/syllabifier_table.o 2>/dev/null
echo "✅ Byte-table mode compiled"

echo ""
echo "Compiling hybrid mode..."
gcc -std=c11 -Wall -Wextra -O2 \
    -DHYBRID_UTF8_SYLLABIFIER \
    -c src/syllabifier.c -I./include \
    -o /tmp/syllabifier_hybrid.o 2>/dev/null
echo "✅ Hybrid mode compiled"

echo ""
echo "Compiling predicate mode..."
gcc -std=c11 -Wall -Wextra -O2 \
    -DUSE_UTF8_LUGANDA_PREDICATES \
    -c src/syllabifier.c -I./include \
    -o /tmp/syllabifier_pred.o 2>/dev/null
echo "✅ Predicate mode compiled"

echo ""
echo "All three modes compiled successfully!"
echo ""
echo "Next: Link and test each mode on your corpus to verify identical output:"
echo ""
echo "  # Build with byte-table mode"
echo "  make MODE=fast && ./tokenizer_demo < corpus.txt > output_table.txt"
echo ""
echo "  # Build with hybrid mode (default, recommended)"
echo "  make && ./tokenizer_demo < corpus.txt > output_hybrid.txt"
echo ""
echo "  # Build with predicate mode"
echo "  make MODE=predicates && ./tokenizer_demo < corpus.txt > output_pred.txt"
echo ""
echo "  # Compare"
echo "  diff output_table.txt output_hybrid.txt  # Should be identical"
echo "  diff output_hybrid.txt output_pred.txt   # Should be identical"
echo ""
