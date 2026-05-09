#!/bin/bash
##############################################################################
# Model Setup for Testing
# ======================
# Create or prepare tokenizer models for running tests
##############################################################################

# This script demonstrates how to set up models for testing

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

##############################################################################
# Option 1: Use Existing Test Model
##############################################################################

use_test_model() {
    echo "Setting up test model..."
    
    # The test directory has a security test model
    if [[ -f test/test_security_model.bin ]]; then
        ln -sf test/test_security_model.bin tokenizer_model.bin
        echo "✓ Linked test model: tokenizer_model.bin -> test/test_security_model.bin"
    else
        echo "✗ Test model not found"
        return 1
    fi
}

##############################################################################
# Option 2: Build Model from Corpus
##############################################################################

build_model_from_corpus() {
    echo "Building tokenizer model from corpus..."
    
    # Check if main.c demo exists
    if [[ ! -f src/main.c ]]; then
        echo "✗ src/main.c not found"
        return 1
    fi
    
    # Compile the demo
    echo "Compiling tokenizer demo..."
    gcc -O2 -o tokenizer_demo src/main.c src/*.c \
        -I./include -lm -pthread 2>/dev/null || {
        echo "Note: Full compilation may require additional setup"
        return 1
    }
    
    # Run demo to generate model
    echo "Generating model..."
    ./tokenizer_demo
    echo "✓ Model generation complete"
}

##############################################################################
# Option 3: Create Minimal Test Model
##############################################################################

create_minimal_model() {
    echo "Creating minimal test model..."
    
    # Create a simple binary stub for testing
    # This is a placeholder - real model would be more complex
    {
        printf '\x54\x4f\x4b\x4e'  # "TOKN" magic
        printf '\x00\x00\x00\x01'  # Version 1
        printf '\x00\x00\x00\x00'  # Placeholder
    } > tokenizer_model.bin
    
    echo "✓ Created minimal model (stub)"
    ls -lh tokenizer_model.bin
}

##############################################################################
# Main Menu
##############################################################################

show_menu() {
    echo ""
    echo "╔════════════════════════════════════════════════════════╗"
    echo "║  Tokeniser Model Setup                                 ║"
    echo "╚════════════════════════════════════════════════════════╝"
    echo ""
    echo "Options:"
    echo "  1. Use existing test model"
    echo "  2. Build from corpus (experimental)"
    echo "  3. Create minimal stub model"
    echo ""
}

case "${1:-0}" in
    1|test)
        use_test_model
        ;;
    2|build)
        build_model_from_corpus
        ;;
    3|stub)
        create_minimal_model
        ;;
    *)
        show_menu
        echo "Usage: bash setup_model.sh [1|2|3|test|build|stub]"
        exit 1
        ;;
esac

echo ""
echo "Model setup: $(ls -lh tokenizer_model.bin 2>/dev/null || echo 'Not found')"
echo ""
echo "Next: Run tests with 'bash run_tests.sh'"
