# Luganda Tokenizer Makefile
# ==========================

# Compiler settings
CC ?= gcc
CFLAGS = -Wall -Wextra -O2 -g -fPIC -MMD -MP -D_GNU_SOURCE \
         -I$(INC_DIR) \
         -I$(SRC_DIR)/lg_engine/include \
         -I$(SRC_DIR)/lg_engine/src/enhanced
CFLAGS_DEBUG = -Wall -Wextra -g -O0 -DDEBUG -fPIC -MMD -MP -D_GNU_SOURCE \
               -I$(INC_DIR) \
               -I$(SRC_DIR)/lg_engine/include \
               -I$(SRC_DIR)/lg_engine/src/enhanced
LDFLAGS = -lm

# Directories
SRC_DIR = src
INC_DIR = include
BUILD_DIR = build
BIN_DIR = bin
TEST_DIR = test
DATA_DIR = data
VOCAB_DIR = vocab

# Include paths
INCLUDES = -I$(INC_DIR)

# Source files (excluding tests)
CORE_SOURCES = \
	$(SRC_DIR)/tokenizer.c \
	$(SRC_DIR)/grammar.c \
	$(SRC_DIR)/louds.c \
	$(SRC_DIR)/repair.c \
	$(SRC_DIR)/syllabifier.c \
	$(SRC_DIR)/syllabifier_seed.c \
	$(SRC_DIR)/stream_tokenizer.c \
	$(SRC_DIR)/corpus_utils.c \
	$(SRC_DIR)/truth_layer.c \
	$(SRC_DIR)/tokenizer_beam.c

MAIN_SOURCES = $(CORE_SOURCES) $(SRC_DIR)/main.c

BENCH_SOURCES = $(CORE_SOURCES) \
	$(SRC_DIR)/benchmark.c

MMAP_SOURCES = $(CORE_SOURCES) $(SRC_DIR)/tokenizer_mmap.c

# Object files
MAIN_OBJS = $(patsubst $(SRC_DIR)/%.c,$(BUILD_DIR)/%.o,$(MAIN_SOURCES))
BENCH_OBJS = $(patsubst $(SRC_DIR)/%.c,$(BUILD_DIR)/%.o,$(BENCH_SOURCES))
MMAP_OBJS = $(patsubst $(SRC_DIR)/%.c,$(BUILD_DIR)/%.o,$(MMAP_SOURCES))

# Test source files
TEST_SOURCES = $(filter-out $(TEST_DIR)/test_edge_cases.c, $(wildcard $(TEST_DIR)/test_*.c))
TEST_BINARIES = $(patsubst $(TEST_DIR)/%.c,$(TEST_DIR)/%,$(TEST_SOURCES))

# Dependency files
DEPS = $(MAIN_OBJS:.o=.d) $(BENCH_OBJS:.o=.d) $(MMAP_OBJS:.o=.d)

# Default target
.PHONY: all clean debug test bench install dirs help asan

all: dirs tokenizer_demo bench lib test_bins engine_tools

# Engine source files and includes
ENGINE_INC = -I$(INC_DIR) -I$(SRC_DIR)/lg_engine/include -I$(SRC_DIR)/lg_engine/src -I$(SRC_DIR)/lg_engine/src/enhanced

# Find all engine source files recursively (excluding demo and test)
ENGINE_ALL_SRC = $(shell find $(SRC_DIR)/lg_engine/src -name "*.c" ! -path "*/demo/*" ! -path "*/test/*")
ENGINE_ALL_OBJ = $(patsubst $(SRC_DIR)/lg_engine/src/%.c,$(BUILD_DIR)/lg_engine/%.o,$(ENGINE_ALL_SRC))

LIBGEMINI_CORE = libgemini_core.a

# Build static library
$(LIBGEMINI_CORE): $(ENGINE_ALL_OBJ)
	ar rcs $(BUILD_DIR)/$@ $^

# Compile engine sources into objects
$(BUILD_DIR)/lg_engine/%.o: $(SRC_DIR)/lg_engine/src/%.c | dirs
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) $(ENGINE_INC) -c $< -o $@

# Detect AVX2 support on the compiling machine
AVX2_SUPPORTED := $(shell grep -q avx2 /proc/cpuinfo && echo yes || echo no)

ifeq ($(AVX2_SUPPORTED),yes)
    AVX2_FLAGS = -mavx2
else
    AVX2_FLAGS =
endif

# Compile engine_lifecycle.c with AVX2 support for SIMD decay pass (if available)
$(BUILD_DIR)/lg_engine/engine_lifecycle.o: $(SRC_DIR)/lg_engine/src/engine_lifecycle.c | dirs
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) $(ENGINE_INC) $(AVX2_FLAGS) -c $< -o $@
	
engine_tools: lib $(LIBGEMINI_CORE) dirs
	$(CC) $(CFLAGS) $(ENGINE_INC) -o $(BIN_DIR)/online_learning $(SRC_DIR)/lg_engine/src/demo/online_learning.c $(BUILD_DIR)/$(LIBGEMINI_CORE) -L$(BUILD_DIR) -lluganda_tok -Wl,-rpath,'$$ORIGIN/../build' $(LDFLAGS) -lpthread -latomic
	$(CC) $(CFLAGS) $(ENGINE_INC) -o $(BIN_DIR)/autocomplete_cli $(SRC_DIR)/lg_engine/src/demo/autocomplete_cli.c $(BUILD_DIR)/$(LIBGEMINI_CORE) -L$(BUILD_DIR) -lluganda_tok -Wl,-rpath,'$$ORIGIN/../build' $(LDFLAGS) -lpthread -latomic
	$(CC) $(CFLAGS) $(ENGINE_INC) -o $(BIN_DIR)/test_lineage $(SRC_DIR)/lg_engine/src/demo/test_lineage.c $(BUILD_DIR)/$(LIBGEMINI_CORE) -L$(BUILD_DIR) -lluganda_tok -Wl,-rpath,'$$ORIGIN/../build' $(LDFLAGS) -lpthread -latomic
	$(CC) $(CFLAGS) $(ENGINE_INC) -o $(BIN_DIR)/benchmark_latency $(SRC_DIR)/lg_engine/src/demo/benchmark_latency.c $(BUILD_DIR)/$(LIBGEMINI_CORE) -L$(BUILD_DIR) -lluganda_tok -Wl,-rpath,'$$ORIGIN/../build' $(LDFLAGS) -lpthread -latomic
	$(CC) $(CFLAGS) $(ENGINE_INC) -o $(BIN_DIR)/compare_snapshots $(SRC_DIR)/lg_engine/src/demo/compare_snapshots.c $(BUILD_DIR)/$(LIBGEMINI_CORE) -L$(BUILD_DIR) -lluganda_tok -Wl,-rpath,'$$ORIGIN/../build' $(LDFLAGS) -lpthread -latomic

# Create build directories
dirs:
	@mkdir -p $(BUILD_DIR) $(BIN_DIR)

# Shared library for Python/FFI integration
lib: dirs libluganda_tok.so libgemini.so

libluganda_tok.so: dirs
	$(CC) $(CFLAGS) -shared -fPIC $(INCLUDES) -o $(BUILD_DIR)/libluganda_tok.so $(MMAP_SOURCES) $(SRC_DIR)/tokenizer_api.c $(LDFLAGS)

libgemini.so: dirs libluganda_tok.so $(LIBGEMINI_CORE)
	$(CC) $(CFLAGS) -shared -fPIC $(ENGINE_INC) -o $(BUILD_DIR)/libgemini.so $(BUILD_DIR)/$(LIBGEMINI_CORE) -L$(BUILD_DIR) -lluganda_tok -Wl,-rpath,'$$ORIGIN' $(LDFLAGS) -lpthread -latomic

# Main tokenizer demo executable
tokenizer_demo: dirs $(MAIN_OBJS)
	$(CC) $(CFLAGS) $(INCLUDES) -o $(BIN_DIR)/$@ $(MAIN_OBJS) $(LDFLAGS)

# Benchmark executable (root level)
bench: dirs $(BENCH_OBJS)
	$(CC) $(CFLAGS) $(INCLUDES) -o $(BIN_DIR)/$@ $(BENCH_OBJS) $(LDFLAGS)

# Luganda-specific benchmark
benchmark_luganda: dirs $(BUILD_DIR)/benchmark_luganda.o $(CORE_OBJS)
	$(CC) $(CFLAGS) $(INCLUDES) -o $(BIN_DIR)/$@ $(BUILD_DIR)/benchmark_luganda.o $(patsubst $(SRC_DIR)/%.c,$(BUILD_DIR)/%.o,$(CORE_SOURCES)) $(LDFLAGS)

# MMAP version
tokenizer_mmap: dirs $(MMAP_OBJS)
	$(CC) $(CFLAGS) $(INCLUDES) -o $(BIN_DIR)/$@ $(MMAP_OBJS) $(LDFLAGS)

# Compile source files to build (release)
$(BUILD_DIR)/%.o: %.c
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) $(INCLUDES) -c $< -o $@

$(BUILD_DIR)/%.o: $(SRC_DIR)/%.c
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) $(INCLUDES) -c $< -o $@

# Test binaries - compile test sources directly
$(TEST_DIR)/test_simple_tokenizer: $(TEST_DIR)/test_simple_tokenizer.c $(CORE_SOURCES)
	$(CC) $(CFLAGS) $(INCLUDES) -o $@ $^ $(LDFLAGS)

$(TEST_DIR)/test_benchmark_simple: $(TEST_DIR)/test_benchmark_simple.c $(CORE_SOURCES)
	$(CC) $(CFLAGS) $(INCLUDES) -o $@ $^ $(LDFLAGS)

$(TEST_DIR)/test_corruption_minimal: $(TEST_DIR)/test_corruption_minimal.c $(CORE_SOURCES)
	$(CC) $(CFLAGS) $(INCLUDES) -o $@ $^ $(LDFLAGS)

$(TEST_DIR)/test_edge_cases_debug: $(TEST_DIR)/test_edge_cases.c $(CORE_SOURCES)
	$(CC) $(CFLAGS) $(INCLUDES) -o $@ $^ $(LDFLAGS)

$(TEST_DIR)/test_fixes: $(TEST_DIR)/test_fixes.c $(CORE_SOURCES)
	$(CC) $(CFLAGS) $(INCLUDES) -o $@ $^ $(LDFLAGS)

$(TEST_DIR)/test_serialization_security: $(TEST_DIR)/test_serialization_security.c $(CORE_SOURCES)
	$(CC) $(CFLAGS) $(INCLUDES) -o $@ $^ $(LDFLAGS)

$(TEST_DIR)/test_grapheme: $(TEST_DIR)/test_grapheme.c $(CORE_SOURCES)
	$(CC) $(CFLAGS) $(INCLUDES) -o $@ $^ $(LDFLAGS)

$(TEST_DIR)/test_utf8: $(TEST_DIR)/test_utf8.c $(CORE_SOURCES)
	$(CC) $(CFLAGS) $(INCLUDES) -o $@ $^ $(LDFLAGS)

$(TEST_DIR)/test_bugs: $(TEST_DIR)/test_bugs.c $(CORE_SOURCES)
	$(CC) $(CFLAGS) $(INCLUDES) -o $@ $^ $(LDFLAGS)

$(TEST_DIR)/test_determinism: $(TEST_DIR)/test_determinism.c $(CORE_SOURCES)
	$(CC) $(CFLAGS) $(INCLUDES) -o $@ $^ $(LDFLAGS)

$(TEST_DIR)/test_edge_cases_simple: $(TEST_DIR)/test_edge_cases_simple.c $(CORE_SOURCES)
	$(CC) $(CFLAGS) $(INCLUDES) -o $@ $^ $(LDFLAGS)

$(TEST_DIR)/test_truth_layer: $(TEST_DIR)/test_truth_layer.c $(CORE_SOURCES)
	$(CC) $(CFLAGS) $(INCLUDES) -o $@ $^ $(LDFLAGS)

$(TEST_DIR)/test_lexical_domain: $(TEST_DIR)/test_lexical_domain.c $(CORE_SOURCES)
	$(CC) $(CFLAGS) $(INCLUDES) -o $@ $^ $(LDFLAGS)

$(TEST_DIR)/test_production_stress: $(TEST_DIR)/test_production_stress.c $(CORE_SOURCES)
	$(CC) $(CFLAGS) $(INCLUDES) -o $@ $^ $(LDFLAGS)

$(TEST_DIR)/test_learning_loop: $(TEST_DIR)/test_learning_loop.c $(LIBGEMINI_CORE) libluganda_tok.so
	$(CC) $(CFLAGS) $(ENGINE_INC) -o $@ $(TEST_DIR)/test_learning_loop.c $(BUILD_DIR)/$(LIBGEMINI_CORE) -L$(BUILD_DIR) -lluganda_tok -Wl,-rpath,'$$ORIGIN' $(LDFLAGS) -lpthread -latomic

$(TEST_DIR)/test_memory_audit: $(TEST_DIR)/test_memory_audit.c $(LIBGEMINI_CORE) libluganda_tok.so
	$(CC) $(CFLAGS) $(ENGINE_INC) -o $@ $(TEST_DIR)/test_memory_audit.c $(BUILD_DIR)/$(LIBGEMINI_CORE) -L$(BUILD_DIR) -lluganda_tok -Wl,-rpath,'$$ORIGIN' $(LDFLAGS) -lpthread -latomic

# Build all test binaries
test_bins: lib $(TEST_BINARIES)

# Debug build
debug: dirs
	$(CC) $(CFLAGS_DEBUG) $(INCLUDES) -o $(BIN_DIR)/tokenizer_debug $(MAIN_SOURCES) $(LDFLAGS)

# Run tests
test: all
	@echo "Running test suite..."
	@bash run_tests.sh

# Run benchmarks
run-bench: bench
	@echo "Running benchmarks..."
	@bash run_benchmarks.sh

# Clean build artifacts
clean:
	rm -rf $(BUILD_DIR) $(BIN_DIR)
	rm -f $(TEST_BINARIES)
	rm -rf test_results benchmark_results

# Install (optional - customize as needed)
install: all
	@echo "Installing executables to /usr/local/bin/..."
	@cp $(BIN_DIR)/* /usr/local/bin/ 2>/dev/null || echo "Install failed: run with sudo"
	@echo "Installing libraries to /usr/local/lib/..."
	@cp $(BUILD_DIR)/*.so /usr/local/lib/ 2>/dev/null || echo "Install failed: run with sudo"
	@echo "Installation complete"

# Include dependency files
-include $(DEPS)

# Help target
help:
	@echo "Luganda Tokenizer Build System"
	@echo "=============================="
	@echo ""
	@echo "Targets:"
	@echo "  make all          - Build all main targets (default)"
	@echo "  make tokenizer_demo - Build the main tokenizer demo"
	@echo "  make bench        - Build benchmark executable"
	@echo "  make benchmark_luganda - Build Luganda-specific benchmark"
	@echo "  make test_bins    - Build all test binaries"
	@echo "  make debug        - Build debug version"
	@echo "  make test         - Build and run all tests"
	@echo "  make run-bench    - Build and run benchmarks"
	@echo "  make clean        - Remove all build artifacts"
	@echo "  make asan         - Build with AddressSanitizer and UndefinedBehaviorSanitizer"
	@echo "  make help         - Show this help message"
	@echo ""
	@echo "Directories:"
	@echo "  src/              - Source files"
	@echo "  include/          - Header files"
	@echo "  test/             - Test files"
	@echo "  build/            - Build objects"

# Build with AddressSanitizer and UndefinedBehaviorSanitizer
asan:
	@echo "🛡️ Building with AddressSanitizer and UndefinedBehaviorSanitizer..."
	@$(MAKE) clean
	@$(MAKE) test_bins \
		CFLAGS="-Wall -Wextra -g3 -O1 -fsanitize=address,undefined -fno-omit-frame-pointer -D_GNU_SOURCE -DDEBUG" \
		LDFLAGS="-fsanitize=address,undefined -lm"
