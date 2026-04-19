# Skeleton build system for parallel K-Means implementations.

RESULTS_DIR := results
BUILD_DIR := build

CXX ?= c++
CPPFLAGS := -I.
OPT ?= -O2
CXXFLAGS := -std=c++17 $(OPT) -g -Wall -Wextra $(CPPFLAGS)
OPENMP_CXXFLAGS := $(CXXFLAGS) -fopenmp

UTILS_CPP := utils/io.cpp utils/distance.cpp utils/validate.cpp \
             utils/kmeans_utils.cpp utils/args.cpp

SERIAL_SOURCES := serial/main.cpp serial/kmeans_serial.cpp $(UTILS_CPP)
OPENMP_SOURCES := openmp/main.cpp openmp/kmeans_openmp.cpp $(UTILS_CPP)
MPI_SOURCES := mpi/main.cpp mpi/kmeans_mpi.cpp $(UTILS_CPP)
CUDA_SOURCES := cuda/main.cu cuda/kmeans_cuda.cu $(UTILS_CPP)
MPI_CUDA_SOURCES := mpi_cuda/main.cu mpi_cuda/kmeans_mpi_cuda.cu $(UTILS_CPP)

SERIAL_OBJS := $(patsubst %.cpp,$(BUILD_DIR)/%.o,$(filter %.cpp,$(SERIAL_SOURCES)))
OPENMP_OBJS := $(patsubst %.cpp,$(BUILD_DIR)/%.o,$(filter %.cpp,$(OPENMP_SOURCES)))
MPI_OBJS := $(patsubst %.cpp,$(BUILD_DIR)/%.o,$(filter %.cpp,$(MPI_SOURCES)))

MPICXX := $(shell command -v mpicxx 2>/dev/null)
ifeq ($(strip $(MPICXX)),)
  MPICXX := $(CXX)
endif

NVCC := $(shell command -v nvcc 2>/dev/null)
ifeq ($(strip $(NVCC)),)
  CUDA_COMPILER := $(CXX)
  CUDA_CXXFLAGS := -std=c++17 $(OPT) -g -x c++ $(CPPFLAGS)
else
  CUDA_COMPILER := $(NVCC)
  CUDA_CXXFLAGS := -std=c++17 $(OPT) -g $(CPPFLAGS)
endif

MPI_INCS := -I/opt/mpich/include
MPI_LIBS := -L/opt/mpich/lib -lmpicxx -lmpi -Xlinker -rpath -Xlinker /opt/mpich/lib

CUDA_OBJS := $(patsubst %.cu,$(BUILD_DIR)/%.o,$(filter %.cu,$(CUDA_SOURCES))) \
             $(patsubst %.cpp,$(BUILD_DIR)/%.o,$(filter %.cpp,$(CUDA_SOURCES)))
MPI_CUDA_OBJS := $(patsubst %.cu,$(BUILD_DIR)/%.o,$(filter %.cu,$(MPI_CUDA_SOURCES))) \
                 $(patsubst %.cpp,$(BUILD_DIR)/%.o,$(filter %.cpp,$(MPI_CUDA_SOURCES)))

.PHONY: serial openmp cuda mpi mpi_cuda clean all run-all perf visualize

all: serial openmp cuda mpi mpi_cuda

run-all: serial openmp cuda
	@mkdir -p $(RESULTS_DIR)
	@echo "Running serial..."
	./$(RESULTS_DIR)/serial --input tests/fixtures/small_100.csv --output $(RESULTS_DIR)/serial_out.csv --k 3 --max_iter 50 --threshold 0.001
	@echo "Running openmp..."
	./$(RESULTS_DIR)/openmp --input tests/fixtures/small_100.csv --output $(RESULTS_DIR)/openmp_out.csv --k 3 --max_iter 50 --threshold 0.001 --threads 4
	@echo "Running cuda..."
	./$(RESULTS_DIR)/cuda --input tests/fixtures/small_100.csv --output $(RESULTS_DIR)/cuda_out.csv --k 3 --max_iter 50 --threshold 0.001 --block_size 256
	@echo "✓ Results written to $(RESULTS_DIR)/*.csv"

perf: serial openmp cuda
	bash run_perf.sh

$(RESULTS_DIR):
	mkdir -p $(RESULTS_DIR)

serial: $(RESULTS_DIR)/serial
openmp: $(RESULTS_DIR)/openmp
cuda: $(RESULTS_DIR)/cuda
mpi: $(RESULTS_DIR)/mpi
mpi_cuda: $(RESULTS_DIR)/mpi_cuda

$(RESULTS_DIR)/serial: $(SERIAL_OBJS) | $(RESULTS_DIR)
	$(CXX) $(CXXFLAGS) -o $@ $^

$(RESULTS_DIR)/openmp: $(OPENMP_OBJS) | $(RESULTS_DIR)
	$(CXX) $(OPENMP_CXXFLAGS) -o $@ $^

$(RESULTS_DIR)/mpi: $(MPI_OBJS) | $(RESULTS_DIR)
	$(MPICXX) $(CXXFLAGS) -o $@ $^

$(RESULTS_DIR)/cuda: $(CUDA_OBJS) | $(RESULTS_DIR)
	$(CUDA_COMPILER) $(CUDA_CXXFLAGS) -o $@ $^

$(RESULTS_DIR)/mpi_cuda: $(MPI_CUDA_OBJS) | $(RESULTS_DIR)
	$(CUDA_COMPILER) $(CUDA_CXXFLAGS) -o $@ $^ $(MPI_LIBS)

$(BUILD_DIR)/%.o: %.cpp
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(BUILD_DIR)/mpi/%.o: mpi/%.cpp
	@mkdir -p $(dir $@)
	$(MPICXX) $(CXXFLAGS) -c $< -o $@

$(BUILD_DIR)/openmp/%.o: openmp/%.cpp
	@mkdir -p $(dir $@)
	$(CXX) $(OPENMP_CXXFLAGS) -c $< -o $@

$(BUILD_DIR)/%.o: %.cu
	@mkdir -p $(dir $@)
	$(CUDA_COMPILER) $(CUDA_CXXFLAGS) -c $< -o $@

$(BUILD_DIR)/mpi_cuda/%.o: mpi_cuda/%.cu
	@mkdir -p $(dir $@)
	$(CUDA_COMPILER) $(CUDA_CXXFLAGS) $(MPI_INCS) -c $< -o $@

# ===== Testing =====
TEST_DIR := tests
TEST_UTILS := utils/io.cpp utils/distance.cpp utils/validate.cpp
UNIT_TESTS := test_distance test_io test_validate test_kmeans_serial

.PHONY: test test_unit test_integration clean_tests

test: test_unit test_integration
	@echo "All tests passed!"

test_unit: $(addprefix $(BUILD_DIR)/$(TEST_DIR)/,$(UNIT_TESTS))
	@mkdir -p $(RESULTS_DIR)
	@echo "Running unit tests..."
	@for test in $^; do \
	  echo "  $$test"; \
	  $$test || exit 1; \
	done
	@echo "✓ All unit tests passed"

test_integration: serial openmp cuda
	@mkdir -p $(RESULTS_DIR)
	@echo "Running integration tests..."
	python3 -m pytest $(TEST_DIR)/integration/ -v --tb=short

$(BUILD_DIR)/$(TEST_DIR)/test_distance: $(TEST_DIR)/test_distance.cpp utils/distance.cpp
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) -I. -o $@ $^

$(BUILD_DIR)/$(TEST_DIR)/test_io: $(TEST_DIR)/test_io.cpp $(TEST_UTILS)
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) -I. -o $@ $^

$(BUILD_DIR)/$(TEST_DIR)/test_validate: $(TEST_DIR)/test_validate.cpp $(TEST_UTILS)
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) -I. -o $@ $^

$(BUILD_DIR)/$(TEST_DIR)/test_kmeans_serial: $(TEST_DIR)/test_kmeans_serial.cpp \
    serial/kmeans_serial.cpp utils/kmeans_utils.cpp $(TEST_UTILS)
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) -I. -o $@ $^

$(BUILD_DIR)/$(TEST_DIR)/test_kmeans_utils: $(TEST_DIR)/test_kmeans_utils.cpp \
    serial/kmeans_serial.cpp $(TEST_UTILS)
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) -I. -o $@ $^

$(BUILD_DIR)/$(TEST_DIR)/test_args: $(TEST_DIR)/test_args.cpp \
    serial/main.cpp serial/kmeans_serial.cpp $(TEST_UTILS)
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) -I. -o $@ $^

clean_tests:
	rm -rf $(BUILD_DIR)/$(TEST_DIR) $(RESULTS_DIR)/*_out.csv

clean: clean_tests
	rm -rf $(BUILD_DIR) $(RESULTS_DIR)/serial $(RESULTS_DIR)/openmp $(RESULTS_DIR)/cuda $(RESULTS_DIR)/mpi $(RESULTS_DIR)/mpi_cuda

# ===== Visualization =====
VENV_DIR := .venv

$(VENV_DIR):
	python3 -m venv $(VENV_DIR)
	$(VENV_DIR)/bin/pip install -q -r requirements.txt

visualize: serial $(VENV_DIR)
	@mkdir -p $(RESULTS_DIR)
	./$(RESULTS_DIR)/serial --input data/tracks_features.csv --output $(RESULTS_DIR)/spotify_viz.csv --k 10 --max_iter 50 --threshold 0.001
	$(VENV_DIR)/bin/python3 scripts/visualize.py \
	  --input $(RESULTS_DIR)/spotify_viz.csv \
	  --k 10 \
	  --features valence danceability energy \
	  --output $(RESULTS_DIR)/cluster_viz.png
	@echo "✓ Visualization saved to $(RESULTS_DIR)/cluster_viz.png"

