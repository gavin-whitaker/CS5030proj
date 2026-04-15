# Skeleton build system for parallel K-Means implementations.

RESULTS_DIR := results
BUILD_DIR := build

CXX ?= c++
CPPFLAGS := -I.
OPT ?= -O2
CXXFLAGS := -std=c++17 $(OPT) -g -Wall -Wextra $(CPPFLAGS)
OPENMP_CXXFLAGS := $(CXXFLAGS) -fopenmp

UTILS_CPP := utils/io.cpp utils/distance.cpp utils/validate.cpp

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

CUDA_OBJS := $(patsubst %.cu,$(BUILD_DIR)/%.o,$(filter %.cu,$(CUDA_SOURCES))) \
             $(patsubst %.cpp,$(BUILD_DIR)/%.o,$(filter %.cpp,$(CUDA_SOURCES)))
MPI_CUDA_OBJS := $(patsubst %.cu,$(BUILD_DIR)/%.o,$(filter %.cu,$(MPI_CUDA_SOURCES))) \
                 $(patsubst %.cpp,$(BUILD_DIR)/%.o,$(filter %.cpp,$(MPI_CUDA_SOURCES)))

.PHONY: serial openmp cuda mpi mpi_cuda clean all

all: serial openmp cuda mpi mpi_cuda

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
	$(CUDA_COMPILER) $(CUDA_CXXFLAGS) -o $@ $^

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

clean:
	rm -rf $(BUILD_DIR) $(RESULTS_DIR)/serial $(RESULTS_DIR)/openmp $(RESULTS_DIR)/cuda $(RESULTS_DIR)/mpi $(RESULTS_DIR)/mpi_cuda

