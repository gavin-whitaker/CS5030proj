CXX      := g++
CXXFLAGS := -std=c++17 -O2 -Wall
OMPFLAGS := -fopenmp
NVCC     := nvcc
NVCCFLAGS:= -std=c++17 -O2
MPICC    := mpicxx
MPICXXFLAGS := -std=c++17 -O2

UTILS_SRC := utils/io.cpp utils/distance.cpp
UTILS_OBJ := $(UTILS_SRC:.cpp=.o)

.PHONY: all serial openmp cuda mpi mpi_cuda clean

all: serial openmp

# ── Serial ────────────────────────────────────────────────────────────────────
serial: serial/kmeans_serial $(UTILS_OBJ)

serial/kmeans_serial: serial/main.cpp serial/kmeans_serial.cpp $(UTILS_OBJ)
	$(CXX) $(CXXFLAGS) -o $@ $^

# ── OpenMP ────────────────────────────────────────────────────────────────────
openmp: openmp/kmeans_openmp $(UTILS_OBJ)

openmp/kmeans_openmp: openmp/main.cpp openmp/kmeans_openmp.cpp $(UTILS_OBJ)
	$(CXX) $(CXXFLAGS) $(OMPFLAGS) -o $@ $^

# ── CUDA ──────────────────────────────────────────────────────────────────────
cuda: cuda/kmeans_cuda

cuda/kmeans_cuda: cuda/main.cu cuda/kmeans_cuda.cu utils/io.cpp utils/distance.cpp
	$(NVCC) $(NVCCFLAGS) -o $@ $^

# ── MPI ───────────────────────────────────────────────────────────────────────
mpi: mpi/kmeans_mpi

mpi/kmeans_mpi: mpi/main.cpp mpi/kmeans_mpi.cpp utils/io.cpp utils/distance.cpp
	$(MPICC) $(MPICXXFLAGS) -o $@ $^

# ── MPI + CUDA ────────────────────────────────────────────────────────────────
mpi_cuda: mpi_cuda/kmeans_mpi_cuda

mpi_cuda/kmeans_mpi_cuda: mpi_cuda/main.cu mpi_cuda/kmeans_mpi_cuda.cu \
                           utils/io.cpp utils/distance.cpp
	$(NVCC) $(NVCCFLAGS) -Xcompiler -fopenmp \
	    $(shell mpicxx --showme:compile) \
	    $(shell mpicxx --showme:link) \
	    -o $@ $^

# ── Shared object compilation ─────────────────────────────────────────────────
%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c -o $@ $<

# ── Clean ─────────────────────────────────────────────────────────────────────
clean:
	rm -f utils/*.o \
	      serial/kmeans_serial \
	      openmp/kmeans_openmp \
	      cuda/kmeans_cuda \
	      mpi/kmeans_mpi \
	      mpi_cuda/kmeans_mpi_cuda
