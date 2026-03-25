#include "kmeans_mpi_cuda.cuh"

#include <iostream>

// Kernel stub: no logic yet.
__global__ void kmeans_assignment_kernel_stub_mpi_cuda() {
  // TODO: GPU assignment kernel for MPI+CUDA implementation.
}

int run_kmeans_mpi_cuda(const Config &cfg) {
  (void)cfg;
  std::cout << "TODO: implement MPI+CUDA distributed-memory K-Means (GPU stub).\n";
  return 0;
}

