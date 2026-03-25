#pragma once

#include "utils/kmeans_common.h"

#ifndef __CUDACC__
#ifndef __global__
#define __global__
#endif
#endif

// MPI + CUDA distributed-memory K-Means entry point (skeleton only).
int run_kmeans_mpi_cuda(const Config &cfg);

// Kernel stub for the GPU assignment step.
__global__ void kmeans_assignment_kernel_stub_mpi_cuda();

