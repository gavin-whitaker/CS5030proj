#pragma once

#include "utils/kmeans_common.h"

// When compiling with a non-CUDA compiler (e.g., for skeleton build sanity),
// ensure CUDA qualifiers exist.
#ifndef __CUDACC__
#ifndef __global__
#define __global__
#endif
#endif

// CUDA K-Means entry point (skeleton only).
int run_kmeans_cuda(const Config &cfg);

// Kernel stub for the assignment step.
__global__ void kmeans_assignment_kernel_stub();
