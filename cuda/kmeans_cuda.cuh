#ifndef KMEANS_CUDA_CUH
#define KMEANS_CUDA_CUH

// CUDA K-Means clustering interface

#include "../utils/kmeans_common.h"
#include <vector>

// Run CUDA-accelerated K-Means on the given points; returns final centroids
std::vector<Point> kmeans_cuda(std::vector<Point> &points,
                               const KMeansParams &params);

#endif // KMEANS_CUDA_CUH
