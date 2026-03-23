#ifndef KMEANS_MPI_CUDA_CUH
#define KMEANS_MPI_CUDA_CUH

// MPI + CUDA hybrid K-Means clustering interface

#include "../utils/kmeans_common.h"
#include <vector>

// Run MPI+CUDA hybrid K-Means; each MPI rank uses a GPU for local computation.
// Returns final centroids on rank 0.
std::vector<Point> kmeans_mpi_cuda(std::vector<Point> &local_points,
                                   const KMeansParams &params);

#endif // KMEANS_MPI_CUDA_CUH
