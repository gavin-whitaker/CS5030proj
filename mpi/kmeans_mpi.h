#ifndef KMEANS_MPI_H
#define KMEANS_MPI_H

// MPI-parallel K-Means clustering interface

#include "../utils/kmeans_common.h"
#include <vector>

// Run MPI-parallel K-Means; each rank operates on its local partition.
// Returns final centroids on rank 0.
std::vector<Point> kmeans_mpi(std::vector<Point> &local_points,
                              const KMeansParams &params);

#endif // KMEANS_MPI_H
