#ifndef KMEANS_OPENMP_H
#define KMEANS_OPENMP_H

// OpenMP-parallel K-Means clustering interface

#include "../utils/kmeans_common.h"
#include <vector>

// Run OpenMP-parallel K-Means on the given points; returns final centroids
std::vector<Point> kmeans_openmp(std::vector<Point> &points,
                                 const KMeansParams &params,
                                 int num_threads);

#endif // KMEANS_OPENMP_H
