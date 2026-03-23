// kmeans_openmp.cpp — OpenMP-parallel K-Means clustering implementation
// TODO: implement K-Means with OpenMP parallelism

#include "kmeans_openmp.h"
#include "../utils/distance.h"
#include <omp.h>

std::vector<Point> kmeans_openmp(std::vector<Point> &points,
                                 const KMeansParams &params,
                                 int num_threads) {
    // TODO: set OMP_NUM_THREADS, parallelize assign and update steps
    std::vector<Point> centroids;
    return centroids;
}
