#ifndef KMEANS_COMMON_H
#define KMEANS_COMMON_H

// Common types and constants shared across all K-Means implementations

#include <vector>

// A single data point with D dimensions
struct Point {
    std::vector<double> coords;
    int cluster; // assigned cluster index
};

// Parameters for K-Means
struct KMeansParams {
    int k;          // number of clusters
    int max_iter;   // maximum iterations
    double tol;     // convergence tolerance
    int dims;       // number of dimensions
};

#endif // KMEANS_COMMON_H
