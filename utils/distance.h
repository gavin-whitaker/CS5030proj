#ifndef DISTANCE_H
#define DISTANCE_H

// Distance functions used by K-Means implementations

#include "kmeans_common.h"

// Euclidean distance squared between two points
double euclidean_distance_sq(const Point &a, const Point &b);

// Find the index of the nearest centroid for a given point
int nearest_centroid(const Point &p, const std::vector<Point> &centroids);

#endif // DISTANCE_H
