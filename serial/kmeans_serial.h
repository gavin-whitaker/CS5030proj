#ifndef KMEANS_SERIAL_H
#define KMEANS_SERIAL_H

// Serial K-Means clustering interface

#include "../utils/kmeans_common.h"
#include <vector>

// Run serial K-Means on the given points; returns final centroids
std::vector<Point> kmeans_serial(std::vector<Point> &points,
                                 const KMeansParams &params);

#endif // KMEANS_SERIAL_H
