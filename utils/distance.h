#pragma once

#include "kmeans_common.h"

// Euclidean distance between two feature vectors of length NUM_FEATURES.
double euclidean_distance(const std::array<double, NUM_FEATURES> &a,
                          const std::array<double, NUM_FEATURES> &b);

