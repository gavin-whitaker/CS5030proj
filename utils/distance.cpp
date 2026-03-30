#include "distance.h"

#include <cmath>

double euclidean_distance(const std::array<double, NUM_FEATURES> &a,
                          const std::array<double, NUM_FEATURES> &b) {
  double sum = 0.0;
  for (int i = 0; i < NUM_FEATURES; ++i) {
    double diff = a[i] - b[i];
    sum += diff * diff;
  }
  return std::sqrt(sum);
}

