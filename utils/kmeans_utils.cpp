#include "utils/kmeans_utils.h"

#include "utils/distance.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <random>
#include <vector>

std::vector<Centroid> init_centroids_pp(const std::vector<Point> &points,
                                         int k,
                                         std::mt19937 &rng) {
  std::vector<Centroid> centroids;
  centroids.reserve(k);

  // 1. Pick the first centroid uniformly at random.
  std::uniform_int_distribution<size_t> uniform(0, points.size() - 1);
  Centroid first;
  first.features = points[uniform(rng)].features;
  centroids.push_back(first);

  std::vector<double> dist_sq(points.size(), std::numeric_limits<double>::max());

  for (int c = 1; c < k; ++c) {
    // Update min squared distances from each point to the nearest centroid.
    for (size_t i = 0; i < points.size(); ++i) {
      double d = euclidean_distance(points[i].features,
                                    centroids.back().features);
      double d2 = d * d;
      if (d2 < dist_sq[i]) dist_sq[i] = d2;
    }

    // Sample next centroid with probability proportional to dist_sq.
    std::discrete_distribution<size_t> weighted(dist_sq.begin(), dist_sq.end());
    Centroid next;
    next.features = points[weighted(rng)].features;
    centroids.push_back(next);
  }

  return centroids;
}

std::vector<Centroid> update_centroids_cpu(const std::vector<Point> &points,
                                           const std::vector<int> &labels,
                                           int k) {
  std::vector<Centroid> centroids(k);
  std::vector<int> counts(k, 0);

  for (size_t i = 0; i < points.size(); ++i) {
    int c = labels[i];
    for (int f = 0; f < NUM_FEATURES; ++f) {
      centroids[c].features[f] += points[i].features[f];
    }
    ++counts[c];
  }

  for (int c = 0; c < k; ++c) {
    if (counts[c] > 0) {
      for (int f = 0; f < NUM_FEATURES; ++f) {
        centroids[c].features[f] /= counts[c];
      }
    }
  }
  return centroids;
}

bool check_convergence(const std::vector<Centroid> &old_c,
                       const std::vector<Centroid> &new_c,
                       double threshold) {
  for (size_t c = 0; c < old_c.size(); ++c) {
    if (euclidean_distance(old_c[c].features, new_c[c].features) >= threshold) {
      return false;
    }
  }
  return true;
}
