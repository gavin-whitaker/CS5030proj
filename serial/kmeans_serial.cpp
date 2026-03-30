#include "kmeans_serial.h"

#include "utils/distance.h"
#include "utils/io.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <vector>

// ---------------------------------------------------------------------------
// K-Means++ initialization
// Picks centroids with probability proportional to squared distance from the
// nearest already-chosen centroid, giving better spread than pure random.
// ---------------------------------------------------------------------------
static std::vector<Centroid> init_centroids_pp(const std::vector<Point> &points,
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

// ---------------------------------------------------------------------------
// Assign each point to the nearest centroid; return label vector.
// ---------------------------------------------------------------------------
static std::vector<int> assign_clusters(const std::vector<Point> &points,
                                        const std::vector<Centroid> &centroids) {
  std::vector<int> labels(points.size());
  int k = static_cast<int>(centroids.size());

  for (size_t i = 0; i < points.size(); ++i) {
    double best_dist = std::numeric_limits<double>::max();
    int best_c = 0;
    for (int c = 0; c < k; ++c) {
      double d = euclidean_distance(points[i].features, centroids[c].features);
      if (d < best_dist) {
        best_dist = d;
        best_c = c;
      }
    }
    labels[i] = best_c;
  }
  return labels;
}

// ---------------------------------------------------------------------------
// Recompute centroid positions as the mean of their assigned points.
// Centroids with no assigned points are left unchanged (rare with KMeans++).
// ---------------------------------------------------------------------------
static std::vector<Centroid> update_centroids(const std::vector<Point> &points,
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

// ---------------------------------------------------------------------------
// Convergence check: true if every centroid moved less than threshold.
// ---------------------------------------------------------------------------
static bool check_convergence(const std::vector<Centroid> &old_c,
                               const std::vector<Centroid> &new_c,
                               double threshold) {
  for (size_t c = 0; c < old_c.size(); ++c) {
    if (euclidean_distance(old_c[c].features, new_c[c].features) >= threshold) {
      return false;
    }
  }
  return true;
}

// ---------------------------------------------------------------------------
// Entry point called from serial/main.cpp
// ---------------------------------------------------------------------------
int run_kmeans_serial(const Config &cfg) {
  auto wall_start = std::chrono::steady_clock::now();

  // --- Load data ---
  std::vector<Point> points = load_data(cfg.input);
  if (points.empty()) {
    std::cerr << "Error: no data loaded. Aborting.\n";
    return 1;
  }

  // --- Initialize centroids (K-Means++) ---
  std::mt19937 rng(42); // fixed seed for reproducibility
  std::vector<Centroid> centroids = init_centroids_pp(points, cfg.k, rng);

  std::cout << "K=" << cfg.k
            << "  max_iter=" << cfg.max_iter
            << "  threshold=" << cfg.threshold
            << "  points=" << points.size() << "\n";

  // --- Main K-Means loop ---
  std::vector<int> labels(points.size(), 0);
  int iter = 0;
  bool converged = false;

  for (; iter < cfg.max_iter; ++iter) {
    std::vector<int> new_labels = assign_clusters(points, centroids);
    std::vector<Centroid> new_centroids = update_centroids(points, new_labels, cfg.k);

    converged = check_convergence(centroids, new_centroids, cfg.threshold);
    labels = std::move(new_labels);
    centroids = std::move(new_centroids);

    if ((iter + 1) % 10 == 0) {
      std::cout << "  iter " << (iter + 1) << " done\n";
    }

    if (converged) {
      ++iter; // report the iteration that converged
      break;
    }
  }

  auto wall_end = std::chrono::steady_clock::now();
  double elapsed =
      std::chrono::duration<double>(wall_end - wall_start).count();

  std::cout << (converged ? "Converged" : "Reached max_iter")
            << " after " << iter << " iterations.\n";
  std::cout << "Elapsed time: " << elapsed << " s\n";

  // --- Write results ---
  write_output_csv(cfg.output, labels, points);
  return 0;
}

