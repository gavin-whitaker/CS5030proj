#include "kmeans_openmp.h"

#include "utils/distance.h"
#include "utils/io.h"
#include "utils/kmeans_utils.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <iostream>
#include <limits>
#include <numeric>
#include <omp.h>
#include <random>
#include <vector>

// ---------------------------------------------------------------------------
// Parallel assignment: each point assigned to nearest centroid
// ---------------------------------------------------------------------------
static std::vector<int> assign_clusters(const std::vector<Point> &points,
                                        const std::vector<Centroid> &centroids) {
  std::vector<int> labels(points.size());
  int k = static_cast<int>(centroids.size());

#pragma omp parallel for schedule(static)
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

// Intentionally differs from utils/kmeans_utils.cpp update_centroids_cpu.
// Uses thread-private accumulators to avoid false sharing.
static std::vector<Centroid> update_centroids(const std::vector<Point> &points,
                                              const std::vector<int> &labels,
                                              int k) {
  int nthreads = omp_get_max_threads();

  // thread-local sums: [thread][centroid][feature]
  std::vector<std::vector<std::array<double, NUM_FEATURES>>> sums(
      nthreads, std::vector<std::array<double, NUM_FEATURES>>(k));
  std::vector<std::vector<int>> counts(nthreads, std::vector<int>(k, 0));

  // Initialize thread-local arrays
  for (int t = 0; t < nthreads; ++t) {
    for (int c = 0; c < k; ++c) {
      for (int f = 0; f < NUM_FEATURES; ++f) {
        sums[t][c][f] = 0.0;
      }
    }
  }

  // Parallel accumulation
#pragma omp parallel
  {
    int tid = omp_get_thread_num();
#pragma omp for schedule(static)
    for (size_t i = 0; i < points.size(); ++i) {
      int c = labels[i];
      for (int f = 0; f < NUM_FEATURES; ++f) {
        sums[tid][c][f] += points[i].features[f];
      }
      ++counts[tid][c];
    }
  }

  // Serial merge
  std::vector<Centroid> centroids(k);
  std::vector<int> total_counts(k, 0);

  for (int t = 0; t < nthreads; ++t) {
    for (int c = 0; c < k; ++c) {
      total_counts[c] += counts[t][c];
      for (int f = 0; f < NUM_FEATURES; ++f) {
        centroids[c].features[f] += sums[t][c][f];
      }
    }
  }

  for (int c = 0; c < k; ++c) {
    if (total_counts[c] > 0) {
      for (int f = 0; f < NUM_FEATURES; ++f) {
        centroids[c].features[f] /= total_counts[c];
      }
    }
  }

  return centroids;
}


// ---------------------------------------------------------------------------
// Main OpenMP K-Means
// ---------------------------------------------------------------------------
int run_kmeans_openmp(const Config &cfg) {
  auto wall_start = std::chrono::steady_clock::now();

  // Set thread count
  omp_set_num_threads(cfg.threads);

  // Load data
  std::vector<Point> points = load_data(cfg.input);
  if (points.empty()) {
    std::cerr << "Error: no data loaded. Aborting.\n";
    return 1;
  }

  // Initialize centroids (K-Means++)
  std::mt19937 rng(42);
  std::vector<Centroid> centroids = init_centroids_pp(points, cfg.k, rng);

  std::cout << "K=" << cfg.k
            << "  max_iter=" << cfg.max_iter
            << "  threshold=" << cfg.threshold
            << "  points=" << points.size()
            << "  threads=" << cfg.threads << "\n";

  // Main K-Means loop
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
      ++iter;
      break;
    }
  }

  auto wall_end = std::chrono::steady_clock::now();
  double elapsed =
      std::chrono::duration<double>(wall_end - wall_start).count();

  std::cout << (converged ? "Converged" : "Reached max_iter")
            << " after " << iter << " iterations.\n";
  std::cout << "Elapsed time: " << elapsed << " s\n";

  // Write results
  write_output_csv(cfg.output, labels, points);
  return 0;
}

