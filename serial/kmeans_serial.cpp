#include "kmeans_serial.h"

#include "utils/distance.h"
#include "utils/io.h"
#include "utils/kmeans_utils.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <vector>

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
      double d_sq = 0.0;
      for (int f = 0; f < NUM_FEATURES; ++f) {
        double diff = points[i].features[f] - centroids[c].features[f];
        d_sq += diff * diff;
      }
      if (d_sq < best_dist) {
        best_dist = d_sq;
        best_c = c;
      }
    }
    labels[i] = best_c;
  }
  return labels;
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
    std::vector<Centroid> new_centroids = update_centroids_cpu(points, new_labels, cfg.k);

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

