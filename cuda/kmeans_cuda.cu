#include "kmeans_cuda.cuh"

#include "utils/distance.h"
#include "utils/io.h"

#include <algorithm>
#include <cfloat>
#include <chrono>
#include <cmath>
#include <cuda_runtime.h>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <vector>

// ---------------------------------------------------------------------------
// CUDA kernel: assign each point to nearest centroid
// ---------------------------------------------------------------------------
__global__ void assign_kernel(const double *pts, const double *cents, int *labels,
                               int n, int k) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n)
    return;

  double best_dist = DBL_MAX;
  int best_c = 0;

  for (int c = 0; c < k; ++c) {
    double d_sq = 0.0;
    for (int f = 0; f < NUM_FEATURES; ++f) {
      double diff = pts[i * NUM_FEATURES + f] - cents[c * NUM_FEATURES + f];
      d_sq += diff * diff;
    }
    if (d_sq < best_dist) {
      best_dist = d_sq;
      best_c = c;
    }
  }

  labels[i] = best_c;
}

// ---------------------------------------------------------------------------
// K-Means++ initialization (CPU)
// ---------------------------------------------------------------------------
static std::vector<Centroid> init_centroids_pp(const std::vector<Point> &points,
                                                int k,
                                                std::mt19937 &rng) {
  std::vector<Centroid> centroids;
  centroids.reserve(k);

  std::uniform_int_distribution<size_t> uniform(0, points.size() - 1);
  Centroid first;
  first.features = points[uniform(rng)].features;
  centroids.push_back(first);

  std::vector<double> dist_sq(points.size(), std::numeric_limits<double>::max());

  for (int c = 1; c < k; ++c) {
    for (size_t i = 0; i < points.size(); ++i) {
      double d = euclidean_distance(points[i].features,
                                    centroids.back().features);
      double d2 = d * d;
      if (d2 < dist_sq[i])
        dist_sq[i] = d2;
    }

    std::discrete_distribution<size_t> weighted(dist_sq.begin(), dist_sq.end());
    Centroid next;
    next.features = points[weighted(rng)].features;
    centroids.push_back(next);
  }

  return centroids;
}

// ---------------------------------------------------------------------------
// CPU-side centroid update
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
// Convergence check (CPU)
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
// Main CUDA K-Means
// ---------------------------------------------------------------------------
int run_kmeans_cuda(const Config &cfg) {
  auto wall_start = std::chrono::steady_clock::now();

  // Load data
  std::vector<Point> points = load_data(cfg.input);
  if (points.empty()) {
    std::cerr << "Error: no data loaded. Aborting.\n";
    return 1;
  }

  size_t n = points.size();
  int k = cfg.k;
  int block_size = cfg.block_size;

  // Flatten points to device format: double[n * NUM_FEATURES]
  std::vector<double> pts_flat(n * NUM_FEATURES);
  for (size_t i = 0; i < n; ++i) {
    for (int f = 0; f < NUM_FEATURES; ++f) {
      pts_flat[i * NUM_FEATURES + f] = points[i].features[f];
    }
  }

  // Allocate device memory
  double *d_pts = nullptr;
  double *d_cents = nullptr;
  int *d_labels = nullptr;

  cudaMalloc(&d_pts, n * NUM_FEATURES * sizeof(double));
  cudaMalloc(&d_cents, k * NUM_FEATURES * sizeof(double));
  cudaMalloc(&d_labels, n * sizeof(int));

  // Copy points to device (once)
  cudaMemcpy(d_pts, pts_flat.data(), n * NUM_FEATURES * sizeof(double),
             cudaMemcpyHostToDevice);

  // Initialize centroids (K-Means++)
  std::mt19937 rng(42);
  std::vector<Centroid> centroids = init_centroids_pp(points, k, rng);

  std::cout << "K=" << k
            << "  max_iter=" << cfg.max_iter
            << "  threshold=" << cfg.threshold
            << "  points=" << n
            << "  block_size=" << block_size << "\n";

  // Main K-Means loop
  std::vector<int> labels(n, 0);
  int iter = 0;
  bool converged = false;

  for (; iter < cfg.max_iter; ++iter) {
    // Flatten centroids
    std::vector<double> cents_flat(k * NUM_FEATURES);
    for (int c = 0; c < k; ++c) {
      for (int f = 0; f < NUM_FEATURES; ++f) {
        cents_flat[c * NUM_FEATURES + f] = centroids[c].features[f];
      }
    }

    // Copy centroids to device
    cudaMemcpy(d_cents, cents_flat.data(), k * NUM_FEATURES * sizeof(double),
               cudaMemcpyHostToDevice);

    // Launch kernel
    int grid_size = (n + block_size - 1) / block_size;
    assign_kernel<<<grid_size, block_size>>>(d_pts, d_cents, d_labels, n, k);
    cudaDeviceSynchronize();

    // Copy labels back
    cudaMemcpy(labels.data(), d_labels, n * sizeof(int), cudaMemcpyDeviceToHost);

    // Update centroids on CPU
    std::vector<Centroid> new_centroids = update_centroids(points, labels, k);

    converged = check_convergence(centroids, new_centroids, cfg.threshold);
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

  // Cleanup
  cudaFree(d_pts);
  cudaFree(d_cents);
  cudaFree(d_labels);

  // Write results
  write_output_csv(cfg.output, labels, points);
  return 0;
}

