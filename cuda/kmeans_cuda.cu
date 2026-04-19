#include "kmeans_cuda.cuh"

#include "utils/io.h"
#include "utils/kmeans_common.h"
#include "utils/kmeans_utils.h"

#include <cfloat>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>
#include <iostream>
#include <random>
#include <vector>

// CUDA, assign each point to nearest centroid
__global__ void assign_kernel(const double *pts, const double *cents,
							  int *labels, int n, int k) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= n)
		return;

	double best_dist = DBL_MAX;
	int best_c = 0;

	for (int c = 0; c < k; ++c) {
		double d_sq = 0.0;
		for (int f = 0; f < NUM_FEATURES; ++f) {
			double diff =
				pts[i * NUM_FEATURES + f] - cents[c * NUM_FEATURES + f];
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
// Main CUDA K-Means
// ---------------------------------------------------------------------------
int run_kmeans_cuda(const Config &cfg) {
	auto wall_start = std::chrono::steady_clock::now();

	// Load data via C++ utility
	std::vector<Point> points_vec = load_data(cfg.input);
	if (points_vec.empty()) {
		std::cerr << "Error: no data loaded. Aborting.\n";
		return 1;
	}

	int n = points_vec.size();
	int k = cfg.k;
	int block_size = cfg.block_size;

	// Flatten points to device format: double[n * NUM_FEATURES]
	double *pts_flat = (double *)malloc(n * NUM_FEATURES * sizeof(double));
	for (int i = 0; i < n; ++i) {
		for (int f = 0; f < NUM_FEATURES; ++f) {
			pts_flat[i * NUM_FEATURES + f] = points_vec[i].features[f];
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
	cudaMemcpy(d_pts, pts_flat, n * NUM_FEATURES * sizeof(double),
			   cudaMemcpyHostToDevice);

	// Initialize centroids via C++ utility
	std::mt19937 rng(42);
	std::vector<Centroid> centroids_vec = init_centroids_pp(points_vec, k, rng);

	double *centroids = (double *)malloc(k * NUM_FEATURES * sizeof(double));
	for (int c = 0; c < k; ++c) {
		for (int f = 0; f < NUM_FEATURES; ++f) {
			centroids[c * NUM_FEATURES + f] = centroids_vec[c].features[f];
		}
	}

	std::cout << "K=" << k << "  max_iter=" << cfg.max_iter
			  << "  threshold=" << cfg.threshold << "  points=" << n
			  << "  block_size=" << block_size << "\n";

	// Allocate working memory
	int *labels = (int *)malloc(n * sizeof(int));
	double *new_centroids = (double *)malloc(k * NUM_FEATURES * sizeof(double));

	memset(labels, 0, n * sizeof(int));

	int iter = 0;
	bool converged = false;

	for (; iter < cfg.max_iter; ++iter) {
		// Copy centroids to device
		cudaMemcpy(d_cents, centroids, k * NUM_FEATURES * sizeof(double),
				   cudaMemcpyHostToDevice);

		// Launch kernel
		int grid_size = (n + block_size - 1) / block_size;
		assign_kernel<<<grid_size, block_size>>>(d_pts, d_cents, d_labels, n,
												 k);
		cudaDeviceSynchronize();

		// Copy labels back
		cudaMemcpy(labels, d_labels, n * sizeof(int), cudaMemcpyDeviceToHost);

		// Update centroids on CPU via C++ utility
		std::vector<Centroid> old_c(k), new_c(k);
		for (int c = 0; c < k; ++c) {
			for (int f = 0; f < NUM_FEATURES; ++f) {
				old_c[c].features[f] = centroids[c * NUM_FEATURES + f];
			}
		}
		std::vector<Centroid> new_c_vec = update_centroids_cpu(
			points_vec, std::vector<int>(labels, labels + n), k);
		for (int c = 0; c < k; ++c) {
			for (int f = 0; f < NUM_FEATURES; ++f) {
				new_centroids[c * NUM_FEATURES + f] = new_c_vec[c].features[f];
				new_c[c].features[f] = new_c_vec[c].features[f];
			}
		}

		converged = check_convergence(old_c, new_c, cfg.threshold);
		memcpy(centroids, new_centroids, k * NUM_FEATURES * sizeof(double));

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

	std::cout << (converged ? "Converged" : "Reached max_iter") << " after "
			  << iter << " iterations.\n";
	std::cout << "Elapsed time: " << elapsed << " s\n";

	// Write results via C++ utility
	write_output_csv(cfg.output, std::vector<int>(labels, labels + n),
					 points_vec);

	// Cleanup
	free(pts_flat);
	free(centroids);
	free(labels);
	free(new_centroids);
	cudaFree(d_pts);
	cudaFree(d_cents);
	cudaFree(d_labels);

	return 0;
}
