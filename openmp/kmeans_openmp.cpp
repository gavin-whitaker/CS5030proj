#include "kmeans_openmp.h"

#include "utils/io.h"
#include "utils/kmeans_common.h"
#include "utils/kmeans_utils.h"

#include <array>
#include <cfloat>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <omp.h>
#include <random>
#include <vector>

// Parallel, each point assigned to nearest centroid
static void assign_clusters(const Point *points, int n, const double *centroids,
							int k, int *labels) {
#pragma omp parallel for schedule(static)
	for (int i = 0; i < n; ++i) {
		double best_dist = DBL_MAX;
		int best_c = 0;

		for (int c = 0; c < k; ++c) {
			double d_sq = 0.0;
			for (int f = 0; f < NUM_FEATURES; ++f) {
				double diff =
					points[i].features[f] - centroids[c * NUM_FEATURES + f];
				d_sq += diff * diff;
			}
			if (d_sq < best_dist) {
				best_dist = d_sq;
				best_c = c;
			}
		}
		labels[i] = best_c;
	}
}

// Parallel centroid update with thread-private accumulators
static void update_centroids(const Point *points, int n, const int *labels,
							 int k, double *new_centroids, int *counts) {
	int nthreads = omp_get_max_threads();

	// [thread][centroid][feature]
	double **sums = (double **)malloc(nthreads * sizeof(double *));
	int **thread_counts = (int **)malloc(nthreads * sizeof(int *));

	for (int t = 0; t < nthreads; ++t) {
		sums[t] = (double *)calloc(k * NUM_FEATURES, sizeof(double));
		thread_counts[t] = (int *)calloc(k, sizeof(int));
	}

// Parallel accumulation
#pragma omp parallel
	{
		int tid = omp_get_thread_num();
#pragma omp for schedule(static)
		for (int i = 0; i < n; ++i) {
			int c = labels[i];
			for (int f = 0; f < NUM_FEATURES; ++f) {
				sums[tid][c * NUM_FEATURES + f] += points[i].features[f];
			}
			thread_counts[tid][c]++;
		}
	}

	// Serial merge
	memset(new_centroids, 0, k * NUM_FEATURES * sizeof(double));
	memset(counts, 0, k * sizeof(int));

	for (int t = 0; t < nthreads; ++t) {
		for (int c = 0; c < k; ++c) {
			counts[c] += thread_counts[t][c];
			for (int f = 0; f < NUM_FEATURES; ++f) {
				new_centroids[c * NUM_FEATURES + f] +=
					sums[t][c * NUM_FEATURES + f];
			}
		}
	}

	for (int c = 0; c < k; ++c) {
		if (counts[c] > 0) {
			for (int f = 0; f < NUM_FEATURES; ++f) {
				new_centroids[c * NUM_FEATURES + f] /= counts[c];
			}
		}
	}

	// Cleanup
	for (int t = 0; t < nthreads; ++t) {
		free(sums[t]);
		free(thread_counts[t]);
	}
	free(sums);
	free(thread_counts);
}

// Main OpenMP K-Means
int run_kmeans_openmp(const Config &cfg) {
	auto wall_start = std::chrono::steady_clock::now();

	omp_set_num_threads(cfg.threads);

	std::vector<Point> points_vec = load_data(cfg.input);
	if (points_vec.empty()) {
		fprintf(stderr, "Error: no data loaded. Aborting.\n");
		return 1;
	}

	int n = points_vec.size();
	int k = cfg.k;

	Point *points = (Point *)malloc(n * sizeof(Point));
	for (int i = 0; i < n; ++i) {
		points[i] = points_vec[i];
	}

	std::mt19937 rng(42);
	std::vector<Centroid> centroids_vec = init_centroids_pp(points_vec, k, rng);

	double *centroids = (double *)malloc(k * NUM_FEATURES * sizeof(double));
	for (int c = 0; c < k; ++c) {
		for (int f = 0; f < NUM_FEATURES; ++f) {
			centroids[c * NUM_FEATURES + f] = centroids_vec[c].features[f];
		}
	}

	printf("K=%d  max_iter=%d  threshold=%g  points=%d  threads=%d\n", k,
		   cfg.max_iter, cfg.threshold, n, cfg.threads);

	int *labels = (int *)malloc(n * sizeof(int));
	int *new_labels = (int *)malloc(n * sizeof(int));
	double *new_centroids = (double *)malloc(k * NUM_FEATURES * sizeof(double));
	int *counts = (int *)malloc(k * sizeof(int));

	memset(labels, 0, n * sizeof(int));

	int iter = 0;
	bool converged = false;

	for (; iter < cfg.max_iter; ++iter) {
		// Assign points to centroids
		assign_clusters(points, n, centroids, k, new_labels);
		memcpy(labels, new_labels, n * sizeof(int));

		// Update centroids
		update_centroids(points, n, labels, k, new_centroids, counts);

		// Check convergence
		std::vector<Centroid> old_c(k), new_c(k);
		for (int c = 0; c < k; ++c) {
			for (int f = 0; f < NUM_FEATURES; ++f) {
				old_c[c].features[f] = centroids[c * NUM_FEATURES + f];
				new_c[c].features[f] = new_centroids[c * NUM_FEATURES + f];
			}
		}
		converged = check_convergence(old_c, new_c, cfg.threshold);

		memcpy(centroids, new_centroids, k * NUM_FEATURES * sizeof(double));

		if ((iter + 1) % 10 == 0) {
			printf("  iter %d done\n", iter + 1);
		}

		if (converged) {
			++iter;
			break;
		}
	}

	auto wall_end = std::chrono::steady_clock::now();
	double elapsed =
		std::chrono::duration<double>(wall_end - wall_start).count();

	printf("%s after %d iterations.\n",
		   converged ? "Converged" : "Reached max_iter", iter);
	printf("Elapsed time: %g s\n", elapsed);

	write_output_csv(cfg.output, std::vector<int>(labels, labels + n),
					 points_vec);

	free(points);
	free(centroids);
	free(labels);
	free(new_labels);
	free(new_centroids);
	free(counts);

	return 0;
}
