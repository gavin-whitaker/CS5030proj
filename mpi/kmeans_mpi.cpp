#include "kmeans_mpi.h"

#include "utils/io.h"
#include "utils/kmeans_common.h"
#include "utils/kmeans_utils.h"

#include <cfloat>
#include <chrono>
#include <cstring>
#include <iostream>
#include <mpi.h>
#include <random>
#include <vector>

int run_kmeans_mpi(const Config &cfg) {
  MPI_Init(nullptr, nullptr);

  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  auto wall_start = std::chrono::steady_clock::now();

  int n = 0, k = cfg.k;
  std::vector<Point> all_points;
  double *all_pts_flat = nullptr;

  // Load data on rank 0
  if (rank == 0) {
    all_points = load_data(cfg.input);
    n = all_points.size();
    if (all_points.empty()) {
      std::cerr << "Error: no data loaded. Aborting.\n";
      n = -1;
    }
  }
  MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);
  if (n < 0) {
    MPI_Finalize();
    return 1;
  }

  // Compute local chunk sizes for uneven split across ranks
  int local_n = n / size + (rank < (n % size) ? 1 : 0);
  std::vector<int> counts(size), displs(size);
  for (int r = 0; r < size; ++r) {
    int sz = n / size + (r < (n % size) ? 1 : 0);
    counts[r] = sz * NUM_FEATURES;
    displs[r] = (r == 0) ? 0 : displs[r - 1] + counts[r - 1];
  }

  // Flatten and scatter points
  if (rank == 0) {
    all_pts_flat = new double[n * NUM_FEATURES];
    for (int i = 0; i < n; ++i) {
      for (int f = 0; f < NUM_FEATURES; ++f) {
        all_pts_flat[i * NUM_FEATURES + f] = all_points[i].features[f];
      }
    }
  }

  double *local_pts = new double[local_n * NUM_FEATURES];
  MPI_Scatterv(all_pts_flat, counts.data(), displs.data(), MPI_DOUBLE,
               local_pts, local_n * NUM_FEATURES, MPI_DOUBLE, 0,
               MPI_COMM_WORLD);

  // Scatter song IDs
  std::vector<int> id_counts(size), id_displs(size);
  for (int r = 0; r < size; ++r) {
    int sz = n / size + (r < (n % size) ? 1 : 0);
    id_counts[r] = sz;
    id_displs[r] = (r == 0) ? 0 : id_displs[r - 1] + id_counts[r - 1];
  }

  int *local_ids = new int[local_n];
  int *all_ids = nullptr;
  if (rank == 0) {
    all_ids = new int[n];
    for (int i = 0; i < n; ++i) all_ids[i] = all_points[i].song_id;
  }
  MPI_Scatterv(all_ids, id_counts.data(), id_displs.data(), MPI_INT, local_ids,
               local_n, MPI_INT, 0, MPI_COMM_WORLD);

  // Init centroids on rank 0, broadcast
  double *centroids = new double[k * NUM_FEATURES];
  if (rank == 0) {
    std::mt19937 rng(42);
    auto cv = init_centroids_pp(all_points, k, rng);
    for (int c = 0; c < k; ++c) {
      for (int f = 0; f < NUM_FEATURES; ++f) {
        centroids[c * NUM_FEATURES + f] = cv[c].features[f];
      }
    }
    std::cout << "K=" << k << "  max_iter=" << cfg.max_iter
              << "  threshold=" << cfg.threshold << "  points=" << n
              << "  mpi_ranks=" << size << "\n";
  }
  MPI_Bcast(centroids, k * NUM_FEATURES, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  // Working buffers
  int *local_labels = new int[local_n];
  double *new_centroids = new double[k * NUM_FEATURES];
  double *partial_sums = new double[k * NUM_FEATURES];
  int *partial_counts = new int[k];
  int *global_counts = new int[k];
  memset(local_labels, 0, local_n * sizeof(int));

  int iter = 0;
  bool converged = false;

  for (; iter < cfg.max_iter; ++iter) {
    // Assign points to nearest centroid (local)
    for (int i = 0; i < local_n; ++i) {
      double best = DBL_MAX;
      int best_c = 0;
      for (int c = 0; c < k; ++c) {
        double d = 0.0;
        for (int f = 0; f < NUM_FEATURES; ++f) {
          double diff = local_pts[i * NUM_FEATURES + f] -
                        centroids[c * NUM_FEATURES + f];
          d += diff * diff;
        }
        if (d < best) {
          best = d;
          best_c = c;
        }
      }
      local_labels[i] = best_c;
    }

    // Compute partial sums and counts
    memset(partial_sums, 0, k * NUM_FEATURES * sizeof(double));
    memset(partial_counts, 0, k * sizeof(int));
    for (int i = 0; i < local_n; ++i) {
      int c = local_labels[i];
      for (int f = 0; f < NUM_FEATURES; ++f)
        partial_sums[c * NUM_FEATURES + f] += local_pts[i * NUM_FEATURES + f];
      partial_counts[c]++;
    }

    // Allreduce to get global sums and counts
    MPI_Allreduce(partial_sums, new_centroids, k * NUM_FEATURES, MPI_DOUBLE,
                  MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(partial_counts, global_counts, k, MPI_INT, MPI_SUM,
                  MPI_COMM_WORLD);

    // Update centroids (all ranks compute identically)
    std::vector<Centroid> old_c(k), new_c(k);
    for (int c = 0; c < k; ++c) {
      for (int f = 0; f < NUM_FEATURES; ++f) {
        old_c[c].features[f] = centroids[c * NUM_FEATURES + f];
        if (global_counts[c] > 0)
          new_centroids[c * NUM_FEATURES + f] /= global_counts[c];
        else
          new_centroids[c * NUM_FEATURES + f] =
              centroids[c * NUM_FEATURES + f];
        new_c[c].features[f] = new_centroids[c * NUM_FEATURES + f];
      }
    }

    converged = check_convergence(old_c, new_c, cfg.threshold);
    memcpy(centroids, new_centroids, k * NUM_FEATURES * sizeof(double));

    // Sync convergence flag across all ranks
    int local_conv = converged ? 1 : 0, global_conv;
    MPI_Allreduce(&local_conv, &global_conv, 1, MPI_INT, MPI_LAND,
                  MPI_COMM_WORLD);
    converged = (global_conv != 0);

    if (rank == 0 && (iter + 1) % 10 == 0)
      std::cout << "  iter " << (iter + 1) << " done\n";

    if (converged) {
      ++iter;
      break;
    }
  }

  auto wall_end = std::chrono::steady_clock::now();
  double elapsed = std::chrono::duration<double>(wall_end - wall_start).count();

  // Gather labels at rank 0
  int *all_labels = nullptr;
  if (rank == 0) all_labels = new int[n];
  MPI_Gatherv(local_labels, local_n, MPI_INT, all_labels, id_counts.data(),
              id_displs.data(), MPI_INT, 0, MPI_COMM_WORLD);

  if (rank == 0) {
    std::cout << (converged ? "Converged" : "Reached max_iter") << " after "
              << iter << " iterations.\n";
    std::cout << "Elapsed time: " << elapsed << " s\n";
    write_output_csv(cfg.output, std::vector<int>(all_labels, all_labels + n),
                     all_points);
    delete[] all_labels;
    delete[] all_pts_flat;
    delete[] all_ids;
  }

  delete[] local_pts;
  delete[] local_ids;
  delete[] centroids;
  delete[] local_labels;
  delete[] new_centroids;
  delete[] partial_sums;
  delete[] partial_counts;
  delete[] global_counts;

  MPI_Finalize();
  return 0;
}
