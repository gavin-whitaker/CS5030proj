#include "tests/test_harness.h"
#include "utils/kmeans_common.h"
#include "utils/distance.h"

#include <random>
#include <vector>
#include <cmath>

// Forward declare the functions we'll test (from kmeans_serial.cpp for now)
std::vector<Centroid> init_centroids_pp(const std::vector<Point> &points,
                                         int k,
                                         std::mt19937 &rng);
std::vector<Centroid> update_centroids(const std::vector<Point> &points,
                                       const std::vector<int> &labels,
                                       int k);
bool check_convergence(const std::vector<Centroid> &old_c,
                       const std::vector<Centroid> &new_c,
                       double threshold);

int main() {
  printf("Testing K-Means utility functions...\n");

  // Test: init_centroids_pp returns k centroids
  {
    std::vector<Point> points(10);
    for (int i = 0; i < 10; ++i) {
      for (int f = 0; f < NUM_FEATURES; ++f) {
        points[i].features[f] = static_cast<double>(i * f);
      }
    }

    std::mt19937 rng(42);
    int k = 3;
    auto centroids = init_centroids_pp(points, k, rng);

    CHECK_EQ((int)centroids.size(), k);
  }

  // Test: init_centroids_pp with k=1
  {
    std::vector<Point> points(5);
    for (int i = 0; i < 5; ++i) {
      for (int f = 0; f < NUM_FEATURES; ++f) {
        points[i].features[f] = static_cast<double>(i);
      }
    }

    std::mt19937 rng(42);
    auto centroids = init_centroids_pp(points, 1, rng);
    CHECK_EQ((int)centroids.size(), 1);
  }

  // Test: init_centroids_pp centroids within data bounds
  {
    std::vector<Point> points(10);
    double min_val = 0.0, max_val = 100.0;
    for (int i = 0; i < 10; ++i) {
      for (int f = 0; f < NUM_FEATURES; ++f) {
        points[i].features[f] = min_val + (max_val - min_val) * i / 10.0;
      }
    }

    std::mt19937 rng(42);
    auto centroids = init_centroids_pp(points, 3, rng);

    for (const auto &c : centroids) {
      for (int f = 0; f < NUM_FEATURES; ++f) {
        // Centroids should be taken directly from points
        bool found = false;
        for (const auto &p : points) {
          if (std::abs(c.features[f] - p.features[f]) < 1e-10) {
            found = true;
            break;
          }
        }
        CHECK(found);
      }
    }
  }

  // Test: update_centroids with trivial case (all points in one cluster)
  {
    std::vector<Point> points(3);
    for (int i = 0; i < 3; ++i) {
      for (int f = 0; f < NUM_FEATURES; ++f) {
        points[i].features[f] = static_cast<double>(i + 1);
      }
    }
    std::vector<int> labels = {0, 0, 0};

    auto centroids = update_centroids(points, labels, 1);
    CHECK_EQ((int)centroids.size(), 1);

    // Centroid should be mean of all points
    for (int f = 0; f < NUM_FEATURES; ++f) {
      double expected = (1.0 + 2.0 + 3.0) / 3.0;
      CHECK_NEAR(centroids[0].features[f], expected, 1e-9);
    }
  }

  // Test: update_centroids with two clusters
  {
    std::vector<Point> points(4);
    for (int i = 0; i < 4; ++i) {
      for (int f = 0; f < NUM_FEATURES; ++f) {
        points[i].features[f] = static_cast<double>(i);
      }
    }
    // First two points in cluster 0, last two in cluster 1
    std::vector<int> labels = {0, 0, 1, 1};

    auto centroids = update_centroids(points, labels, 2);
    CHECK_EQ((int)centroids.size(), 2);

    // Cluster 0: mean of points 0, 1
    for (int f = 0; f < NUM_FEATURES; ++f) {
      double expected = (0.0 + 1.0) / 2.0;
      CHECK_NEAR(centroids[0].features[f], expected, 1e-9);
    }

    // Cluster 1: mean of points 2, 3
    for (int f = 0; f < NUM_FEATURES; ++f) {
      double expected = (2.0 + 3.0) / 2.0;
      CHECK_NEAR(centroids[1].features[f], expected, 1e-9);
    }
  }

  // Test: update_centroids with empty cluster
  {
    std::vector<Point> points(3);
    for (int i = 0; i < 3; ++i) {
      for (int f = 0; f < NUM_FEATURES; ++f) {
        points[i].features[f] = static_cast<double>(i + 1);
      }
    }
    std::vector<int> labels = {0, 0, 0}; // All in cluster 0, cluster 1 empty

    auto centroids = update_centroids(points, labels, 2);
    CHECK_EQ((int)centroids.size(), 2);

    // Cluster 0: mean of points
    for (int f = 0; f < NUM_FEATURES; ++f) {
      double expected = (1.0 + 2.0 + 3.0) / 3.0;
      CHECK_NEAR(centroids[0].features[f], expected, 1e-9);
    }
    // Cluster 1: unchanged (all zeros from default init)
    for (int f = 0; f < NUM_FEATURES; ++f) {
      CHECK_NEAR(centroids[1].features[f], 0.0, 1e-9);
    }
  }

  // Test: check_convergence returns true when all moved less than threshold
  {
    std::vector<Centroid> old_c(2), new_c(2);
    for (int f = 0; f < NUM_FEATURES; ++f) {
      old_c[0].features[f] = 0.0;
      new_c[0].features[f] = 0.01; // Moved 0.01

      old_c[1].features[f] = 10.0;
      new_c[1].features[f] = 10.005; // Moved ~0.005
    }

    bool converged = check_convergence(old_c, new_c, 0.1); // Threshold 0.1
    CHECK(converged);
  }

  // Test: check_convergence returns false when at least one moved >= threshold
  {
    std::vector<Centroid> old_c(2), new_c(2);
    for (int f = 0; f < NUM_FEATURES; ++f) {
      old_c[0].features[f] = 0.0;
      new_c[0].features[f] = 0.2; // Moved 0.2

      old_c[1].features[f] = 10.0;
      new_c[1].features[f] = 10.0; // No movement
    }

    bool converged = check_convergence(old_c, new_c, 0.1); // Threshold 0.1
    CHECK(!converged);
  }

  // Test: check_convergence with no movement
  {
    std::vector<Centroid> old_c(1), new_c(1);
    for (int f = 0; f < NUM_FEATURES; ++f) {
      old_c[0].features[f] = 5.0;
      new_c[0].features[f] = 5.0;
    }

    bool converged = check_convergence(old_c, new_c, 0.001);
    CHECK(converged);
  }

  // Test: check_convergence exactly at threshold
  {
    std::vector<Centroid> old_c(1), new_c(1);
    double threshold = 0.5;
    double dist = 0.5; // Exactly at threshold

    // Create points that move by exactly dist
    for (int f = 0; f < NUM_FEATURES; ++f) {
      old_c[0].features[f] = 0.0;
      new_c[0].features[f] = dist / std::sqrt(NUM_FEATURES);
    }

    bool converged = check_convergence(old_c, new_c, threshold);
    CHECK(!converged); // >= threshold means not converged
  }

  print_summary();
  return g_failures;
}
