#include "tests/test_harness.h"
#include "utils/distance.h"
#include <cmath>
#include <array>

int main() {
  printf("Testing euclidean_distance...\n");

  std::array<double, NUM_FEATURES> a = {0, 0, 0, 0, 0, 0};
  std::array<double, NUM_FEATURES> b = {1, 1, 1, 1, 1, 1};
  std::array<double, NUM_FEATURES> c = {0, 0, 0, 0, 0, 0};
  std::array<double, NUM_FEATURES> d = {3, 4, 0, 0, 0, 0}; // dist = 5

  // Test 1: Distance from point to itself
  double dist_aa = euclidean_distance(a, a);
  CHECK_NEAR(dist_aa, 0.0, 1e-9);

  // Test 2: Symmetry
  double dist_ab = euclidean_distance(a, b);
  double dist_ba = euclidean_distance(b, a);
  CHECK_NEAR(dist_ab, dist_ba, 1e-9);

  // Test 3: All-zeros vs all-ones
  // sqrt(1^2 + 1^2 + 1^2 + 1^2 + 1^2 + 1^2) = sqrt(6) ≈ 2.449
  double expected_sqrt6 = std::sqrt(6.0);
  CHECK_NEAR(dist_ab, expected_sqrt6, 1e-7);

  // Test 4: 3-4-5 triangle (3D subset)
  // d = sqrt(3^2 + 4^2 + 0 + 0 + 0 + 0) = 5
  std::array<double, NUM_FEATURES> origin = {0, 0, 0, 0, 0, 0};
  double dist_3_4_0 = euclidean_distance(origin, d);
  CHECK_NEAR(dist_3_4_0, 5.0, 1e-7);

  // Test 5: Single feature differs
  std::array<double, NUM_FEATURES> e = {0.5, 0, 0, 0, 0, 0};
  std::array<double, NUM_FEATURES> f = {0, 0, 0, 0, 0, 0};
  double dist_e_f = euclidean_distance(e, f);
  CHECK_NEAR(dist_e_f, 0.5, 1e-9);

  // Test 6: Both points identical
  CHECK_NEAR(euclidean_distance(c, a), 0.0, 1e-9);

  // Test 7: Non-negative
  double dist_any = euclidean_distance(a, b);
  CHECK(dist_any >= 0.0);

  print_summary();
  return g_failures;
}
