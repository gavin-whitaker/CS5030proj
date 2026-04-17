#include "tests/test_harness.h"
#include "utils/io.h"
#include "utils/validate.h"
#include <cstdio>
#include <vector>

int main() {
  printf("Testing validate_outputs...\n");

  // Ensure temp directory exists
  std::system("mkdir -p tests/tmp");

  // Test 1: Identical files validate
  {
    auto points = load_data("tests/fixtures/small_10.csv");
    std::vector<int> cluster_ids(points.size());
    for (int i = 0; i < (int)cluster_ids.size(); ++i) {
      cluster_ids[i] = i % 3;
    }

    write_output_csv("tests/tmp/val_file1.csv", cluster_ids, points);
    write_output_csv("tests/tmp/val_file2.csv", cluster_ids, points);

    bool result = validate_outputs("tests/tmp/val_file1.csv", "tests/tmp/val_file2.csv", 0.0);
    CHECK(result);

    std::remove("tests/tmp/val_file1.csv");
    std::remove("tests/tmp/val_file2.csv");
  }

  // Test 2: Different cluster assignments fail validation
  {
    auto points = load_data("tests/fixtures/small_10.csv");
    std::vector<int> cluster_ids1(points.size());
    std::vector<int> cluster_ids2(points.size());

    for (int i = 0; i < (int)cluster_ids1.size(); ++i) {
      cluster_ids1[i] = i % 3;
      cluster_ids2[i] = (i + 1) % 3; // Shifted assignments
    }

    write_output_csv("tests/tmp/val_diff1.csv", cluster_ids1, points);
    write_output_csv("tests/tmp/val_diff2.csv", cluster_ids2, points);

    bool result = validate_outputs("tests/tmp/val_diff1.csv", "tests/tmp/val_diff2.csv", 0.0);
    CHECK(!result); // Should fail because assignments differ

    std::remove("tests/tmp/val_diff1.csv");
    std::remove("tests/tmp/val_diff2.csv");
  }

  // Test 3: Non-existent file fails
  {
    bool result = validate_outputs("tests/tmp/nonexistent1.csv", "tests/fixtures/small_10.csv", 0.0);
    CHECK(!result);
  }

  // Test 4: Tolerance parameter is ignored (document this limitation)
  {
    auto points = load_data("tests/fixtures/small_10.csv");
    std::vector<int> cluster_ids(points.size());
    for (int i = 0; i < (int)cluster_ids.size(); ++i) {
      cluster_ids[i] = i % 3;
    }

    write_output_csv("tests/tmp/val_tol1.csv", cluster_ids, points);
    write_output_csv("tests/tmp/val_tol2.csv", cluster_ids, points);

    // Tolerance should have no effect on exact cluster match
    bool result1 = validate_outputs("tests/tmp/val_tol1.csv", "tests/tmp/val_tol2.csv", 0.0);
    bool result2 = validate_outputs("tests/tmp/val_tol1.csv", "tests/tmp/val_tol2.csv", 1.0);

    CHECK(result1);
    CHECK(result2); // Both should pass for identical files
    // Note: tolerance is ignored in implementation, so different files will fail regardless

    std::remove("tests/tmp/val_tol1.csv");
    std::remove("tests/tmp/val_tol2.csv");
  }

  // Test 5: Row count mismatch fails
  {
    auto points10 = load_data("tests/fixtures/small_10.csv");
    auto points100 = load_data("tests/fixtures/small_100.csv");

    std::vector<int> ids10(points10.size(), 0);
    std::vector<int> ids100(points100.size(), 0);

    write_output_csv("tests/tmp/val_mismatch1.csv", ids10, points10);
    write_output_csv("tests/tmp/val_mismatch2.csv", ids100, points100);

    bool result = validate_outputs("tests/tmp/val_mismatch1.csv", "tests/tmp/val_mismatch2.csv", 0.0);
    CHECK(!result);

    std::remove("tests/tmp/val_mismatch1.csv");
    std::remove("tests/tmp/val_mismatch2.csv");
  }

  print_summary();
  return g_failures;
}
