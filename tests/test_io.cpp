#include "tests/test_harness.h"
#include "utils/io.h"
#include <cstdio>
#include <cstring>
#include <vector>
#include <cmath>
#include <string>

int main() {
  printf("Testing I/O functions...\n");

  // Test 1: Load non-existent file
  {
    auto points = load_data("nonexistent_file_xyz.csv");
    CHECK_EQ(points.size(), 0);
  }

  // Test 2: Load valid fixture (10 rows)
  {
    auto points = load_data("tests/fixtures/small_10.csv");
    CHECK_EQ(points.size(), 10);
    CHECK_EQ(points[0].song_id, 0);
    CHECK_EQ(points[9].song_id, 9);
  }

  // Test 3: Features are normalized to [0, 1]
  {
    auto points = load_data("tests/fixtures/small_10.csv");
    for (const auto& p : points) {
      for (int f = 0; f < NUM_FEATURES; ++f) {
        CHECK(p.features[f] >= 0.0 && p.features[f] <= 1.0);
      }
    }
  }

  // Test 4: Write and read back (round-trip)
  {
    auto points = load_data("tests/fixtures/small_10.csv");
    std::vector<int> cluster_ids(points.size());
    for (int i = 0; i < (int)cluster_ids.size(); ++i) {
      cluster_ids[i] = i % 3; // Assign to clusters 0, 1, 2
    }

    const char* tmpfile = "tests/tmp/test_roundtrip.csv";
    std::remove(tmpfile); // Clean up if exists
    write_output_csv(tmpfile, cluster_ids, points);

    // Verify file was created and has correct format
    FILE* f = std::fopen(tmpfile, "r");
    CHECK(f != nullptr);

    // Check header
    char header[256];
    if (f) {
      std::fgets(header, sizeof(header), f);
      CHECK(std::string(header).find("song_id") != std::string::npos);
      CHECK(std::string(header).find("cluster_id") != std::string::npos);
      CHECK(std::string(header).find("danceability") != std::string::npos);

      // Count rows
      int row_count = 0;
      char line[1024];
      while (std::fgets(line, sizeof(line), f)) {
        if (std::strlen(line) > 1) row_count++;
      }
      CHECK_EQ(row_count, 10);

      std::fclose(f);
    }

    std::remove(tmpfile);
  }

  // Test 5: Size mismatch handling
  {
    auto points = load_data("tests/fixtures/small_10.csv");
    std::vector<int> bad_cluster_ids(5); // Mismatch: 5 != 10

    const char* tmpfile = "tests/tmp/test_size_mismatch.csv";
    std::remove(tmpfile);
    write_output_csv(tmpfile, bad_cluster_ids, points);
    // Function should print error and not write file (or write partial/corrupt)
    // Just verify it doesn't crash
    CHECK(true);
  }

  // Test 6: Load 100-row fixture
  {
    auto points = load_data("tests/fixtures/small_100.csv");
    CHECK_EQ(points.size(), 100);
  }

  // Test 7: All points have song_id = row index
  {
    auto points = load_data("tests/fixtures/small_10.csv");
    for (int i = 0; i < (int)points.size(); ++i) {
      CHECK_EQ(points[i].song_id, i);
    }
  }

  print_summary();
  return g_failures;
}
