#include "tests/test_harness.h"
#include "serial/kmeans_serial.h"
#include "utils/io.h"
#include <cstdio>
#include <fstream>

int main() {
  printf("Testing serial K-Means implementation...\n");

  // Ensure temp directory exists
  std::system("mkdir -p tests/tmp");

  // Test 1: K=1 on small fixture (should converge in 1 iteration)
  {
    Config cfg;
    cfg.input = "tests/fixtures/small_10.csv";
    cfg.output = "tests/tmp/serial_k1_out.csv";
    cfg.k = 1;
    cfg.max_iter = 100;
    cfg.threshold = 0.001;

    int result = run_kmeans_serial(cfg);
    CHECK_EQ(result, 0);

    // Verify output file exists and has correct format
    std::ifstream f(cfg.output);
    CHECK(f.good());

    // Check header
    std::string header;
    std::getline(f, header);
    CHECK(header.find("song_id") != std::string::npos);

    // Count rows (should be 10 data rows + 1 header)
    int row_count = 1; // Already read header
    std::string line;
    while (std::getline(f, line)) {
      if (!line.empty()) row_count++;
    }
    CHECK_EQ(row_count, 11);

    f.close();
    std::remove(cfg.output.c_str());
  }

  // Test 2: K=2 on small fixture
  {
    Config cfg;
    cfg.input = "tests/fixtures/small_10.csv";
    cfg.output = "tests/tmp/serial_k2_out.csv";
    cfg.k = 2;
    cfg.max_iter = 50;
    cfg.threshold = 0.001;

    int result = run_kmeans_serial(cfg);
    CHECK_EQ(result, 0);

    auto points = load_data(cfg.input);
    std::ifstream f(cfg.output);
    int row_count = 0;
    std::string line;
    std::getline(f, line); // skip header
    while (std::getline(f, line)) {
      if (!line.empty()) row_count++;
    }
    CHECK_EQ(row_count, (int)points.size());

    f.close();
    std::remove(cfg.output.c_str());
  }

  // Test 3: K=5 on 100-row fixture
  {
    Config cfg;
    cfg.input = "tests/fixtures/small_100.csv";
    cfg.output = "tests/tmp/serial_k5_out.csv";
    cfg.k = 5;
    cfg.max_iter = 50;
    cfg.threshold = 0.001;

    int result = run_kmeans_serial(cfg);
    CHECK_EQ(result, 0);

    auto points = load_data(cfg.input);
    std::ifstream f(cfg.output);
    int row_count = 0;
    std::string line;
    std::getline(f, line); // skip header
    while (std::getline(f, line)) {
      if (!line.empty()) row_count++;
    }
    CHECK_EQ(row_count, (int)points.size());

    f.close();
    std::remove(cfg.output.c_str());
  }

  // Test 4: max_iter=1 (only 1 iteration)
  {
    Config cfg;
    cfg.input = "tests/fixtures/small_10.csv";
    cfg.output = "tests/tmp/serial_iter1_out.csv";
    cfg.k = 3;
    cfg.max_iter = 1;
    cfg.threshold = 0.001;

    int result = run_kmeans_serial(cfg);
    CHECK_EQ(result, 0);

    std::ifstream f(cfg.output);
    CHECK(f.good());

    f.close();
    std::remove(cfg.output.c_str());
  }

  // Test 5: Empty input file returns error
  {
    // Create empty file with just header
    std::ofstream empty("tests/tmp/empty.csv");
    empty << "id,name\n"; // Header only
    empty.close();

    Config cfg;
    cfg.input = "tests/tmp/empty.csv";
    cfg.output = "tests/tmp/empty_out.csv";
    cfg.k = 2;
    cfg.max_iter = 10;
    cfg.threshold = 0.001;

    int result = run_kmeans_serial(cfg);
    CHECK(result != 0); // Should return error

    std::remove("tests/tmp/empty.csv");
  }

  // Test 6: K >= n_points is handled
  {
    // With 10 points and k=15, should still work (just have empty clusters)
    Config cfg;
    cfg.input = "tests/fixtures/small_10.csv";
    cfg.output = "tests/tmp/serial_k_big_out.csv";
    cfg.k = 15;
    cfg.max_iter = 10;
    cfg.threshold = 0.001;

    int result = run_kmeans_serial(cfg);
    CHECK_EQ(result, 0);

    auto points = load_data(cfg.input);
    std::ifstream f(cfg.output);
    int row_count = 0;
    std::string line;
    std::getline(f, line); // skip header
    while (std::getline(f, line)) {
      if (!line.empty()) row_count++;
    }
    CHECK_EQ(row_count, (int)points.size());

    f.close();
    std::remove(cfg.output.c_str());
  }

  // Test 7: Large threshold converges in 1 iteration
  {
    Config cfg;
    cfg.input = "tests/fixtures/small_10.csv";
    cfg.output = "tests/tmp/serial_big_threshold_out.csv";
    cfg.k = 2;
    cfg.max_iter = 100;
    cfg.threshold = 1e6; // Very large: should converge immediately

    int result = run_kmeans_serial(cfg);
    CHECK_EQ(result, 0);

    std::remove(cfg.output.c_str());
  }

  print_summary();
  return g_failures;
}
