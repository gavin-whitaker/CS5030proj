#include "tests/test_harness.h"
#include "utils/kmeans_common.h"

#include <cstring>

// Forward declare parse_args and print_usage (will be in utils/args.cpp)
void parse_args(int argc, char **argv, Config &cfg);
void print_usage(const char *prog, const char *backend_hint = "");

int main() {
  printf("Testing argument parsing...\n");

  // Test: parse_args with --k flag
  {
    Config cfg;
    cfg.k = 1; // Default

    const char *argv[] = {"prog", "--k", "5"};
    int argc = 3;

    parse_args(argc, const_cast<char**>(argv), cfg);
    CHECK_EQ(cfg.k, 5);
  }

  // Test: parse_args with --max_iter flag
  {
    Config cfg;
    cfg.max_iter = 10; // Default

    const char *argv[] = {"prog", "--max_iter", "100"};
    int argc = 3;

    parse_args(argc, const_cast<char**>(argv), cfg);
    CHECK_EQ(cfg.max_iter, 100);
  }

  // Test: parse_args with --threshold flag
  {
    Config cfg;
    cfg.threshold = 0.01; // Default

    const char *argv[] = {"prog", "--threshold", "0.001"};
    int argc = 3;

    parse_args(argc, const_cast<char**>(argv), cfg);
    CHECK_NEAR(cfg.threshold, 0.001, 1e-9);
  }

  // Test: parse_args with --input flag
  {
    Config cfg;
    cfg.input = ""; // Default empty

    const char *argv[] = {"prog", "--input", "data/test.csv"};
    int argc = 3;

    parse_args(argc, const_cast<char**>(argv), cfg);
    CHECK(cfg.input == "data/test.csv");
  }

  // Test: parse_args with --output flag
  {
    Config cfg;
    cfg.output = ""; // Default empty

    const char *argv[] = {"prog", "--output", "output.csv"};
    int argc = 3;

    parse_args(argc, const_cast<char**>(argv), cfg);
    CHECK(cfg.output == "output.csv");
  }

  // Test: parse_args with --threads flag
  {
    Config cfg;
    cfg.threads = 1; // Default

    const char *argv[] = {"prog", "--threads", "4"};
    int argc = 3;

    parse_args(argc, const_cast<char**>(argv), cfg);
    CHECK_EQ(cfg.threads, 4);
  }

  // Test: parse_args with --block_size flag
  {
    Config cfg;
    cfg.block_size = 256; // Default

    const char *argv[] = {"prog", "--block_size", "512"};
    int argc = 3;

    parse_args(argc, const_cast<char**>(argv), cfg);
    CHECK_EQ(cfg.block_size, 512);
  }

  // Test: parse_args with multiple flags
  {
    Config cfg;
    cfg.k = 1;
    cfg.max_iter = 10;
    cfg.threshold = 0.01;
    cfg.input = "";
    cfg.output = "";

    const char *argv[] = {"prog", "--k", "3", "--max_iter", "50", "--input", "data.csv", "--threshold", "0.005"};
    int argc = 9;

    parse_args(argc, const_cast<char**>(argv), cfg);
    CHECK_EQ(cfg.k, 3);
    CHECK_EQ(cfg.max_iter, 50);
    CHECK(cfg.input == "data.csv");
    CHECK_NEAR(cfg.threshold, 0.005, 1e-9);
  }

  // Test: parse_args ignores unknown flags
  {
    Config cfg;
    cfg.k = 1; // Default

    const char *argv[] = {"prog", "--unknown", "value", "--k", "2"};
    int argc = 5;

    parse_args(argc, const_cast<char**>(argv), cfg);
    CHECK_EQ(cfg.k, 2); // Should still parse --k
  }

  // Test: parse_args with value at end of args (no next arg)
  {
    Config cfg;
    cfg.k = 1; // Default

    const char *argv[] = {"prog", "--k"};
    int argc = 2;

    parse_args(argc, const_cast<char**>(argv), cfg);
    // k should remain unchanged (no value provided)
    CHECK_EQ(cfg.k, 1);
  }

  // Test: parse_args with invalid integer converts to 0
  {
    Config cfg;
    cfg.k = 1; // Default

    const char *argv[] = {"prog", "--k", "notanumber"};
    int argc = 3;

    parse_args(argc, const_cast<char**>(argv), cfg);
    // atoi() of "notanumber" is 0
    CHECK_EQ(cfg.k, 0);
  }

  // Test: parse_args with floating point
  {
    Config cfg;
    cfg.threshold = 0.01;

    const char *argv[] = {"prog", "--threshold", "0.123"};
    int argc = 3;

    parse_args(argc, const_cast<char**>(argv), cfg);
    CHECK_NEAR(cfg.threshold, 0.123, 1e-9);
  }

  // Test: print_usage with backend_hint (smoke test, no crash)
  {
    print_usage("test_prog", "serial");
    print_usage("test_prog", "");
    print_usage("test_prog");
    // If we reach here without segfault, test passes
    CHECK(true);
  }

  print_summary();
  return g_failures;
}
