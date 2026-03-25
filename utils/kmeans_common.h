#pragma once

#include <array>
#include <string>

// Shared configuration and data types for all K-Means implementations.
// TODO: replace placeholders with the full dataset/feature extraction once implementing logic.

struct Config {
  std::string input;
  std::string output;

  int k = 10;
  int max_iter = 100;
  double threshold = 0.001;

  // Runtime parameters for parallel implementations (optional for the baseline stubs).
  int threads = 1;     // OpenMP thread count
  int block_size = 256; // CUDA thread block size
};

// NOTE: The assignment focuses on selecting 3 features.
// This skeleton fixes it to 3 features for compilation purposes.
struct Point {
  int song_id = -1;
  std::array<double, 3> features{};
};

struct Centroid {
  std::array<double, 3> features{};
};

