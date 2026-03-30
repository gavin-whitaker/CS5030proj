#pragma once

#include <array>
#include <string>

// Number of audio features extracted per song.
// Features (in order): danceability, energy, acousticness,
//                      instrumentalness, valence, tempo (normalized)
static constexpr int NUM_FEATURES = 6;

// Human-readable feature names matching the column order above.
static constexpr const char *FEATURE_NAMES[NUM_FEATURES] = {
    "danceability", "energy", "acousticness",
    "instrumentalness", "valence", "tempo"};

// Shared configuration used by all K-Means implementations.
struct Config {
  std::string input;
  std::string output;

  int k = 10;
  int max_iter = 100;
  double threshold = 0.001;

  int threads = 1;      // OpenMP thread count
  int block_size = 256; // CUDA thread block size
};

struct Point {
  int song_id = -1;
  std::array<double, NUM_FEATURES> features{};
};

struct Centroid {
  std::array<double, NUM_FEATURES> features{};
};

