#include "io.h"

#include <algorithm>
#include <fstream>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>

// ---------------------------------------------------------------------------
// CSV tokenizer: handles double-quoted fields that may contain commas/newlines
// ---------------------------------------------------------------------------
static std::vector<std::string> parse_csv_line(const std::string &line) {
  std::vector<std::string> fields;
  std::string field;
  bool in_quotes = false;

  for (size_t i = 0; i < line.size(); ++i) {
    char c = line[i];
    if (c == '"') {
      if (in_quotes && i + 1 < line.size() && line[i + 1] == '"') {
        field += '"';
        ++i;
      } else {
        in_quotes = !in_quotes;
      }
    } else if (c == ',' && !in_quotes) {
      fields.push_back(field);
      field.clear();
    } else {
      field += c;
    }
  }
  fields.push_back(field);
  return fields;
}

// ---------------------------------------------------------------------------
// Column indices in the Kaggle Spotify 1.2M dataset (tracks_features.csv)
// Header: id,name,album,album_id,artists,artist_ids,track_number,disc_number,
//         explicit,danceability,energy,key,loudness,mode,speechiness,
//         acousticness,instrumentalness,liveness,valence,tempo,...
// ---------------------------------------------------------------------------
static constexpr int COL_DANCEABILITY    =  9;
static constexpr int COL_ENERGY          = 10;
static constexpr int COL_ACOUSTICNESS    = 15;
static constexpr int COL_INSTRUMENTALNESS = 16;
static constexpr int COL_VALENCE         = 18;
static constexpr int COL_TEMPO           = 19;

// Feature order mirrors FEATURE_NAMES in kmeans_common.h:
// [0] danceability  [1] energy  [2] acousticness
// [3] instrumentalness  [4] valence  [5] tempo
static constexpr int FEATURE_COLS[NUM_FEATURES] = {
    COL_DANCEABILITY, COL_ENERGY, COL_ACOUSTICNESS,
    COL_INSTRUMENTALNESS, COL_VALENCE, COL_TEMPO};

// ---------------------------------------------------------------------------
// load_data: parse CSV, extract features, apply per-feature min-max norm
// ---------------------------------------------------------------------------
std::vector<Point> load_data(const std::string &csv_path) {
  std::ifstream file(csv_path);
  if (!file.is_open()) {
    std::cerr << "Error: cannot open input file: " << csv_path << "\n";
    return {};
  }

  std::vector<Point> points;
  points.reserve(1300000);

  std::string line;
  // Skip header
  if (!std::getline(file, line)) {
    std::cerr << "Error: empty file: " << csv_path << "\n";
    return {};
  }

  int row = 0;
  int required_cols = *std::max_element(FEATURE_COLS, FEATURE_COLS + NUM_FEATURES) + 1;

  while (std::getline(file, line)) {
    if (line.empty()) continue;

    auto fields = parse_csv_line(line);
    if (static_cast<int>(fields.size()) < required_cols) {
      // Skip malformed rows silently
      ++row;
      continue;
    }

    Point p;
    p.song_id = row;
    bool valid = true;
    for (int f = 0; f < NUM_FEATURES; ++f) {
      try {
        p.features[f] = std::stod(fields[FEATURE_COLS[f]]);
      } catch (...) {
        valid = false;
        break;
      }
    }
    if (valid) {
      points.push_back(p);
    }
    ++row;
  }

  if (points.empty()) {
    std::cerr << "Warning: no valid data rows loaded from " << csv_path << "\n";
    return points;
  }

  // Per-feature min-max normalization so all features are in [0, 1].
  // Tempo is ~0–250; others are already ~0–1; normalization makes them comparable.
  std::array<double, NUM_FEATURES> fmin, fmax;
  fmin.fill(std::numeric_limits<double>::max());
  fmax.fill(std::numeric_limits<double>::lowest());

  for (const auto &p : points) {
    for (int f = 0; f < NUM_FEATURES; ++f) {
      if (p.features[f] < fmin[f]) fmin[f] = p.features[f];
      if (p.features[f] > fmax[f]) fmax[f] = p.features[f];
    }
  }

  for (auto &p : points) {
    for (int f = 0; f < NUM_FEATURES; ++f) {
      double range = fmax[f] - fmin[f];
      p.features[f] = (range > 0.0) ? (p.features[f] - fmin[f]) / range : 0.0;
    }
  }

  std::cout << "Loaded " << points.size() << " songs from " << csv_path << "\n";
  return points;
}

// ---------------------------------------------------------------------------
// write_output_csv: song_id, cluster_id, f0, f1, ..., f5
// ---------------------------------------------------------------------------
void write_output_csv(const std::string &csv_path,
                      const std::vector<int> &cluster_ids,
                      const std::vector<Point> &points) {
  if (cluster_ids.size() != points.size()) {
    std::cerr << "Error: cluster_ids and points size mismatch in write_output_csv\n";
    return;
  }

  std::ofstream out(csv_path);
  if (!out.is_open()) {
    std::cerr << "Error: cannot open output file: " << csv_path << "\n";
    return;
  }

  // Header
  out << "song_id,cluster_id";
  for (int f = 0; f < NUM_FEATURES; ++f) {
    out << "," << FEATURE_NAMES[f];
  }
  out << "\n";

  out.precision(8);
  out << std::fixed;
  for (size_t i = 0; i < points.size(); ++i) {
    out << points[i].song_id << "," << cluster_ids[i];
    for (int f = 0; f < NUM_FEATURES; ++f) {
      out << "," << points[i].features[f];
    }
    out << "\n";
  }

  std::cout << "Output written to " << csv_path << "\n";
}

