#include "io.h"

#include <iostream>

std::vector<Point> load_data(const std::string &csv_path) {
  (void)csv_path;
  // TODO: parse CSV at runtime and extract selected audio features.
  return {};
}

void write_output_csv(const std::string &csv_path,
                        const std::vector<int> &cluster_ids,
                        const std::vector<Point> &points) {
  (void)csv_path;
  (void)cluster_ids;
  (void)points;
  // TODO: write the required output CSV format:
  // song_id, cluster_id, feature_1, feature_2, feature_3, ...
  // For now, do nothing (skeleton only).
}

