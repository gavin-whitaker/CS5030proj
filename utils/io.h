#pragma once

#include <string>
#include <vector>

#include "kmeans_common.h"

// Load CSV data from the given path, extract features, and apply min-max normalization.
std::vector<Point> load_data(const std::string &csv_path);

void write_output_csv(const std::string &csv_path,
                        const std::vector<int> &cluster_ids,
                        const std::vector<Point> &points);

