#pragma once

#include <string>
#include <vector>

#include "kmeans_common.h"

// TODO: implement CSV parsing (load_data) and output CSV writing (write_output_csv).

std::vector<Point> load_data(const std::string &csv_path);

void write_output_csv(const std::string &csv_path,
                        const std::vector<int> &cluster_ids,
                        const std::vector<Point> &points);

