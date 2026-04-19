#pragma once

#include <random>
#include <vector>
#include "utils/kmeans_common.h"

std::vector<Centroid> init_centroids_pp(const std::vector<Point> &points,
                                         int k,
                                         std::mt19937 &rng);

std::vector<Centroid> update_centroids_cpu(const std::vector<Point> &points,
                                           const std::vector<int> &labels,
                                           int k);

bool check_convergence(const std::vector<Centroid> &old_c,
                       const std::vector<Centroid> &new_c,
                       double threshold);
