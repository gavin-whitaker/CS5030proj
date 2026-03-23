// io.cpp — Implementation of I/O utilities for K-Means clustering
// TODO: implement read_points and write_results

#include "io.h"
#include <fstream>
#include <sstream>
#include <stdexcept>

std::vector<Point> read_points(const std::string &filename, int dims) {
    // TODO: open filename, parse CSV rows into Point structs
    std::vector<Point> points;
    return points;
}

void write_results(const std::string &filename,
                   const std::vector<Point> &points,
                   const std::vector<Point> &centroids) {
    // TODO: write cluster assignments and centroid coordinates to filename
}
