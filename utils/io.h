#ifndef IO_H
#define IO_H

// I/O utilities: reading datasets and writing results

#include "kmeans_common.h"
#include <string>
#include <vector>

// Read points from a CSV file (one point per line, comma-separated values)
std::vector<Point> read_points(const std::string &filename, int dims);

// Write cluster assignments to a CSV file
void write_results(const std::string &filename,
                   const std::vector<Point> &points,
                   const std::vector<Point> &centroids);

#endif // IO_H
