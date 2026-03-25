#include "kmeans_mpi.h"

#include <cstdlib>
#include <iostream>
#include <string>

static void print_usage(const char *prog) {
  std::cout << "Usage: " << prog
            << " --input <csv> --k <int> [--max_iter <int>] "
               "[--threshold <double>]\n";
}

static void parse_args(int argc, char **argv, Config &cfg) {
  for (int i = 1; i < argc; ++i) {
    std::string arg(argv[i]);

    auto take_value = [&](std::string &out) {
      if (i + 1 < argc) out = argv[++i];
    };

    if (arg == "--input") {
      take_value(cfg.input);
    } else if (arg == "--output") {
      take_value(cfg.output);
    } else if (arg == "--k") {
      std::string v;
      take_value(v);
      if (!v.empty()) cfg.k = std::atoi(v.c_str());
    } else if (arg == "--max_iter") {
      std::string v;
      take_value(v);
      if (!v.empty()) cfg.max_iter = std::atoi(v.c_str());
    } else if (arg == "--threshold") {
      std::string v;
      take_value(v);
      if (!v.empty()) cfg.threshold = std::atof(v.c_str());
    } else if (arg == "--threads") {
      std::string v;
      take_value(v);
      if (!v.empty()) cfg.threads = std::atoi(v.c_str());
    } else if (arg == "--block_size") {
      std::string v;
      take_value(v);
      if (!v.empty()) cfg.block_size = std::atoi(v.c_str());
    } else if (arg == "--help" || arg == "-h") {
      print_usage(argv[0]);
      std::exit(0);
    }
  }
}

int main(int argc, char **argv) {
  Config cfg;
  cfg.output = "results/mpi_out.csv";
  parse_args(argc, argv, cfg);
  std::cout << "TODO: argument parsing done. TODO: implement MPI K-Means.\n";
  return run_kmeans_mpi(cfg);
}

