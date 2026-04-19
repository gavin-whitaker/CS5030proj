#include "kmeans_mpi.h"

#include "utils/args.h"

#include <cstdlib>
#include <iostream>

int main(int argc, char **argv) {
  Config cfg;
  cfg.output = "results/mpi_out.csv";
  parse_args(argc, argv, cfg);
  if (cfg.input.empty()) {
    print_usage(argv[0], "mpi");
    return 1;
  }
  return run_kmeans_mpi(cfg);
}

