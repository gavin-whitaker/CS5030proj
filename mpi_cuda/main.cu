#include "kmeans_mpi_cuda.cuh"

#include "utils/args.h"

#include <cstdlib>
#include <iostream>

int main(int argc, char **argv) {
  Config cfg;
  cfg.output = "results/mpi_cuda_out.csv";
  parse_args(argc, argv, cfg);
  if (cfg.input.empty()) {
    print_usage(argv[0], "mpi_cuda --block_size <int>");
    return 1;
  }
  std::cout << "TODO: implement MPI+CUDA K-Means.\n";
  return run_kmeans_mpi_cuda(cfg);
}

