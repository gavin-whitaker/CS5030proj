#include "kmeans_cuda.cuh"

#include "utils/args.h"

#include <cstdlib>
#include <iostream>

int main(int argc, char **argv) {
  Config cfg;
  cfg.output = "results/cuda_out.csv";
  parse_args(argc, argv, cfg);
  if (cfg.input.empty()) {
    print_usage(argv[0], "cuda --block_size <int>");
    return 1;
  }
  return run_kmeans_cuda(cfg);
}

