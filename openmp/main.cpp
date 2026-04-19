#include "kmeans_openmp.h"

#include "utils/args.h"

int main(int argc, char **argv) {
	Config cfg;
	cfg.output = "results/openmp_out.csv";
	parse_args(argc, argv, cfg);
	if (cfg.input.empty()) {
		print_usage(argv[0], "openmp [--threads <int>]");
		return 1;
	}
	return run_kmeans_openmp(cfg);
}
