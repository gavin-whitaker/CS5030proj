#pragma once

#include "utils/kmeans_common.h"

void parse_args(int argc, char **argv, Config &cfg);
void print_usage(const char *prog, const char *backend_hint = "");
