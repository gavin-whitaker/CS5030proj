#pragma once

#include <string>

// TODO: implement output validation by comparing a serial CSV and a parallel CSV.
bool validate_outputs(const std::string &serial_csv,
                       const std::string &parallel_csv,
                       double tolerance);

