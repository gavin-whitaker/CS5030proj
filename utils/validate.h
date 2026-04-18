#pragma once

#include <string>

// Compare cluster assignments from serial and parallel implementations.
// Returns true if outputs match (within tolerance for song count).
bool validate_outputs(const std::string &serial_csv,
                       const std::string &parallel_csv,
                       double tolerance);

