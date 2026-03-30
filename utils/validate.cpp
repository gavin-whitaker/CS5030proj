#include "validate.h"

#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <string>

// Parse a results CSV (song_id, cluster_id, features...) into a map.
// Returns false if the file cannot be opened.
static bool parse_result_csv(const std::string &path,
                              std::map<int, int> &out) {
  std::ifstream f(path);
  if (!f.is_open()) {
    std::cerr << "validate: cannot open file: " << path << "\n";
    return false;
  }
  std::string line;
  std::getline(f, line); // skip header
  while (std::getline(f, line)) {
    if (line.empty()) continue;
    std::istringstream ss(line);
    std::string tok;
    if (!std::getline(ss, tok, ',')) continue;
    int song_id = std::stoi(tok);
    if (!std::getline(ss, tok, ',')) continue;
    int cluster_id = std::stoi(tok);
    out[song_id] = cluster_id;
  }
  return true;
}

bool validate_outputs(const std::string &serial_csv,
                      const std::string &parallel_csv,
                      double tolerance) {
  (void)tolerance; // cluster IDs are integers; tolerance is used by validate.py

  std::map<int, int> serial_map, parallel_map;
  if (!parse_result_csv(serial_csv, serial_map)) return false;
  if (!parse_result_csv(parallel_csv, parallel_map)) return false;

  if (serial_map.size() != parallel_map.size()) {
    std::cerr << "validate: row count mismatch ("
              << serial_map.size() << " vs " << parallel_map.size() << ")\n";
    return false;
  }

  int mismatches = 0;
  for (const auto &kv : serial_map) {
    auto it = parallel_map.find(kv.first);
    if (it == parallel_map.end() || it->second != kv.second) {
      ++mismatches;
    }
  }

  if (mismatches == 0) {
    std::cout << "validate: PASS (" << serial_map.size() << " songs match)\n";
    return true;
  }
  std::cout << "validate: FAIL — " << mismatches << "/" << serial_map.size()
            << " cluster assignments differ\n";
  return false;
}

