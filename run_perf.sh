#!/bin/bash
# Comprehensive performance benchmark for K-Means implementations
# Runs on full 1.2M Spotify dataset

set -e

OUTPUT="perf_results.txt"
> "$OUTPUT"

echo "=== K-Means Scaling Study (Full Dataset: 1.2M songs) ===" | tee -a "$OUTPUT"
echo "Date: $(date)" | tee -a "$OUTPUT"
echo "Dataset: data/tracks_features.csv (1,204,025 points, K=10, max_iter=50)" | tee -a "$OUTPUT"
echo "" | tee -a "$OUTPUT"

mkdir -p results

# Run benchmark and return just the time in seconds
run_once() {
  $1 2>&1 | grep "Elapsed time:" | awk '{print $3}'
}

# Run 3 times, compute average
avg_time() {
  local cmd="$1"
  local t1=$(run_once "$cmd")
  local t2=$(run_once "$cmd")
  local t3=$(run_once "$cmd")
  awk "BEGIN {printf \"%.5f\", ($t1 + $t2 + $t3) / 3}"
}

# ============================================================================
# Serial baseline
# ============================================================================
echo "=== EXPERIMENT 1: Serial vs OpenMP ===" | tee -a "$OUTPUT"
echo "" | tee -a "$OUTPUT"

serial_t=$(avg_time "./results/serial --input data/tracks_features.csv --output /tmp/serial.csv --k 10 --max_iter 50 --threshold 0.001")
echo "Serial: $serial_t s" | tee -a "$OUTPUT"

# OpenMP with different thread counts
for nt in 1 2 4 8; do
  t=$(avg_time "./results/openmp --input data/tracks_features.csv --output /tmp/omp_${nt}.csv --k 10 --max_iter 50 --threads $nt")
  speedup=$(awk "BEGIN {printf \"%.2f\", $serial_t / $t}")
  eff=$(awk "BEGIN {printf \"%.0f\", 100 * $serial_t / $t / $nt}")
  echo "OpenMP ($nt threads): $t s (speedup=$speedup, eff=${eff}%)" | tee -a "$OUTPUT"
done
echo "" | tee -a "$OUTPUT"

# ============================================================================
# CUDA block size tuning
# ============================================================================
echo "=== EXPERIMENT 2: CUDA Block Size ===" | tee -a "$OUTPUT"
echo "" | tee -a "$OUTPUT"

for bs in 64 128 256 512; do
  t=$(avg_time "./results/cuda --input data/tracks_features.csv --output /tmp/cuda_${bs}.csv --k 10 --max_iter 50 --block_size $bs")
  speedup=$(awk "BEGIN {printf \"%.2f\", $serial_t / $t}")
  echo "CUDA (block_size=$bs): $t s (speedup=$speedup)" | tee -a "$OUTPUT"
done
echo "" | tee -a "$OUTPUT"

# ============================================================================
# MPI process scaling
# ============================================================================
echo "=== EXPERIMENT 3: MPI Process Scaling ===" | tee -a "$OUTPUT"
echo "" | tee -a "$OUTPUT"

for np in 1 2 4; do
  t=$(avg_time "mpirun -n $np ./results/mpi --input data/tracks_features.csv --output /tmp/mpi_${np}.csv --k 10 --max_iter 50")
  speedup=$(awk "BEGIN {printf \"%.2f\", $serial_t / $t}")
  eff=$(awk "BEGIN {printf \"%.0f\", 100 * $serial_t / $t / $np}")
  echo "MPI ($np processes): $t s (speedup=$speedup, eff=${eff}%)" | tee -a "$OUTPUT"
done
echo "" | tee -a "$OUTPUT"

# ============================================================================
# MPI+CUDA process scaling
# ============================================================================
echo "=== EXPERIMENT 4: MPI+CUDA Process Scaling ===" | tee -a "$OUTPUT"
echo "" | tee -a "$OUTPUT"

for np in 1 2 4; do
  t=$(avg_time "mpirun -n $np ./results/mpi_cuda --input data/tracks_features.csv --output /tmp/mpi_cuda_${np}.csv --k 10 --max_iter 50 --block_size 256")
  speedup=$(awk "BEGIN {printf \"%.2f\", $serial_t / $t}")
  eff=$(awk "BEGIN {printf \"%.0f\", 100 * $serial_t / $t / $np}")
  echo "MPI+CUDA ($np processes): $t s (speedup=$speedup, eff=${eff}%)" | tee -a "$OUTPUT"
done
echo "" | tee -a "$OUTPUT"

# ============================================================================
# Validation
# ============================================================================
echo "=== Validation ===" | tee -a "$OUTPUT"
source .venv/bin/activate 2>/dev/null || true

for cfg in "omp_4" "cuda_256" "mpi_2" "mpi_cuda_1"; do
  python scripts/validate.py --serial /tmp/serial.csv --parallel /tmp/${cfg}.csv 2>&1 | grep -E "(PASS|FAIL)" | tee -a "$OUTPUT"
done
echo "" | tee -a "$OUTPUT"

echo "Results saved to $OUTPUT" | tee -a "$OUTPUT"
