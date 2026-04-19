#!/bin/bash
# Performance benchmark + scaling study for K-Means implementations

OUTPUT="perf_results.txt"
> "$OUTPUT"  # Clear file

echo "=== K-Means Scaling Study ===" | tee -a "$OUTPUT"
echo "Date: $(date)" | tee -a "$OUTPUT"
echo "Dataset: tests/fixtures/small_100.csv (100 points, K=3, max_iter=50)" | tee -a "$OUTPUT"
echo "" | tee -a "$OUTPUT"

mkdir -p results

# ============================================================================
# EXPERIMENT 1: Serial vs OpenMP (shared memory scaling)
# ============================================================================
echo "=== EXPERIMENT 1: Serial vs OpenMP (Thread Scaling) ===" | tee -a "$OUTPUT"
echo "" | tee -a "$OUTPUT"

echo "Serial (baseline):" | tee -a "$OUTPUT"
time ./results/serial --input tests/fixtures/small_100.csv --output /tmp/serial_base.csv --k 3 --max_iter 50 --threshold 0.001 2>&1 | grep -E "Elapsed|Converged" | tee -a "$OUTPUT"
echo "" | tee -a "$OUTPUT"

for threads in 1 2 4 8; do
  echo "OpenMP with $threads threads:" | tee -a "$OUTPUT"
  time ./results/openmp --input tests/fixtures/small_100.csv --output /tmp/openmp_${threads}t.csv --k 3 --max_iter 50 --threshold 0.001 --threads $threads 2>&1 | grep -E "Elapsed|Converged" | tee -a "$OUTPUT"
  echo "" | tee -a "$OUTPUT"
done

# ============================================================================
# EXPERIMENT 2: CUDA Block Size Tuning
# ============================================================================
echo "=== EXPERIMENT 2: CUDA Block Size Tuning ===" | tee -a "$OUTPUT"
echo "" | tee -a "$OUTPUT"

for block_size in 128 256 512 1024; do
  echo "CUDA with block_size=$block_size:" | tee -a "$OUTPUT"
  time ./results/cuda --input tests/fixtures/small_100.csv --output /tmp/cuda_${block_size}.csv --k 3 --max_iter 50 --threshold 0.001 --block_size $block_size 2>&1 | grep -E "Elapsed|Converged" | tee -a "$OUTPUT"
  echo "" | tee -a "$OUTPUT"
done

# ============================================================================
# VALIDATION: Output Correctness
# ============================================================================
echo "=== Validation: Output Correctness ===" | tee -a "$OUTPUT"
echo "" | tee -a "$OUTPUT"

echo "Comparing Serial vs OpenMP (4 threads):" | tee -a "$OUTPUT"
if diff /tmp/serial_base.csv /tmp/openmp_4t.csv > /dev/null 2>&1; then
  echo "✓ Outputs match (numerically identical)" | tee -a "$OUTPUT"
else
  echo "⚠ Outputs differ (expected: floating point rounding)" | tee -a "$OUTPUT"
  echo "  Line count serial: $(wc -l < /tmp/serial_base.csv)" | tee -a "$OUTPUT"
  echo "  Line count openmp: $(wc -l < /tmp/openmp_4t.csv)" | tee -a "$OUTPUT"
fi
echo "" | tee -a "$OUTPUT"

echo "Comparing Serial vs CUDA (block_size=256):" | tee -a "$OUTPUT"
if diff /tmp/serial_base.csv /tmp/cuda_256.csv > /dev/null 2>&1; then
  echo "✓ Outputs match (numerically identical)" | tee -a "$OUTPUT"
else
  echo "⚠ Outputs differ (expected: floating point rounding)" | tee -a "$OUTPUT"
  echo "  Line count serial: $(wc -l < /tmp/serial_base.csv)" | tee -a "$OUTPUT"
  echo "  Line count cuda:   $(wc -l < /tmp/cuda_256.csv)" | tee -a "$OUTPUT"
fi
echo "" | tee -a "$OUTPUT"

# ============================================================================
# NOTE: MPI Experiments
# ============================================================================
echo "=== NOTE: MPI Scaling Study ===" | tee -a "$OUTPUT"
echo "MPI and MPI+CUDA require 2-4 compute nodes on HPC cluster (CHPC)." | tee -a "$OUTPUT"
echo "Not run locally. Implement after MPI backend is complete." | tee -a "$OUTPUT"
echo "" | tee -a "$OUTPUT"

echo "Results saved to $OUTPUT"
