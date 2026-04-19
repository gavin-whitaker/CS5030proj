#!/bin/bash
mkdir -p results
echo "Running serial..."
./results/serial --input tests/fixtures/small_100.csv --output results/serial_out.csv --k 3 --max_iter 50 --threshold 0.001
echo "Running openmp..."
./results/openmp --input tests/fixtures/small_100.csv --output results/openmp_out.csv --k 3 --max_iter 50 --threshold 0.001 --threads 4
echo "Running cuda..."
./results/cuda --input tests/fixtures/small_100.csv --output results/cuda_out.csv --k 3 --max_iter 50 --threshold 0.001 --block_size 256
echo "✓ Done. Check results/*.csv"
