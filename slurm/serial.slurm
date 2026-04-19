#!/bin/bash
#SBATCH --job-name=serial-kmeans
#SBATCH --output=serial-%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:30:00
#SBATCH --account=notchpeak-shared-short
#SBATCH --partition=notchpeak-shared-short

module load gcc

set -e

echo "=== Serial K-Means ==="
echo "Date: $(date)"
echo "Node: $(hostname)"

srun ./results/serial \
  --input data/tracks_features.csv \
  --k 10 \
  --max_iter 50 \
  --threshold 0.001 \
  --output results/serial_out.csv

echo "Done. Output: results/serial_out.csv"
