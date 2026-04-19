#!/bin/bash
#SBATCH --job-name=openmp-kmeans
#SBATCH --output=openmp-%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=00:15:00
#SBATCH --account=notchpeak-shared-short
#SBATCH --partition=notchpeak-shared-short

module load gcc

set -e

echo "=== OpenMP K-Means (Scaling Study 1) ==="
echo "Date: $(date)"
echo "Node: $(hostname)"
echo "Available CPUs: ${SLURM_CPUS_PER_TASK}"

# Serial baseline
echo ""
echo "--- Serial baseline ---"
srun ./results/serial \
  --input data/tracks_features.csv \
  --k 10 --max_iter 50 --threshold 0.001 \
  --output results/serial_out.csv

# OpenMP scaling: 1, 2, 4, 8, 16 threads
for THREADS in 1 2 4 8 16; do
  echo ""
  echo "--- OpenMP ${THREADS} threads ---"
  export OMP_NUM_THREADS=${THREADS}
  srun ./results/openmp \
    --input data/tracks_features.csv \
    --k 10 --max_iter 50 --threshold 0.001 \
    --threads ${THREADS} \
    --output results/openmp_${THREADS}t_out.csv
done

echo ""
echo "Done. Study 1 outputs: results/openmp_*t_out.csv"
