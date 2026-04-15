#!/bin/bash
#SBATCH --job-name=openmp-kmeans
#SBATCH --output=openmp-%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:15:00
#SBATCH --account=notchpeak-shared-short
#SBATCH --partition=notchpeak-shared-short

module load gcc

set -e

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

srun ./results/openmp \
  --input data/songs.csv \
  --k 10 \
  --threads "${OMP_NUM_THREADS}" \
  --max_iter 100 \
  --threshold 0.001

