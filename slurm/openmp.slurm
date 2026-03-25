#!/bin/bash
#SBATCH --job-name=openmp-kmeans
#SBATCH --output=openmp-%j.out
#SBATCH --ntasks=1
#SBATCH --time=00:10:00

module load gcc
module load cuda
module load openmpi

set -e

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

srun ./results/openmp \
  --input data/songs.csv \
  --k 10 \
  --threads "${OMP_NUM_THREADS}" \
  --max_iter 100 \
  --threshold 0.001

