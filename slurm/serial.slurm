#!/bin/bash
#SBATCH --job-name=serial-kmeans
#SBATCH --output=serial-%j.out
#SBATCH --ntasks=1
#SBATCH --time=00:10:00

module load gcc
module load cuda
module load openmpi

set -e

srun ./results/serial \
  --input data/songs.csv \
  --k 10 \
  --max_iter 100 \
  --threshold 0.001

