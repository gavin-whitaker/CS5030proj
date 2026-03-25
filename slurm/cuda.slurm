#!/bin/bash
#SBATCH --job-name=cuda-kmeans
#SBATCH --output=cuda-%j.out
#SBATCH --ntasks=1
#SBATCH --time=00:10:00

module load gcc
module load cuda
module load openmpi

set -e

srun ./results/cuda \
  --input data/songs.csv \
  --k 10 \
  --block_size 256 \
  --max_iter 100 \
  --threshold 0.001

