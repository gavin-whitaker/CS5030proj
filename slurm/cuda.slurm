#!/bin/bash
#SBATCH --job-name=cuda-kmeans
#SBATCH --output=cuda-%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --time=00:15:00
#SBATCH --account=notchpeak-gpu
#SBATCH --partition=notchpeak-gpu

module load gcc
module load cuda

set -e

srun ./results/cuda \
  --input data/songs.csv \
  --k 10 \
  --block_size 256 \
  --max_iter 100 \
  --threshold 0.001

