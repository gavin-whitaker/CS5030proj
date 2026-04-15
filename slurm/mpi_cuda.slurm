#!/bin/bash
#SBATCH --job-name=mpi-cuda-kmeans
#SBATCH --output=mpi_cuda-%j.out
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --time=00:15:00
#SBATCH --account=notchpeak-gpu
#SBATCH --partition=notchpeak-gpu

module load gcc
module load cuda
module load openmpi

set -e

mpirun -n "${SLURM_NTASKS}" ./results/mpi_cuda \
  --input data/songs.csv \
  --k 10 \
  --block_size 256 \
  --max_iter 100 \
  --threshold 0.001

