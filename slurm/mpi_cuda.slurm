#!/bin/bash
#SBATCH --job-name=mpi-cuda-kmeans
#SBATCH --output=mpi_cuda-%j.out
#SBATCH --ntasks=4
#SBATCH --time=00:10:00

module load gcc
module load cuda
module load openmpi

set -e

mpirun -n "${SLURM_NTASKS:-4}" ./results/mpi_cuda \
  --input data/songs.csv \
  --k 10 \
  --block_size 256 \
  --max_iter 100 \
  --threshold 0.001

