#!/bin/bash
#SBATCH --job-name=mpi-kmeans
#SBATCH --output=mpi-%j.out
#SBATCH --ntasks=4
#SBATCH --time=00:10:00

module load gcc
module load cuda
module load openmpi

set -e

mpirun -n "${SLURM_NTASKS:-4}" ./results/mpi \
  --input data/songs.csv \
  --k 10 \
  --max_iter 100 \
  --threshold 0.001

