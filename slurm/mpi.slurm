#!/bin/bash
#SBATCH --job-name=mpi-kmeans
#SBATCH --output=mpi-%j.out
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:15:00
#SBATCH --account=notchpeak-shared-short
#SBATCH --partition=notchpeak-shared-short

module load gcc
module load openmpi

set -e

mpirun -n "${SLURM_NTASKS}" ./results/mpi \
  --input data/songs.csv \
  --k 10 \
  --max_iter 100 \
  --threshold 0.001

