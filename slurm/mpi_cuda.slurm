#!/bin/bash
#SBATCH --job-name=kmeans_mpi_cuda
#SBATCH --account=kingspeak-gpu
#SBATCH --partition=kingspeak-gpu
#SBATCH --nodes=2
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=results/mpi_cuda_%j.out
#SBATCH --error=results/mpi_cuda_%j.err

# Load required modules on CHPC Kingspeak
module load gcc cuda openmpi

# TODO: set paths to input data and output directory
INPUT=data/input.csv
OUTPUT=results/mpi_cuda_output.csv
K=10
MAX_ITER=100
DIMS=2

# Run MPI+CUDA K-Means (one MPI rank per GPU)
mpirun -n $SLURM_NTASKS ./mpi_cuda/kmeans_mpi_cuda $INPUT $K $MAX_ITER $DIMS $OUTPUT
