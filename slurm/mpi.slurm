#!/bin/bash
#SBATCH --job-name=kmeans_mpi
#SBATCH --account=kingspeak-shared-short
#SBATCH --partition=kingspeak-shared-short
#SBATCH --nodes=2
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=2G
#SBATCH --time=01:00:00
#SBATCH --output=results/mpi_%j.out
#SBATCH --error=results/mpi_%j.err

# Load required modules on CHPC Kingspeak
module load gcc openmpi

# TODO: set paths to input data and output directory
INPUT=data/input.csv
OUTPUT=results/mpi_output.csv
K=10
MAX_ITER=100
DIMS=2

# Run MPI K-Means
mpirun -n $SLURM_NTASKS ./mpi/kmeans_mpi $INPUT $K $MAX_ITER $DIMS $OUTPUT
