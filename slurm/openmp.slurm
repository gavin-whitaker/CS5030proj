#!/bin/bash
#SBATCH --job-name=kmeans_openmp
#SBATCH --account=kingspeak-shared-short
#SBATCH --partition=kingspeak-shared-short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=results/openmp_%j.out
#SBATCH --error=results/openmp_%j.err

# Load required modules on CHPC Kingspeak
module load gcc

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# TODO: set paths to input data and output directory
INPUT=data/input.csv
OUTPUT=results/openmp_output.csv
K=10
MAX_ITER=100
DIMS=2

# Run OpenMP K-Means
./openmp/kmeans_openmp $INPUT $K $MAX_ITER $DIMS $OMP_NUM_THREADS $OUTPUT
