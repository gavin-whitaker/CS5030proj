#!/bin/bash
#SBATCH --job-name=kmeans_serial
#SBATCH --account=kingspeak-shared-short
#SBATCH --partition=kingspeak-shared-short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=01:00:00
#SBATCH --output=results/serial_%j.out
#SBATCH --error=results/serial_%j.err

# Load required modules on CHPC Kingspeak
module load gcc

# TODO: set paths to input data and output directory
INPUT=data/input.csv
OUTPUT=results/serial_output.csv
K=10
MAX_ITER=100
DIMS=2

# Run serial K-Means
./serial/kmeans_serial $INPUT $K $MAX_ITER $DIMS $OUTPUT
