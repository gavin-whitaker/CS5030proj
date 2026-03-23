#!/bin/bash
#SBATCH --job-name=kmeans_cuda
#SBATCH --account=kingspeak-gpu
#SBATCH --partition=kingspeak-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=results/cuda_%j.out
#SBATCH --error=results/cuda_%j.err

# Load required modules on CHPC Kingspeak
module load gcc cuda

# TODO: set paths to input data and output directory
INPUT=data/input.csv
OUTPUT=results/cuda_output.csv
K=10
MAX_ITER=100
DIMS=2

# Run CUDA K-Means
./cuda/kmeans_cuda $INPUT $K $MAX_ITER $DIMS $OUTPUT
