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

echo "=== CUDA K-Means (Scaling Study 2: Block Size Tuning) ==="
echo "Date: $(date)"
echo "Node: $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true

# Run with different block sizes: 64, 128, 256, 512
for BS in 64 128 256 512; do
  echo ""
  echo "--- CUDA block_size=${BS} ---"
  srun ./results/cuda \
    --input data/tracks_features.csv \
    --k 10 --max_iter 50 --threshold 0.001 \
    --block_size ${BS} \
    --output results/cuda_bs${BS}_out.csv
done

echo ""
echo "Done. Study 2 outputs: results/cuda_bs*_out.csv"
