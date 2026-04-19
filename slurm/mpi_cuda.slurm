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

echo "=== MPI+CUDA K-Means (Scaling Study 4: Hybrid Scaling) ==="
echo "Date: $(date)"
echo "Nodes: ${SLURM_JOB_NODELIST}"
echo "Total ranks: ${SLURM_NTASKS}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true

# MPI+CUDA scaling: 1, 2, 3, 4 ranks (one GPU per rank)
for NP in 1 2 3 4; do
  echo ""
  echo "--- MPI+CUDA ${NP} ranks ---"
  mpirun -n ${NP} ./results/mpi_cuda \
    --input data/tracks_features.csv \
    --k 10 --max_iter 50 --threshold 0.001 \
    --block_size 256 \
    --output results/mpi_cuda_${NP}p_out.csv
done

echo ""
echo "Done. Study 4 outputs: results/mpi_cuda_*p_out.csv"
