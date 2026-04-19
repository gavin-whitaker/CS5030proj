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

echo "=== MPI K-Means (Scaling Study 3: Process Scaling) ==="
echo "Date: $(date)"
echo "Nodes: ${SLURM_JOB_NODELIST}"
echo "Total ranks: ${SLURM_NTASKS}"

# MPI scaling: 1, 2, 3, 4 processes
for NP in 1 2 3 4; do
  echo ""
  echo "--- MPI ${NP} ranks ---"
  mpirun -n ${NP} ./results/mpi \
    --input data/tracks_features.csv \
    --k 10 --max_iter 50 --threshold 0.001 \
    --output results/mpi_${NP}p_out.csv
done

echo ""
echo "Done. Study 3 outputs: results/mpi_*p_out.csv"
