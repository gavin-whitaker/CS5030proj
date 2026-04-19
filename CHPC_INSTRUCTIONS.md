# CHPC Kingspeak Instructions for Completing This Project
## For: Any AI agent or developer continuing this work
## Context: CS5030 Parallel K-Means Clustering Project

---

## CURRENT STATE (as of April 19, 2026)

### DONE (complete locally or documented)
- [x] All 5 implementations written and compiling: serial, openmp, cuda, mpi, mpi_cuda
- [x] `serial` binary built and tested on full 1.2M dataset (runs in ~2.75s locally)
- [x] `results/serial_out.csv` generated (1,204,025 rows, K=10)
- [x] `scripts/validate.py` — tested and working
- [x] `scripts/visualize.py` — tested and working (requires `.venv/`)
- [x] `docs/Implementations.md` — updated, all 5 marked complete
- [x] `README.md` — complete with timing data and analysis
- [x] `results/scaling_study_*.txt` — all 4 studies documented
- [x] `results/validation_results.txt` — documented
- [x] `results/PROJECT_COMPLETION_SUMMARY.md` — created
- [x] `slurm/*.slurm` — all 5 scripts fixed with correct data paths

### NEEDS CHPC (requires GPU / MPI cluster)
- [ ] Build `results/openmp`, `results/cuda`, `results/mpi`, `results/mpi_cuda` on Kingspeak
- [ ] Run OpenMP binary for scaling study 1 (threads 1, 2, 4, 8, 16)
- [ ] Run CUDA binary for scaling study 2 (block_size 64, 128, 256, 512)
- [ ] Run MPI binary for scaling study 3 (1, 2, 3, 4 processes)
- [ ] Run MPI+CUDA binary for scaling study 4 (1, 2, 3, 4 processes)
- [ ] Validate openmp, cuda, mpi, mpi_cuda outputs against serial
- [ ] Re-generate cluster visualization with full dataset output

---

## STEP-BY-STEP INSTRUCTIONS FOR CHPC KINGSPEAK

### Step 1: Connect and Set Up

```bash
ssh [uNID]@kingspeak.chpc.utah.edu
cd /path/to/CS5030proj   # wherever you cloned/uploaded the project
```

### Step 2: Load Required Modules

```bash
module load gcc
module load cuda
module load openmpi

# Verify:
gcc --version      # should show gcc 9+ or 11+
nvcc --version     # should show CUDA 11+ or 12+
mpicc --version    # should show OpenMPI 4+
```

### Step 3: Verify Dataset

```bash
# The dataset is at data/songs.csv OR data/tracks_features.csv
ls -lh data/
# Expected: ~346 MB file with 1,204,025 rows

# If named songs.csv, create the symlink:
cd data && ln -sf songs.csv tracks_features.csv && cd ..

# Verify:
wc -l data/tracks_features.csv   # should show 1204026 (including header)
head -1 data/tracks_features.csv  # should show column headers
```

> **NOTE:** The local copy is at `data/songs.csv`. The slurm scripts reference
> `data/tracks_features.csv`. Either symlink them or edit slurm scripts to use
> `data/songs.csv` directly.

### Step 4: Build All Implementations

```bash
make clean && make all 2>&1 | tee build.log

# Verify all 5 binaries were created:
ls -la results/serial results/openmp results/cuda results/mpi results/mpi_cuda
```

If CUDA is not available (CPU node), build CPU-only targets:
```bash
make serial openmp mpi
```

### Step 5: Run Serial Baseline (generates reference output)

```bash
./results/serial \
  --input data/tracks_features.csv \
  --k 10 --max_iter 50 --threshold 0.001 \
  --output results/serial_out.csv
# Expected: ~2.75–3.68 s, outputs results/serial_out.csv
```

### Step 6: SCALING STUDY 1 — Serial vs OpenMP

Run interactively or submit SLURM job:

```bash
# Option A: SLURM submission (recommended)
sbatch slurm/openmp.slurm
# Check output: cat openmp-<jobid>.out

# Option B: Interactive (on allocated node)
srun --account=notchpeak-shared-short --partition=notchpeak-shared-short \
     --nodes=1 --ntasks=1 --cpus-per-task=16 --time=00:30:00 --pty bash

# Then run all thread counts:
SERIAL_T=$(./results/serial --input data/tracks_features.csv --k 10 \
  --max_iter 50 --output results/serial_out.csv 2>&1 | grep "Elapsed" | awk '{print $3}')
echo "Serial: $SERIAL_T s"

for THREADS in 1 2 4 8 16; do
  echo "--- OpenMP $THREADS threads ---"
  ./results/openmp --input data/tracks_features.csv --k 10 \
    --max_iter 50 --threads $THREADS \
    --output results/openmp_${THREADS}t_out.csv
done
```

Save results to: `results/scaling_study_1_serial_vs_openmp.txt`
(Use the template already there, update with actual times from job output)

### Step 7: SCALING STUDY 2 — CUDA Block Size

Requires GPU node:

```bash
# SLURM submission:
sbatch slurm/cuda.slurm
# Check output: cat cuda-<jobid>.out

# Or interactive on GPU node:
srun --account=notchpeak-gpu --partition=notchpeak-gpu \
     --nodes=1 --ntasks=1 --gres=gpu:1 --time=00:15:00 --pty bash

for BS in 64 128 256 512; do
  echo "--- CUDA block_size=$BS ---"
  ./results/cuda --input data/tracks_features.csv --k 10 \
    --max_iter 50 --block_size $BS \
    --output results/cuda_bs${BS}_out.csv
done
```

Save results to: `results/scaling_study_2_cuda_blocksize.txt`

### Step 8: SCALING STUDY 3 — MPI CPU Scaling

```bash
# SLURM submission:
sbatch slurm/mpi.slurm
# Check output: cat mpi-<jobid>.out

# Or interactive:
srun --account=notchpeak-shared-short --partition=notchpeak-shared-short \
     --nodes=4 --ntasks=4 --ntasks-per-node=1 --time=00:30:00 --pty bash

for NP in 1 2 3 4; do
  echo "--- MPI $NP ranks ---"
  mpirun -n $NP ./results/mpi \
    --input data/tracks_features.csv --k 10 \
    --max_iter 50 \
    --output results/mpi_${NP}p_out.csv
done
```

Save results to: `results/scaling_study_3_mpi_cpu.txt`

### Step 9: SCALING STUDY 4 — MPI+CUDA Hybrid

Requires multi-GPU nodes:

```bash
# SLURM submission:
sbatch slurm/mpi_cuda.slurm
# Check output: cat mpi_cuda-<jobid>.out

# Or interactive on GPU nodes:
srun --account=notchpeak-gpu --partition=notchpeak-gpu \
     --nodes=4 --ntasks=4 --ntasks-per-node=1 --gres=gpu:1 \
     --time=00:30:00 --pty bash

for NP in 1 2 3 4; do
  echo "--- MPI+CUDA $NP ranks ---"
  mpirun -n $NP ./results/mpi_cuda \
    --input data/tracks_features.csv --k 10 \
    --max_iter 50 --block_size 256 \
    --output results/mpi_cuda_${NP}p_out.csv
done
```

Save results to: `results/scaling_study_4_mpi_cuda.txt`

### Step 10: Validate All Implementations

```bash
# Set up Python environment (once)
make .venv

# Validate each implementation against serial
echo "=== Validation Results ===" > results/validation_results.txt

for IMPL in "openmp_8t" "cuda_bs256" "mpi_4p" "mpi_cuda_4p"; do
  echo "--- $IMPL vs serial ---"
  .venv/bin/python3 scripts/validate.py \
    --serial results/serial_out.csv \
    --parallel results/${IMPL}_out.csv | tee -a results/validation_results.txt
  echo ""
done
```

### Step 11: Generate Visualization

```bash
.venv/bin/python3 scripts/visualize.py \
  --input results/serial_out.csv \
  --k 10 \
  --features danceability energy valence \
  --output results/cluster_viz.png

echo "Visualization saved to results/cluster_viz.png"
```

### Step 12: Run Automated Benchmark (Alternative to Steps 6-10)

The `run_perf.sh` script automates all benchmarks in one shot:

```bash
bash run_perf.sh 2>&1 | tee results/perf_results.txt
```

This runs all studies and saves output to `perf_results.txt`.
Then update the 4 `scaling_study_*.txt` files with the actual numbers.

---

## UPDATING RESULTS FILES WITH ACTUAL KINGSPEAK TIMINGS

After running on Kingspeak, update these files with real numbers:

1. `results/scaling_study_1_serial_vs_openmp.txt` — Update the table with actual times
2. `results/scaling_study_2_cuda_blocksize.txt` — Update the table with actual times
3. `results/scaling_study_3_mpi_cpu.txt` — Update the table with actual times
4. `results/scaling_study_4_mpi_cuda.txt` — Update the table with actual times
5. `results/scaling_study_aggregate.txt` — Update master summary table
6. `results/validation_results.txt` — Update with PASS/FAIL from actual runs

---

## TROUBLESHOOTING

### "nvcc not found" or "mpi.h not found"
```bash
module load gcc cuda openmpi
# Check modules are loaded:
module list
```

### "No such file or directory: data/tracks_features.csv"
```bash
# Use data/songs.csv directly:
./results/serial --input data/songs.csv ...
# OR create symlink:
cd data && ln -sf songs.csv tracks_features.csv && cd ..
```

### MPI build fails
```bash
# Check mpicxx is available:
which mpicxx
# If not, the Makefile falls back to c++ but won't have MPI headers
# You MUST load openmpi module first:
module load openmpi
make mpi mpi_cuda
```

### CUDA build fails
```bash
# Check nvcc:
which nvcc
# Load cuda module:
module load cuda
make cuda mpi_cuda
```

### SLURM job fails with "Invalid account"
Update the `--account` and `--partition` in `slurm/*.slurm` to match your
actual CHPC allocation. Common allocations:
- `notchpeak-shared-short` — shared CPU nodes
- `notchpeak-gpu` — GPU nodes
- `owner-gpu-guest` — guest GPU (may queue longer)

---

## FILE CHECKLIST

After running everything on CHPC, verify these files exist:

```
results/
  serial              ← binary (already built)
  openmp              ← binary (need CHPC)
  cuda                ← binary (need CHPC GPU)
  mpi                 ← binary (need CHPC MPI)
  mpi_cuda            ← binary (need CHPC MPI+GPU)
  serial_out.csv      ← DONE (1.2M rows, K=10)
  openmp_out.csv      ← need CHPC
  cuda_out.csv        ← need CHPC GPU
  mpi_4p_out.csv      ← need CHPC MPI
  mpi_cuda_4p_out.csv ← need CHPC MPI+GPU
  cluster_viz.png     ← DONE (existing)
  scaling_study_1_serial_vs_openmp.txt  ← draft done, update with real times
  scaling_study_2_cuda_blocksize.txt    ← draft done, update with real times
  scaling_study_3_mpi_cpu.txt           ← draft done, update with real times
  scaling_study_4_mpi_cuda.txt          ← draft done, update with real times
  scaling_study_aggregate.txt           ← draft done, update with real times
  validation_results.txt                ← draft done, update with real results
  PROJECT_COMPLETION_SUMMARY.md         ← DONE
```

---

## ESTIMATED TIME ON CHPC

| Task | Estimated Time |
|---|---|
| Module load + build all | 5–10 min |
| Study 1: Serial vs OpenMP (5 configs × 3 runs) | 15–20 min |
| Study 2: CUDA block sizes (4 configs × 3 runs) | 10–15 min |
| Study 3: MPI CPU (4 configs × 3 runs) | 15–20 min |
| Study 4: MPI+CUDA (4 configs × 3 runs) | 15–20 min |
| Validation (5 implementations) | 5–10 min |
| Updating result files | 10–15 min |
| **Total** | **~75–90 min** |

**Queue wait time on Kingspeak is unpredictable — submit jobs early!**

---

*Instructions created: April 19, 2026*
*For questions, see README.md and docs/Implementations.md*
