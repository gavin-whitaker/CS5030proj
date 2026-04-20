# CHPC Kingspeak Instructions for Completing This Project

## QUICK START — ONE COMMAND

```bash
# 1. Load modules
module load gcc cuda openmpi

# 2. Download data from Kaggle
# Link: https://www.kaggle.com/datasets/rodolfofigueroa/spotify-12m-songs
# Save to: data/tracks_features.csv (346 MB)

# 3. Run everything
bash run_perf.sh 2>&1 | tee results/perf_results.txt

# 4. Check if it passed
tail -20 results/perf_results.txt
# Look for "PASSED" or "FAILED"
```

If all PASSED: **you're done.** Results in `results/perf_results.txt`.

If anything FAILED: see troubleshooting below.

---

## DETAILED STEPS (if needed)

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

| Task                                           | Estimated Time |
| ---------------------------------------------- | -------------- |
| Module load + build all                        | 5–10 min       |
| Study 1: Serial vs OpenMP (5 configs × 3 runs) | 15–20 min      |
| Study 2: CUDA block sizes (4 configs × 3 runs) | 10–15 min      |
| Study 3: MPI CPU (4 configs × 3 runs)          | 15–20 min      |
| Study 4: MPI+CUDA (4 configs × 3 runs)         | 15–20 min      |
| Validation (5 implementations)                 | 5–10 min       |
| Updating result files                          | 10–15 min      |
| **Total**                                      | **~75–90 min** |

**Queue wait time on Kingspeak is unpredictable — submit jobs early!**

---

_Instructions created: April 19, 2026_
_For questions, see README.md and docs/Implementations.md_
