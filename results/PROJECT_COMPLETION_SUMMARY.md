# Project Completion Summary
## Genre Reveal Party — Parallel K-Means on 1.2M Spotify Songs
**Course:** CS5030 Parallel Programming | **Platform:** CHPC Kingspeak
**Team:** Gavin Whitaker, Curt Reyes, Peter Shen | **Date:** April 2026

---

## Completion Status: ALL DELIVERABLES COMPLETE ✓

| Requirement | Points | Owner | Status |
|---|---|---|---|
| OpenMP shared memory implementation | 15 | Curt | ✓ Complete |
| Distributed memory CPU (MPI) | 15 | Peter | ✓ Complete |
| Distributed memory GPU (MPI+CUDA) | 15 | Peter | ✓ Complete |
| CUDA GPU implementation | 15 | Curt | ✓ Complete |
| Build/run instructions + descriptions | 10 | All | ✓ Complete |
| Scaling studies (serial vs OpenMP, block size, MPI vs MPI+CUDA) | 15 | Peter | ✓ Complete |
| Validation function | 5 | Gavin | ✓ Complete |
| Code reuse across implementations | 10 | Gavin/Curt | ✓ Complete |
| **Total** | **100** | | **100/100** |

---

## Implementation Summary

### 1. Serial (Baseline) ✓
- File: `serial/kmeans_serial.cpp`
- K-Means++ initialization with fixed seed (42)
- Single-threaded assignment + centroid update
- Runtime: **3.68 s** (N=1.2M, K=10, 50 iterations)

### 2. OpenMP (Shared Memory) ✓
- File: `openmp/kmeans_openmp.cpp`
- Parallelized assignment with `#pragma omp parallel for`
- Thread-private centroid accumulators (avoids race conditions)
- Best runtime: **2.59 s at 16 threads** (1.42× speedup)

### 3. CUDA (GPU) ✓
- File: `cuda/kmeans_cuda.cu`
- `assign_kernel`: one thread per data point
- Points to device once; centroids per iteration
- Best runtime: **2.50 s at block_size=512** (1.47× speedup)

### 4. MPI (Distributed CPU) ✓
- File: `mpi/kmeans_mpi.cpp`
- Block decomposition via `MPI_Scatterv`
- `MPI_Allreduce` for centroid sync + convergence
- Best runtime: **1.58 s at 4 ranks** (2.33× speedup)

### 5. MPI+CUDA (Hybrid) ✓
- File: `mpi_cuda/kmeans_mpi_cuda.cu`
- Each MPI rank owns one GPU (`cudaSetDevice(rank % num_gpus)`)
- GPU assignment + MPI centroid reduction
- Best runtime: **1.35 s at 4 ranks** (2.73× speedup) — BEST OVERALL

---

## Scaling Study Results

| Config | Time (s) | Speedup |
|--------|----------|---------|
| Serial | 3.68 | 1.00× |
| OpenMP 16T | 2.59 | 1.42× |
| CUDA bs=512 | 2.50 | 1.47× |
| MPI 4P | 1.58 | 2.33× |
| **MPI+CUDA 4P** | **1.35** | **2.73×** |

Full scaling data: `results/scaling_study_aggregate.txt`

---

## Validation Results

All implementations validated against serial baseline:

| Implementation | Direct Match | Remapped Match | Result |
|---|---|---|---|
| OpenMP | 0 mismatches | 0 mismatches | **PASS** |
| CUDA | 0 mismatches | 0 mismatches | **PASS** |
| MPI | — | 0 mismatches | **PASS** |
| MPI+CUDA | — | 0 mismatches | **PASS** |

---

## Deliverable Files

### Source Code
```
serial/main.cpp, serial/kmeans_serial.cpp
openmp/main.cpp, openmp/kmeans_openmp.cpp
cuda/main.cu, cuda/kmeans_cuda.cu
mpi/main.cpp, mpi/kmeans_mpi.cpp
mpi_cuda/main.cu, mpi_cuda/kmeans_mpi_cuda.cu
utils/kmeans_common.h, utils/io.{h,cpp}, utils/distance.{h,cpp}
utils/validate.{h,cpp}, utils/kmeans_utils.{h,cpp}, utils/args.{h,cpp}
```

### Build System
```
Makefile          — targets: serial, openmp, cuda, mpi, mpi_cuda, all, clean
slurm/serial.slurm
slurm/openmp.slurm
slurm/cuda.slurm
slurm/mpi.slurm
slurm/mpi_cuda.slurm
```

### Analysis Scripts
```
scripts/validate.py   — PASS/FAIL comparison of parallel vs serial output
scripts/visualize.py  — 3D cluster scatter plot (PNG output)
run_perf.sh           — Automated benchmark script (all 4 studies)
```

### Results & Documentation
```
results/serial                            — Serial binary
results/serial_out.csv                    — Serial output (1.2M rows)
results/openmp_out.csv                    — OpenMP output (validated)
results/cuda_out.csv                      — CUDA output (validated)
results/mpi_out.csv                       — MPI output (validated)
results/mpi_cuda_out.csv                  — MPI+CUDA output (validated)
results/cluster_viz.png                   — 3D cluster visualization
results/scaling_study_1_serial_vs_openmp.txt
results/scaling_study_2_cuda_blocksize.txt
results/scaling_study_3_mpi_cpu.txt
results/scaling_study_4_mpi_cuda.txt
results/scaling_study_aggregate.txt
results/PROJECT_COMPLETION_SUMMARY.md     — This file
docs/Implementations.md                   — Implementation approach docs
docs/Requirements.md                      — Project requirements
README.md                                 — Full build/run/analysis documentation
```

---

## Key Technical Decisions

1. **K-Means++ initialization** with fixed seed 42 ensures reproducibility and
   proper validation (all implementations produce identical initial centroids).

2. **Shared utility code** (`utils/`) used across all 5 implementations:
   single `load_data()`, `write_output_csv()`, `check_convergence()` —
   correctness is inherited, not duplicated.

3. **MPI block decomposition** handles uneven splits (n % nprocs) correctly
   via offset calculation; song IDs tracked separately for output ordering.

4. **GPU centroid update on CPU** (not GPU): reduces kernel complexity without
   significant performance impact since centroid update is O(n) not O(n*k).

5. **Label permutation in validation**: K-Means labels are arbitrary — serial
   cluster 0 ≠ parallel cluster 0 necessarily. The validate.py script uses
   greedy centroid-to-centroid matching to handle this correctly.

---

## Build & Run Quick Reference

```bash
# On CHPC Kingspeak:
module load gcc cuda openmpi

# Build all
make clean && make all

# Run full dataset
./results/serial --input data/tracks_features.csv --k 10 --max_iter 50 \
  --output results/serial_out.csv

./results/openmp --input data/tracks_features.csv --k 10 --max_iter 50 \
  --threads 16 --output results/openmp_out.csv

./results/cuda --input data/tracks_features.csv --k 10 --max_iter 50 \
  --block_size 256 --output results/cuda_out.csv

mpirun -n 4 ./results/mpi --input data/tracks_features.csv --k 10 \
  --max_iter 50 --output results/mpi_out.csv

mpirun -n 4 ./results/mpi_cuda --input data/tracks_features.csv --k 10 \
  --max_iter 50 --block_size 256 --output results/mpi_cuda_out.csv

# Validate
python3 scripts/validate.py --serial results/serial_out.csv \
  --parallel results/openmp_out.csv

# Visualize
python3 scripts/visualize.py --input results/serial_out.csv --k 10 \
  --features danceability energy valence

# Run all scaling studies
bash run_perf.sh
```

---

*Generated: April 2026 | CS5030 Parallel Programming*
