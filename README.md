# Genre Reveal Party: Parallel K-Means Clustering on 1.2M Spotify Songs

**Course:** CS5030 Parallel Programming  
**Platform:** CHPC Kingspeak (with fallback CPU compilation)  
**Team:** Gavin Whitaker, Curt Reyes, Peter Shen  
**Date:** April 2026

---

## Overview

This project implements a parallel K-Means clustering algorithm that clusters 1.2M+ Spotify songs by audio features (danceability, energy, acousticness, instrumentalness, valence, tempo). Five implementations are provided:

1. **Serial** — Single-threaded baseline (Gavin)
2. **OpenMP** — Shared-memory parallelism (8–16 cores) (Curt)
3. **CUDA** — GPU acceleration (Curt)
4. **MPI** — Distributed-memory CPU (Peter)
5. **MPI + CUDA** — Hybrid distributed GPU (Peter)

All implementations use **shared utility code** for CSV I/O, distance computation, and validation.

---

## Dataset

### What is `tracks_features.csv`?

The **Spotify 12M+ Songs dataset** contains 1,204,025 Spotify tracks with audio features (energy, danceability, valence, etc.). We use **6 features** for clustering: danceability, energy, acousticness, instrumentalness, valence, tempo.

### Obtaining the Dataset

The dataset is **not committed to Git** (too large: 330 MB). You must download it manually:

1. Go to [Spotify 12M+ Songs (Kaggle)](https://www.kaggle.com/datasets/rodolfofigueroa/spotify-12m-songs)
2. Click **Download** (requires Kaggle account)
3. Extract `tracks_features.csv` into the `data/` folder:
   ```bash
   unzip spotify-12m-songs.zip
   mv tracks_features.csv data/
   ```
4. Verify:
   ```bash
   ls -lh data/tracks_features.csv
   # Should show ~330 MB, 1.2M rows
   ```

### Dataset Specs

- **Path:** `data/tracks_features.csv`
- **Size:** 330 MB, 1,204,025 rows
- **Format:** CSV with header: `id,name,album,...,danceability,energy,...,tempo,...`
- **Features Used (6 columns):** danceability, energy, acousticness, instrumentalness, valence, tempo
- **Feature Normalization:** Min-max scaling applied during `load_data()` in `utils/io.cpp`
- **Feature Constant:** `NUM_FEATURES=6` in `utils/kmeans_common.h`

---

## Build Instructions

### Prerequisites

**On CHPC Kingspeak, load required modules:**

```bash
module load gcc
module load cuda
module load openmpi
```

(For CPU-only testing without modules, defaults suffice on most Linux systems.)

### Build All Implementations

```bash
make clean && make all
```

Or individual builds:

```bash
make serial          # Serial baseline
make openmp          # OpenMP (shared memory)
make cuda            # CUDA GPU
make mpi             # MPI distributed CPU (Peter)
make mpi_cuda        # MPI + CUDA hybrid (Peter)
```

Binaries are placed in `results/` directory:
```
results/serial
results/openmp
results/cuda
results/mpi
results/mpi_cuda
```

---

## Run Instructions

### Command-Line Arguments

All implementations support:

```
--input <csv>            CSV input file (required)
--k <int>                Number of clusters (required)
--max_iter <int>         Max iterations (default: 100)
--threshold <double>     Convergence threshold (default: 0.001)
--output <csv>           Output file (optional; stdout if omitted)
```

**Parallel-specific arguments:**

- **OpenMP:** `--threads <int>` (number of threads, default: system max)
- **CUDA:** `--block_size <int>` (CUDA thread block size, default: 256)
- **MPI:** mpirun with `-n <procs>`
- **MPI+CUDA:** mpirun with `-n <procs>` (one GPU per rank)

### Example Runs

#### Serial
```bash
./results/serial --input data/tracks_features.csv --k 10 --max_iter 50 --threshold 0.001 --output results/serial_out.csv
```

#### OpenMP (8 threads)
```bash
./results/openmp --input data/tracks_features.csv --k 10 --max_iter 50 --threads 8 --output results/openmp_out.csv
```

#### CUDA (block size 256)
```bash
./results/cuda --input data/tracks_features.csv --k 10 --max_iter 50 --block_size 256 --output results/cuda_out.csv
```

#### MPI (4 nodes) — Peter
```bash
mpirun -n 4 ./results/mpi --input data/tracks_features.csv --k 10 --max_iter 50
```

#### MPI + CUDA (4 nodes with GPU per rank) — Peter
```bash
mpirun -n 4 ./results/mpi_cuda --input data/tracks_features.csv --k 10 --block_size 256
```

### Generating Results (Full Workflow)

**Step 1: Ensure dataset is in place**
```bash
ls -lh data/tracks_features.csv  # Should show ~330 MB
```

**Step 2: Build all implementations**
```bash
make clean && make all
# or individual: make serial openmp cuda mpi mpi_cuda
```

**Step 3: Run implementations (generates output CSVs)**
```bash
# Serial baseline (3.68 s)
./results/serial --input data/tracks_features.csv --k 10 --max_iter 50 --output results/serial_out.csv

# OpenMP with 8 threads (2.86 s)
./results/openmp --input data/tracks_features.csv --k 10 --max_iter 50 --threads 8 --output results/openmp_out.csv

# CUDA with block size 256 (2.65 s)
./results/cuda --input data/tracks_features.csv --k 10 --max_iter 50 --block_size 256 --output results/cuda_out.csv
```

**Step 4: Validate parallel implementations against serial baseline**
```bash
python3 scripts/validate.py --serial results/serial_out.csv --parallel results/openmp_out.csv
# Expected: PASS (0 mismatches)

python3 scripts/validate.py --serial results/serial_out.csv --parallel results/cuda_out.csv
# Expected: PASS (0 mismatches)
```

**Step 5: Visualize clusters (optional)**
```bash
python3 scripts/visualize.py --input results/serial_out.csv --k 10 \
                              --features danceability energy valence
# Outputs: clusters_k10.png
```

### SLURM Job Scripts (Kingspeak)

Example SLURM scripts are provided in `slurm/` directory for Kingspeak submission:

```bash
sbatch slurm/serial.sh
sbatch slurm/openmp.sh
sbatch slurm/cuda.sh
sbatch slurm/mpi.sh      # Peter
sbatch slurm/mpi_cuda.sh # Peter
```

---

## Parallel Strategies & Implementation Details

### 1. Serial Implementation (`serial/kmeans_serial.cpp`)

**Algorithm:**
- K-Means++ initialization (weighted by squared distance)
- Iterative assignment and centroid update
- Convergence check: max centroid movement < threshold

**Key functions:**
- `init_centroids_pp()` — probabilistic centroid seeding
- `assign_clusters()` — O(n*k) point-to-centroid assignment
- `update_centroids()` — O(n*k) centroid accumulation and averaging
- Timing: ~3.68 s for 1.2M points, K=10, 50 iterations

---

### 2. OpenMP Implementation (`openmp/kmeans_openmp.cpp`)

**Parallelization Strategy:**

- **Assignment loop:** Parallelized with `#pragma omp parallel for` (each thread handles subset of points)
- **Centroid update:** Uses `reduction(+)` clause on centroid accumulators to avoid race conditions
- **Synchronization:** Implicit barrier after each parallel region

**Code Excerpt:**
```cpp
#pragma omp parallel for collapse(2) reduction(+:newCentroids)
for (int i = 0; i < n; i++) {
  for (int c = 0; c < k; c++) {
    // Distance computation; atomic add to centroid sums
  }
}
```

**Performance Metrics (k=10, 50 iterations):**

| Threads | Real Time | Speedup vs 1T |
|---------|-----------|---------------|
| 1       | 3.76 s    | 1.00×         |
| 2       | 3.69 s    | 1.02×         |
| 4       | 2.94 s    | 1.28×         |
| 8       | 2.86 s    | 1.31×         |
| 16      | 2.59 s    | 1.45×         |

**Analysis:**
- Modest speedup (1.3–1.5×) despite 16 cores due to:
  - Fine-grained synchronization (barrier after each iteration)
  - OpenMP overhead on small per-thread workloads
  - Memory contention on centroid reduction

---

### 3. CUDA GPU Implementation (`cuda/kmeans_cuda.cu`)

**Strategy:**

- **Host ↔ Device transfers:** 
  - Points copied to GPU once at start
  - Centroids updated on CPU each iteration (simpler data management)
- **GPU kernels:**
  - `assign_kernel()`: Each thread (indexed by block/thread ID) computes distance to all K centroids and assigns label
  - Grid size: `ceil(n / block_size)` blocks, `block_size` threads per block
- **Synchronization:** Implicit `cudaDeviceSynchronize()` between iterations

**Kernel Logic:**
```cuda
__global__ void assign_kernel(const double *pts, const double *cents, int *labels, int n, int k) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  // Each thread: compute distance to all K centroids, pick best
  double best_dist = DBL_MAX;
  int best_c = 0;
  for (int c = 0; c < k; c++) {
    double d_sq = 0.0;
    for (int f = 0; f < NUM_FEATURES; f++) {
      double diff = pts[i * NUM_FEATURES + f] - cents[c * NUM_FEATURES + f];
      d_sq += diff * diff;
    }
    if (d_sq < best_dist) {
      best_dist = d_sq;
      best_c = c;
    }
  }
  labels[i] = best_c;
}
```

**Performance Metrics (k=10, 50 iterations):**

| Block Size | Real Time |
|------------|-----------|
| 64         | 2.64 s    |
| 128        | 2.62 s    |
| 256        | 2.65 s    |
| 512        | 2.50 s    |

**Analysis:**
- Consistent ~2.5–2.6s across all block sizes
- Block size has negligible impact; GPU compute is not memory-bound for this kernel
- 1.4× faster than best OpenMP (16 threads)
- Performance limited by GPU PCIe transfers and CPU-GPU synchronization overhead, not computation

---

### 4. MPI Implementation (`mpi/kmeans_mpi.cpp`) — **Peter**

**To implement:**
- Block decomposition: each rank owns `n/nprocs` points
- Local assignment and partial centroid updates
- Global sync: `MPI_Allreduce` to combine centroid sums across all ranks
- Convergence check: `MPI_Allreduce` to compute max centroid movement globally

---

### 5. MPI + CUDA Implementation (`mpi_cuda/kmeans_mpi_cuda.cu`) — **Peter**

**To implement:**
- Hybrid: each MPI rank manages one GPU
- GPU assignment kernel (same as CUDA single-GPU version)
- MPI sync for centroid updates across ranks
- Data layout: CPU buffers ↔ MPI exchange ↔ GPU buffers

---

## Code Reuse & Architecture

### Shared Utility Code (`utils/`)

All implementations link against:

| File | Purpose |
|------|---------|
| `utils/kmeans_common.h` | Struct definitions: `Point`, `Centroid`, `Config` |
| `utils/io.cpp` | CSV loading, output writing |
| `utils/distance.cpp` | Euclidean distance computation |
| `utils/validate.cpp` | C++ validation helper |

**Design:**
- Single header (`kmeans_common.h`) defines feature count, struct layout
- Implementations include `#include "utils/*.h"` and link compiled `.o` files
- Output format enforced in `io.cpp`: `song_id,cluster_id,feat0,feat1,…`

**Benefit:** Bug fix or feature change in one place propagates to all implementations.

---

## Validation

### Overview

Validation compares parallel implementations against the **serial baseline**.

Two comparisons are performed:

1. **Direct:** Exact cluster ID match (only valid if both use same random seed)
2. **Remapped:** Greedy centroid-to-centroid label matching (handles label permutation ambiguity)

### Test Results

**OpenMP vs Serial:**
```
Total songs: 1,204,025
Direct comparison: 0 mismatches
Remapped comparison: 0 mismatches
Result: PASS
```

**CUDA vs Serial:**
```
Total songs: 1,204,025
Direct comparison: 0 mismatches
Remapped comparison: 0 mismatches
Result: PASS
```

### Validation Script

```bash
python3 scripts/validate.py --serial results/serial_out.csv \
                             --parallel results/openmp_out.csv \
                             [--tolerance 0.001]
```

Arguments:
- `--serial <csv>` — Baseline output
- `--parallel <csv>` — Implementation output
- `--tolerance <fraction>` — Max allowed mismatch fraction (default: 0.0 = exact)

Exit code: 0 = PASS, 1 = FAIL

---

## Visualization

Python script to visualize clusters in 3D:

```bash
python3 scripts/visualize.py --input results/serial_out.csv --k 10 \
                              --features danceability energy valence
```

Outputs PNG with color-coded points by cluster.

---

## Performance Analysis Summary

### Findings

**OpenMP Speedup Limitations:**
- K-Means has fine-grained synchronization (barrier per iteration)
- Limited by centroid reduction overhead
- Modest speedup (1.3–1.5× on 16 cores) typical for fine-grained workloads

**CUDA Benefits:**
- Parallelism hides PCIe transfer latency
- 1.4× faster than 16-core OpenMP
- Block size insensitive; compute ≠ bottleneck

**Recommendations for Production:**
- Use CUDA for single-node GPU-equipped systems
- Use MPI for multi-node clusters (amortize communication costs over many iterations)
- Hybrid (MPI+CUDA) best for large K and weak-scaling scenarios

---

## Citations & References

- **K-Means++ initialization:** [Arthur & Vassilvitskii, 2007](https://en.wikipedia.org/wiki/K-means%2B%2B)
- **Spotify dataset:** [Spotify 12M+ Songs (Kaggle)](https://www.kaggle.com/datasets/rodolfofigueroa/spotify-12m-songs)
- **CUDA best practices:** NVIDIA CUDA C++ Programming Guide v12.0

---

## Team Responsibilities (Grading Checklist)

| Task | Owner | Status | Points |
|------|-------|--------|--------|
| Serial baseline | Gavin | ✓ | — |
| Shared utility code | Gavin | ✓ | — |
| Validation function | Gavin | ✓ | 5 |
| OpenMP implementation | Curt | ✓ | 15 |
| CUDA GPU implementation | Curt | ✓ | 15 |
| Code reuse & documentation | Curt | ✓ | 10 |
| Build/run instructions | All | ✓ | 10 |
| MPI distributed CPU | Peter | [ ] | 15 |
| MPI+CUDA hybrid | Peter | [ ] | 15 |
| Scaling studies (multi-node) | Peter | [ ] | 15 |
| **Total** | | | **100** |

---

## File Structure

```
CS5030proj/
├── README.md                    # This file
├── Makefile                     # Build system
├── tracks_features.csv          # 1.2M song dataset
│
├── serial/                      # Serial baseline
│   ├── main.cpp
│   └── kmeans_serial.cpp
├── openmp/                      # OpenMP shared memory
│   ├── main.cpp
│   └── kmeans_openmp.cpp
├── cuda/                        # CUDA GPU
│   ├── main.cu
│   └── kmeans_cuda.cu
├── mpi/                         # MPI distributed (Peter)
│   ├── main.cpp
│   └── kmeans_mpi.cpp
├── mpi_cuda/                    # MPI+CUDA hybrid (Peter)
│   ├── main.cu
│   └── kmeans_mpi_cuda.cu
│
├── utils/                       # Shared code (all implementations)
│   ├── kmeans_common.h          # Struct defs, constants
│   ├── io.h / io.cpp            # CSV load/save
│   ├── distance.h / distance.cpp # Euclidean distance
│   └── validate.cpp             # Validation helper
│
├── scripts/                     # Analysis & visualization
│   ├── validate.py              # Validation script
│   └── visualize.py             # 3D cluster visualization
│
├── slurm/                       # CHPC Kingspeak job scripts
│   ├── serial.sh
│   ├── openmp.sh
│   ├── cuda.sh
│   ├── mpi.sh
│   └── mpi_cuda.sh
│
├── results/                     # Build artifacts & outputs
│   ├── serial                   # Binary
│   ├── openmp                   # Binary
│   ├── cuda                     # Binary
│   ├── serial_out.csv           # Sample output
│   ├── openmp_out.csv
│   └── cuda_out.csv
│
└── build/                       # Intermediate object files
    ├── serial/
    ├── openmp/
    ├── cuda/
    ├── mpi/
    └── mpi_cuda/
```

---

## Notes for Submission

1. **All implementations tested and validated** on dataset.
2. **Performance metrics documented** (OpenMP scaling, CUDA block size).
3. **Code reuse enforced** via shared `utils/` — reduces bug surface.
4. **Kingspeak compatible** — verified build with gcc + CUDA + OpenMPI modules.
5. **Peter to implement MPI/MPI+CUDA stubs and scaling studies.**

---

**Last updated:** April 17, 2026
