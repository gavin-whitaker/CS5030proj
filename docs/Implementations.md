# K-Means Implementation Approaches

## 1. Serial (CPU Single-Thread)
**File:** `serial/kmeans_serial.cpp`

**Approach:**
- Pure C++ single-threaded execution on CPU
- Simple loop-based assignment: for each point, find nearest centroid
- Serial centroid update: accumulate sums, divide by counts
- K-Means++ initialization with fixed seed (42) for reproducibility
- No parallelization overhead; baseline for speedup measurement

**Key functions:** `assign_clusters()`, `update_centroids()` (utils)

---

## 2. Parallel Shared Memory (OpenMP)
**File:** `openmp/kmeans_openmp.cpp`

**Approach:**
- Shared memory parallelization using OpenMP on multi-core CPU
- **Assignment:** `#pragma omp parallel for schedule(static)` — each thread processes chunk of points independently
- **Centroid update:** Thread-private accumulators to avoid false sharing
  - Each thread maintains local sums/counts for all centroids
  - Serial merge phase combines thread results
  - Final normalization (division) is serial
- K-Means++ init remains serial (random sampling, not parallelizable)
- Scales with thread count up to physical core count

**Key optimization:** Thread-private sums avoid synchronization overhead

---

## 3. Parallel GPU (CUDA)
**File:** `cuda/kmeans_cuda.cu`

**Approach:**
- GPU-accelerated assignment phase; CPU-based centroid update
- **Assignment kernel:** `assign_kernel<<<grid, block>>>` launches one thread per point
  - Each thread finds nearest centroid via sequential search
  - Coalesced writes to global labels array
  - No shared memory required (simple implementation)
- **Centroid update:** CPU-side (no GPU optimization needed; I/O bound dominates)
- **Data flow:** Points copied once to GPU; centroids copied each iteration
- Block size configurable (128–1024) to tune occupancy/latency tradeoff

**Scalability:** Faster on large datasets; overhead dominates on small data

---

## 4. Distributed Memory CPU (MPI)
**File:** `mpi/kmeans_mpi.cpp`

**Status:** Complete ✓

**Approach:**
- Each MPI rank processes subset of points (1D block decomposition)
- Rank 0 loads all data, flattens to double array, scatters via `MPI_Scatterv`
- Song IDs also scattered to preserve output ordering during `MPI_Gatherv`
- Assignment: Each rank assigns local points independently (no communication needed)
- Centroid update: Local partial sums → `MPI_Allreduce` (SUM) for global centroid update
  - All ranks compute new centroids identically; no extra broadcast needed
- Convergence: `MPI_Allreduce` with `MPI_LAND` to synchronize convergence decision
- Output: `MPI_Gatherv` collects all labels at rank 0, rank 0 writes CSV

**Key MPI calls:** `MPI_Scatterv`, `MPI_Bcast`, `MPI_Allreduce` (×2), `MPI_Gatherv`

**Performance (K=10, 50 iterations, 1.2M songs):**

| Processes | Real Time | Speedup vs Serial |
|-----------|-----------|-------------------|
| 1         | 2.37 s    | 1.21×             |
| 2         | 1.95 s    | 1.47×             |
| 4         | 1.58 s    | 1.82×             |

---

## 5. Distributed Memory GPU (MPI + CUDA)
**File:** `mpi_cuda/kmeans_mpi_cuda.cu`

**Status:** Complete ✓

**Approach:**
- Hybrid: each MPI rank owns one GPU (`cudaSetDevice(rank % num_gpus)`)
- Each rank receives `n/nprocs` points via `MPI_Scatterv` (identical to MPI version)
- Data flow per iteration:
  1. `MPI_Bcast` centroids to all ranks
  2. Copy centroids host→device (`cudaMemcpy`)
  3. Launch `assign_kernel` — one thread per local point
  4. Copy labels device→host
  5. CPU accumulates partial sums/counts
  6. `MPI_Allreduce` for global centroid update
  7. `MPI_Allreduce` with `MPI_LAND` for convergence sync
- Points copied to GPU once before iteration loop; only centroids and labels transfer each iteration
- Output: `MPI_Gatherv` at rank 0, rank 0 writes CSV

**Performance (K=10, 50 iterations, 1.2M songs):**

| Processes | Real Time | Speedup vs Serial |
|-----------|-----------|-------------------|
| 1         | 1.61 s    | 1.78×             |
| 2         | 1.42 s    | 2.02×             |
| 4         | 1.35 s    | 2.13×             |

**Analysis:** GPU assignment dominates per-iteration cost; MPI centroid communication (small: K×NUM_FEATURES doubles) is negligible.

---

## Summary Table

| Implementation | Parallelism | Memory Model | Scaling Factor | Implemented |
|---|---|---|---|---|
| Serial | None | Single-thread CPU | Baseline | ✓ |
| OpenMP | Multi-core | Shared (on-node) | # threads | ✓ |
| CUDA | GPU | Device memory | Block size tuning | ✓ |
| MPI | Multi-node CPU | Distributed | # ranks | ✓ |
| MPI+CUDA | Multi-node GPU | Hybrid | # ranks × GPUs | ✓ |

