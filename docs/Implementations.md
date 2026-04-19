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

**Status:** Stub (TODO)

**Intended approach:**
- Each MPI rank processes subset of points (1D data decomposition)
- Assignment: Local + global centroid broadcast (MPI_Bcast)
- Centroid update: Local accumulation → global reduction (MPI_Allreduce)
- Requires synchronization at each iteration; communication overhead at rank boundary
- Scales with cluster size if computation >> communication

---

## 5. Distributed Memory GPU (MPI + CUDA)
**File:** `mpi_cuda/kmeans_mpi_cuda.cu`

**Status:** Stub (TODO)

**Intended approach:**
- Hybrid: each MPI rank owns GPU(s)
- Each rank runs CUDA assignment kernel on its data partition
- CPU side: MPI collective operations (broadcast centroids, allreduce counts/sums)
- Data movement: Host ↔ Device within rank; Host ↔ Host across ranks
- Maximizes GPU utilization while distributing data across multiple GPUs

**Challenge:** PCIe bandwidth + MPI latency can dominate for small K

---

## Summary Table

| Implementation | Parallelism | Memory Model | Scaling Factor | Implemented |
|---|---|---|---|---|
| Serial | None | Single-thread CPU | Baseline | ✓ |
| OpenMP | Multi-core | Shared (on-node) | # threads | ✓ |
| CUDA | GPU | Device memory | Block size tuning | ✓ |
| MPI | Multi-node CPU | Distributed | # ranks | ✗ Stub |
| MPI+CUDA | Multi-node GPU | Hybrid | # ranks × GPUs | ✗ Stub |

