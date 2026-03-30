# Genre Reveal Party — Peter's Tasks (33%)
**Parallel K-Means Clustering on 1.2M Spotify Songs**  
**Course:** Parallel Programming | **Platform:** CHPC Kingspeak

---

## Peter's Responsibilities at a Glance

| Task | Share |
|---|---|
| MPI distributed-memory CPU implementation | 1/3 |
| MPI + CUDA distributed-memory GPU implementation | 1/3 |
| Scaling studies (distributed runs) | 1/3 |
| **Peter's total** | **~33%** |

---

## 1) MPI Distributed Memory CPU
**Owner: Peter | Builds on serial + shared utils**

Implement K-Means with MPI across nodes:
- split data with `MPI_Scatter`/`MPI_Scatterv`
- assign clusters locally per process
- combine centroid sums with `MPI_Allreduce`
- check convergence globally

Example:
```bash
mpirun -n 4 ./mpi --input data/songs.csv --k 10 --max_iter 100
```

---

## 2) MPI + CUDA Distributed Memory GPU
**Owner: Peter | Builds on MPI CPU + CUDA baseline**

Each process uses one GPU for assignment, then synchronizes centroids with MPI:
- copy local chunk and centroids to GPU
- run assignment kernel
- copy assignments back
- combine partial centroid results with `MPI_Allreduce`

Example:
```bash
mpirun -n 4 ./mpi_cuda --input data/songs.csv --k 10 --block_size 256
```

---

## 3) Distributed Scaling Studies
**Owner: Peter**

Run and report scaling for distributed versions:
- node counts: 2, 3, 4 Kingspeak nodes
- compare MPI CPU vs MPI+CUDA
- collect runtime, speedup, efficiency
- run each setup at least 3 times and average

---

## Build Targets Peter Maintains
```bash
make mpi
make mpi_cuda
```

---

## Shared Output Format
```text
song_id, cluster_id, feature_1, feature_2, feature_3, ...
```

One row per song, same format for all implementations.
