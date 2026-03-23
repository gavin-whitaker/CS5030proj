# Genre Reveal Party — Peter's Tasks (40%)
**Parallel K-Means Clustering on 1.2M Spotify Songs**  
**Course:** Parallel Programming | **Platform:** CHPC Kingspeak  

---

## Peter's Responsibilities at a Glance

| Task | Points |
|---|---|
| MPI distributed memory CPU | 15 pts |
| MPI + CUDA distributed memory GPU | 15 pts |
| Scaling studies on Kingspeak | 15 pts |
| Output visualization script | — |
| Build/run instructions + descriptions (shared) | 5 pts |
| **Peter's total** | **~50/100** |

> Peter's work builds on top of the serial + CUDA foundation. Wait for the serial implementation and shared utils to be ready before starting MPI — they are the building blocks.

---

## What Peter Needs From Me First

Before starting, make sure these are ready and tested:

- `utils/io.h` — CSV loading and output writing
- `utils/distance.h` — `compute_distance()` function
- `utils/kmeans_common.h` — shared structs (Point, Centroid, Config)
- Serial implementation passing and producing correct output CSV
- CUDA implementation passing validation (needed for MPI+CUDA)

---

## Step 1 — MPI Distributed Memory CPU (15 pts)
**Owner: Peter | Builds on: serial + shared utils**

Divide the 1.2M songs across MPI processes. Each process handles its chunk independently, then they sync centroids every iteration.

**How to approach it:**
- Process 0 loads the full dataset, then scatters chunks to all processes using `MPI_Scatter` or `MPI_Scatterv`
- Each iteration:
  1. Every process runs `assign_clusters()` on its local chunk
  2. Every process computes partial centroid sums locally
  3. Use `MPI_Allreduce` to sum centroid contributions across all processes
  4. All processes update centroids with the global result
  5. Check convergence globally with `MPI_Allreduce`
- All processes write their chunk to the output, or process 0 gathers and writes

**Run it like this:**
```bash
mpirun -n 4 ./mpi --input data/songs.csv --k 10 --max_iter 100
```

**What to document:**
- How the data is split across processes (block decomposition)
- Which MPI calls are used and why (`MPI_Scatter`, `MPI_Allreduce`, etc.)
- How convergence is checked globally

**Experiments:** Run on 2, 3, and 4 Kingspeak nodes — results go into scaling study.

**Done when:** passes validation against serial output.

---

## Step 2 — MPI + CUDA Distributed Memory GPU (15 pts)
**Owner: Peter | Builds on: MPI CPU version + my CUDA version**

Each MPI process gets one GPU. The GPU handles the assignment step, MPI handles syncing centroids across nodes.

**How to approach it:**
- Same data distribution as MPI CPU version
- Each iteration:
  1. Copy local song chunk to GPU (do this once before the loop)
  2. Copy centroids to GPU
  3. Launch CUDA kernel for assignment (reuse my kernel)
  4. Copy assignments back to CPU
  5. Compute partial centroid sums on CPU
  6. `MPI_Allreduce` to sync centroids across nodes
  7. Check convergence globally

**Run it like this:**
```bash
mpirun -n 4 ./mpi_cuda --input data/songs.csv --k 10 --block_size 256
```

**What to document:**
- How MPI and CUDA interact
- Data flow: CSV → CPU → GPU → CPU → MPI → repeat
- Any bottlenecks (e.g., CPU↔GPU transfer vs MPI communication)

**Experiments:** Run on 2, 3, and 4 Kingspeak nodes — results go into scaling study.

**Done when:** passes validation against serial output.

---

## Step 3 — Scaling Studies (15 pts)
**Owner: Peter | Submit jobs early — queues can be slow**

Three studies total. Run each configuration at least 3 times and average the results.

### Study 1: Serial vs OpenMP (my implementations)
- Fix K, vary thread count: 1, 2, 4, 8, 16
- Report: runtime, speedup, efficiency
- Peter runs these experiments and records results even though I wrote the code

### Study 2: CUDA block size
- Fix K, vary block size: 64, 128, 256, 512
- Report: runtime per iteration, total runtime

### Study 3: MPI CPU vs MPI+CUDA
- Fix K, vary node count: 2, 3, and 4 Kingspeak nodes
- Report: runtime, speedup, efficiency

**How to run on Kingspeak (SLURM example):**
```bash
#!/bin/bash
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --time=01:00:00
#SBATCH --partition=kingspeak-gpu

module load gcc cuda openmpi
mpirun -n 4 ./mpi_cuda --input data/songs.csv --k 10 --block_size 256
```

> Submit jobs at the start of week 3, not the end. Queue wait times on Kingspeak can eat up a full day.

---

## Step 4 — Output Visualization (no parallelization needed)
**Owner: Peter**

Write a Python script using matplotlib that shows the clustering results.

**What it should do:**
- Load the output CSV
- Plot songs in 3D using 3 audio features (e.g., valence, danceability, energy)
- Color each point by its cluster assignment
- Work with any K > 2
- Save the plot as a PNG

```bash
python visualize.py --input output/serial_out.csv --k 10 --features valence danceability energy
```

**Simple starting point:**
```python
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

df = pd.read_csv("output/serial_out.csv")
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(df['valence'], df['danceability'], df['energy'],
                     c=df['cluster_id'], cmap='tab20', s=0.5, alpha=0.5)
plt.colorbar(scatter)
plt.savefig("clusters.png", dpi=150)
```

---

## Build Instructions (Peter's implementations)

### Modules to load on Kingspeak
```bash
module load gcc
module load cuda
module load openmpi
```

### Makefile targets
```bash
make mpi        # builds MPI CPU version
make mpi_cuda   # builds MPI + CUDA version
```

---

## Output Format
Use the same CSV format as my implementations so validation works:

```
song_id, cluster_id, feature_1, feature_2, feature_3, ...
```

One row per song.

---

## Peter's Timeline

| Days | What Peter's working on |
|---|---|
| 1–7 | Wait for serial + CUDA to be ready, set up Kingspeak environment |
| 8–10 | MPI distributed CPU version |
| 11–13 | MPI + CUDA version |
| 14–16 | Submit and run all scaling studies on Kingspeak |
| 17–18 | Visualization script |
| 19–21 | Descriptions, build instructions, final cleanup |

> If MPI CPU is done early, start submitting scaling study jobs right away — don't wait until everything is finished.

---

## Notes
- Cite any external code used in both comments and the README
- All MPI code must build and run on Kingspeak — test early
- Validation against serial output must pass before scaling studies are run
- Coordinate with me on the shared utils if anything needs to change
