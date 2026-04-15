# Genre Reveal Party — Project Requirements
**Parallel K-Means Clustering on 1.2M Spotify Songs**  
**Course:** Parallel Programming | **Platform:** CHPC Kingspeak  
**Team:** 2 people
 
---
 
## Table of Contents
1. [Project Overview](#project-overview)
2. [Dataset](#dataset)
3. [Team Responsibilities](#team-responsibilities)
4. [Implementations Required](#implementations-required)
5. [Shared Utility Code](#shared-utility-code)
6. [Build & Run Instructions](#build--run-instructions)
7. [Scaling Studies](#scaling-studies)
8. [Validation](#validation)
9. [Visualization](#visualization)
10. [Output Format](#output-format)
11. [Grading Checklist](#grading-checklist)
 
---
 
## Project Overview
 
Implement a parallel K-Means clustering algorithm in C or C++ that clusters 1.2M+ Spotify songs using audio features (e.g., energy, valence, danceability, tempo, acousticness, instrumentalness). The program must support configurable K at runtime. All implementations must run on CHPC Kingspeak.
 
**Reference tutorial for serial implementation:**  
https://www.kaggle.com/datasets/rodolfofigueroa/spotify-12m-songs
 
---
 
## Dataset
 
- **Source:** https://www.kaggle.com/datasets/rodolfofigueroa/spotify-12m-songs
- **Size:** 1.2M+ songs
- **Features to use:** valence, danceability, energy, acousticness, instrumentalness, tempo (minimum 3, choose the most discriminant)
- **Format:** CSV — must be parsed at runtime, path passed as a command-line argument
 
---
 
## Team Responsibilities
 
| Task | Owner |
|---|---|
| Serial implementation | Person 1 (you) |
| OpenMP shared memory | Person 1 (you) |
| CUDA GPU implementation | Person 1 (you) |
| Shared utility code (I/O, distance, centroid update) | Person 1 (you) |
| Validation function | Person 1 (you) |
| MPI distributed CPU | Person 2 (teammate) |
| MPI + CUDA distributed GPU | Person 2 (teammate) |
| Scaling studies on Kingspeak | Person 2 (teammate) |
| Output visualization script | Person 2 (teammate) |
 
> **Note:** All external code reused must be cited in comments and in the README.
 
---
 
## Implementations Required
 
All 5 implementations must share the same utility code and produce the same output format. Build each one before moving to the next.
 
---
 
### 1. Serial (Baseline)
- Single-threaded C/C++ implementation
- Used as the correctness baseline for all other versions
- Must support configurable K as a command-line argument
- Must support configurable max iterations and convergence threshold
 
**Key functions to implement (shared across all versions):**
- `load_data()` — parse CSV, extract selected features
- `compute_distance()` — Euclidean distance between a point and a centroid
- `assign_clusters()` — assign each point to nearest centroid
- `update_centroids()` — recompute centroid positions
- `check_convergence()` — stop when centroids move less than threshold
 
---
 
### 2. Shared Memory CPU (OpenMP)
- Parallelize the assignment and centroid update loops using OpenMP
- Watch for race conditions on centroid accumulation — use reduction or atomic operations
- K and number of threads should be runtime arguments
- Must produce the same cluster assignments as the serial version (within tolerance)
 
**Parallelization strategy to document:**
- Which loops are parallelized
- How centroid updates are handled safely
- Thread count used in experiments
 
---
 
### 3. CUDA GPU
- Each thread handles one data point in the assignment step
- No CUDA shared memory (tiling) required, but allowed for extra credit
- Data transfer: copy songs to device once before the loop, copy centroids each iteration
- K and block size should be runtime arguments
 
**Things to document:**
- How data is laid out in device memory
- Block size used and why
- How centroid updates are handled (reduction on GPU or copy back to CPU each iteration)
 
**Experiment:** Run with different block sizes (e.g., 64, 128, 256, 512) and report performance.
 
---
 
### 4. Distributed Memory CPU (MPI)
- Divide the 1.2M songs evenly across MPI processes
- Each process computes local cluster assignments and partial centroid sums
- Use `MPI_Allreduce` to combine centroid updates across all processes each iteration
- All processes check convergence together
 
**Things to document:**
- How data is split (block decomposition)
- Which MPI calls are used and why
- How convergence is checked globally
 
**Experiments:** Run on 2, 3, and 4 Kingspeak nodes.
 
---
 
### 5. Distributed Memory GPU (MPI + CUDA)
- Combines implementations 3 and 4
- Each MPI process manages one GPU
- GPU handles assignment step per process, MPI handles centroid sync across nodes
- Use `MPI_Allreduce` for centroid updates same as implementation 4
 
**Things to document:**
- How MPI and CUDA interact
- Data flow: CPU ↔ GPU ↔ MPI
- Any bottlenecks identified
 
**Experiments:** Run on 2, 3, and 4 Kingspeak nodes.
 
---
 
## Shared Utility Code
 
These must be written once and reused across all implementations. Put them in a `utils/` folder:
 
```
utils/
  io.h / io.cpp          — CSV loading, output writing
  distance.h / distance.cpp  — Euclidean distance
  kmeans_common.h        — shared structs (Point, Centroid, Config)
  validate.h / validate.cpp  — compare two output CSV files
```
 
---
 
## Build & Run Instructions
 
Provide a `Makefile` with targets for each implementation:
 
```bash
make serial
make openmp
make cuda
make mpi
make mpi_cuda
```
 
### CHPC Kingspeak Modules to Load
 
```bash
module load gcc
module load cuda
module load openmpi
```
 
### Example Run Commands
 
```bash
# Serial
./serial --input data/songs.csv --k 10 --max_iter 100 --threshold 0.001
 
# OpenMP
./openmp --input data/songs.csv --k 10 --threads 8
 
# CUDA
./cuda --input data/songs.csv --k 10 --block_size 256
 
# MPI (4 processes)
mpirun -n 4 ./mpi --input data/songs.csv --k 10
 
# MPI + CUDA (4 nodes)
mpirun -n 4 ./mpi_cuda --input data/songs.csv --k 10 --block_size 256
```
 
> All commands should work as-is on Kingspeak after loading modules. Include a sample SLURM job script for each implementation.
 
---
 
## Scaling Studies
 
### Study 1: Serial vs OpenMP (implementations 1 vs 2)
- Fix K, vary thread count: 1, 2, 4, 8, 16
- Report: runtime, speedup, efficiency
- Run each configuration at least 3 times and average
 
### Study 2: CUDA block size (implementation 3)
- Fix K, vary block size: 64, 128, 256, 512
- Report: runtime per iteration, total runtime
 
### Study 3: MPI CPU vs MPI+CUDA (implementations 4 vs 5)
- Fix K, vary node count: 2, 3, 4 nodes on Kingspeak
- Report: runtime, speedup, efficiency
- Submit jobs early — Kingspeak queues can be slow near deadlines
 
---
 
## Validation
 
Write a validation function (C/C++ or Python, no parallelization needed) that:
- Loads the serial output CSV and a parallel output CSV
- Compares cluster assignments for each song
- Allows a small tolerance for floating point differences in centroids
- Prints a PASS/FAIL result with the number of mismatches if any
 
```bash
# Example usage
python validate.py --serial output/serial_out.csv --parallel output/openmp_out.csv
```
 
Every implementation must pass validation against the serial baseline before it is considered done.
 
---
 
## Visualization
 
Write a Python script (no parallelization needed) using matplotlib or similar that:
- Loads the output CSV
- Plots songs in 3D using 3 audio features (e.g., valence, danceability, energy)
- Colors each point by its cluster assignment
- Works with any K > 2
- Saves the plot as a PNG
 
```bash
python visualize.py --input output/serial_out.csv --k 10 --features valence danceability energy
```
 
You can also use Paraview or ImageJ for data exploration as noted in the rubric.
 
---
 
## Output Format
 
Every implementation must write the same output CSV format:
 
```
song_id, cluster_id, feature_1, feature_2, feature_3, ...
```
 
One row per song. This is what validation and visualization scripts will read.
 
---
 
## Grading Checklist
 
| Requirement | Points | Owner | Done? |
|---|---|---|---|
| OpenMP shared memory implementation | 15 | Person 1 | [ ] |
| Distributed memory CPU (MPI) | 15 | Person 2 | [ ] |
| Distributed memory GPU (MPI+CUDA) | 15 | Person 2 | [ ] |
| CUDA GPU implementation | 15 | Person 1 | [ ] |
| Build/run instructions + descriptions | 10 | Both | [ ] |
| Scaling studies (serial vs OpenMP, block size, MPI vs MPI+CUDA) | 15 | Person 2 | [ ] |
| Validation function | 5 | Person 1 | [ ] |
| Code reuse across implementations | 10 | Person 1 | [ ] |
| **Total** | **100** | | |
 
---
 
## Notes
 
- Cite any external code used, in both code comments and the README
- Develop locally if easier, but all final code must build and run on Kingspeak
- Submit cluster jobs early in the last week to avoid queue delays
- Each implementation description should explain the parallel strategy, not just the code