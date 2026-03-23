# Genre Reveal Party — My Tasks (60%)
**Parallel K-Means Clustering on 1.2M Spotify Songs**  
**Course:** Parallel Programming | **Platform:** CHPC Kingspeak  
**My teammate:** Peter

---

## My Responsibilities at a Glance

| Task | Points |
|---|---|
| Shared utility code (foundation for everything) | — |
| Serial implementation (baseline) | — |
| OpenMP shared memory | 15 pts |
| CUDA GPU implementation | 15 pts |
| Validation function | 5 pts |
| Code reuse across implementations | 10 pts |
| Build/run instructions + descriptions (shared with Peter) | 5 pts |
| **My total** | **~50/100** |

> The serial + OpenMP + CUDA chain is the technical foundation that Peter's MPI work builds on top of. Get these done in week 1–2 so Peter isn't blocked.

---

## Step 1 — Shared Utility Code
**Do this first. Everything else depends on it.**

Put all shared code in a `utils/` folder so every implementation can reuse it:

```
utils/
  io.h / io.cpp              — load CSV, write output
  distance.h / distance.cpp  — Euclidean distance between a point and centroid
  kmeans_common.h            — shared structs: Point, Centroid, Config
  validate.h / validate.cpp  — compare two output CSVs (used for validation)
```

**How to approach it:**
- Write `load_data()` first — you need it to test anything
- Keep `compute_distance()` as a standalone function so CUDA can call a device version of it later
- `Config` struct should hold K, max_iter, threshold, thread count, block size — so all implementations share the same config parsing

---

## Step 2 — Serial Implementation (Baseline)
**Owner: me | Goal: get the algorithm right before parallelizing anything**

This is the correctness reference. Every other implementation must match this output.

**What to build:**
- Single-threaded K-Means loop in C/C++
- Configurable K, max iterations, and convergence threshold via command-line args
- Reads dataset from a path passed as argument
- Writes output CSV

**Key functions (reuse from utils where possible):**
```cpp
load_data()         // parse CSV, extract features
assign_clusters()   // assign each song to nearest centroid
update_centroids()  // recompute centroid positions
check_convergence() // stop when centroids move less than threshold
```

**Run it like this:**
```bash
./serial --input data/songs.csv --k 10 --max_iter 100 --threshold 0.001
```

**Done when:** it runs, converges, and writes a correct output CSV.

---

## Step 3 — OpenMP Shared Memory (15 pts)
**Owner: me | Builds on: serial**

Parallelize the two inner loops using OpenMP — the assignment loop and the centroid update loop.

**How to approach it:**
- Add `#pragma omp parallel for` to the assignment loop first — this is the easy win
- The centroid update loop has a race condition: multiple threads writing to the same centroid. Fix it with either:
  - `#pragma omp reduction` (cleaner)
  - `#pragma omp atomic` (more manual)
- K and thread count should be runtime arguments

**Run it like this:**
```bash
./openmp --input data/songs.csv --k 10 --threads 8
```

**What to document:**
- Which loops are parallelized
- How race conditions are handled
- Thread counts used in scaling study (1, 2, 4, 8, 16)

**Done when:** passes validation against serial output.

---

## Step 4 — CUDA GPU Implementation (15 pts)
**Owner: me | Builds on: serial + utils**

Each CUDA thread handles one song in the assignment step.

**How to approach it:**
- Copy all songs to device memory once before the loop starts
- Each iteration:
  1. Copy centroids to device
  2. Launch kernel — each thread computes distance to all centroids, picks closest
  3. Copy updated assignments back
  4. Update centroids on CPU (or write a reduction kernel — CPU is fine for now)
- No CUDA shared memory (tiling) required

**Kernel structure:**
```cuda
__global__ void assign_clusters_kernel(
    float* songs, float* centroids, int* assignments,
    int n_songs, int k, int n_features
)
```

**Run it like this:**
```bash
./cuda --input data/songs.csv --k 10 --block_size 256
```

**What to document:**
- How data is laid out in device memory
- Block size used and why
- How centroid updates are handled

**Experiment:** Run with block sizes 64, 128, 256, 512 and report runtime.

**Done when:** passes validation against serial output.

---

## Step 5 — Validation Function (5 pts)
**Owner: me | No parallelization needed**

Write a Python script that compares the serial output to any parallel output.

**What it should do:**
- Load two output CSVs (serial and parallel)
- Compare cluster assignments song by song
- Allow small tolerance for floating point centroid differences
- Print PASS/FAIL + number of mismatches

```bash
python validate.py --serial output/serial_out.csv --parallel output/openmp_out.csv
```

Run this after every implementation is done. Every version must PASS before it's considered complete.

---

## Build Instructions (my implementations)

### Modules to load on Kingspeak
```bash
module load gcc
module load cuda
```

### Makefile targets
```bash
make serial    # builds serial
make openmp    # builds OpenMP version
make cuda      # builds CUDA version
```

---

## Output Format
Every implementation I write must produce this CSV format so Peter's visualization script works:

```
song_id, cluster_id, feature_1, feature_2, feature_3, ...
```

One row per song.

---

## My Timeline

| Days | What I'm working on |
|---|---|
| 1–2 | Shared utils + serial implementation |
| 3–4 | OpenMP version + validate against serial |
| 5–7 | CUDA version + validate against serial |
| 8 | Validation script, descriptions, build instructions |
| 9–21 | Support Peter if needed, final cleanup |

> Finish CUDA by day 7 at the latest so Peter can start on MPI+CUDA without waiting on me.

---

## Notes
- Cite any external code used in both comments and the README
- Keep `compute_distance()` clean and reusable — Peter's MPI version will use it too
- Test locally first, then verify it builds and runs on Kingspeak before calling it done
