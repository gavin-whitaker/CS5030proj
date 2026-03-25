# Genre Reveal Party — Curt's Tasks (33%)
**Parallel K-Means Clustering on 1.2M Spotify Songs**  
**Course:** Parallel Programming | **Platform:** CHPC Kingspeak

---

## Curt's Responsibilities at a Glance

| Task | Share |
|---|---|
| OpenMP shared-memory implementation | 1/3 |
| CUDA single-node GPU implementation | 1/3 |
| Visualization + documentation + final integration checks | 1/3 |
| **Curt's total** | **~33%** |

---

## 1) OpenMP Shared Memory
**Owner: Curt | Builds on serial baseline**

Parallelize CPU loops with OpenMP:
- assignment loop with `#pragma omp parallel for`
- centroid update with reduction/atomic to avoid races
- support runtime thread count argument

Example:
```bash
./openmp --input data/songs.csv --k 10 --threads 8
```

---

## 2) CUDA GPU Implementation
**Owner: Curt | Builds on shared utils**

Implement single-node CUDA assignment kernel:
- copy songs to device once before iterations
- copy centroids each iteration
- launch kernel (one thread per song)
- copy assignments back and update centroids

Example:
```bash
./cuda --input data/songs.csv --k 10 --block_size 256
```

---

## 3) Visualization + Documentation + Integration
**Owner: Curt**

Own final presentation and integration tasks:
- write `visualize.py` (3D cluster visualization PNG output)
- document OpenMP/CUDA design decisions and run steps
- verify all outputs use the shared CSV format
- confirm all implementations pass validation before final submission

Example:
```bash
python visualize.py --input output/serial_out.csv --k 10 --features valence danceability energy
```

---

## Build Targets Curt Maintains
```bash
make openmp
make cuda
```

---

## Shared Output Format
```text
song_id, cluster_id, feature_1, feature_2, feature_3, ...
```

One row per song, same format for all implementations.
