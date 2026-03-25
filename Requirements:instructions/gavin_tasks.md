# Genre Reveal Party — My Tasks (33%)
**Parallel K-Means Clustering on 1.2M Spotify Songs**  
**Course:** Parallel Programming | **Platform:** CHPC Kingspeak  
**Team:** Me, Peter, Curt

---

## My Responsibilities at a Glance

| Task | Share |
|---|---|
| Shared utility code foundation (`utils/`) | 1/3 |
| Serial implementation (baseline correctness) | 1/3 |
| Validation script + output consistency checks | 1/3 |
| **My total** | **~33%** |

---

## 1) Shared Utility Code
**Owner: me | Do first**

Create and own reusable utility code used by all implementations:

```
utils/
  io.h / io.cpp              — load CSV, write output
  distance.h / distance.cpp  — Euclidean distance
  kmeans_common.h            — Point, Centroid, Config
  validate.h / validate.cpp  — comparison helpers
```

**Done when:**
- `load_data()` works for the full dataset
- `compute_distance()` is reusable by CPU and GPU implementations
- `Config` supports K, max_iter, threshold, threads, block_size

---

## 2) Serial Baseline
**Owner: me | Source of truth for correctness**

Build single-threaded K-Means in C/C++:
- configurable `k`, `max_iter`, `threshold`
- CSV input path from CLI
- output CSV in shared format

Example run:
```bash
./serial --input data/songs.csv --k 10 --max_iter 100 --threshold 0.001
```

---

## 3) Validation + Consistency
**Owner: me**

Write/maintain `validate.py`:
- compare serial output to OpenMP/CUDA/MPI/MPI+CUDA outputs
- report PASS/FAIL and mismatches
- allow tolerance for floating-point differences

Example:
```bash
python validate.py --serial output/serial_out.csv --parallel output/openmp_out.csv
```

All team implementations must pass validation before scaling results are finalized.

---

## Build Targets I Maintain
```bash
make serial
```

---

## Shared Output Format
```text
song_id, cluster_id, feature_1, feature_2, feature_3, ...
```

One row per song, same format for all implementations.
