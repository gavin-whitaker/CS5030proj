#!/usr/bin/env python3
"""
validate.py — Compare a serial K-Means output CSV against a parallel one.

Usage:
    python validate.py --serial results/serial_out.csv \
                       --parallel results/openmp_out.csv \
                       [--tolerance 1e-4]

Both CSVs must have the shared output format:
    song_id, cluster_id, feature_0, feature_1, ...

Two comparisons are performed:

  1. Direct: exact cluster_id match per song_id.  Only meaningful when both
     implementations use the same random seed / initialization.

  2. Remapped: remap parallel cluster labels to best-matching serial labels
     using greedy nearest-centroid matching, then recount mismatches.  This
     handles the label-permutation ambiguity that arises when implementations
     start from different random seeds.

PASS is declared when the remapped mismatch count is zero (or within
--tolerance as a fraction of total songs).
"""

import argparse
import csv
import math
import sys
from typing import Dict, List, Tuple


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------
Record = Tuple[int, List[float]]   # (cluster_id, [features])
Dataset = Dict[int, Record]        # song_id -> (cluster_id, features)


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------
def load_csv(path: str) -> Dataset:
    data: Dataset = {}
    with open(path, newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        if len(header) < 2:
            sys.exit(f"Error: {path} has fewer than 2 columns.")
        for row in reader:
            if not row:
                continue
            try:
                song_id = int(row[0])
                cluster_id = int(row[1])
                features = [float(v) for v in row[2:]]
            except (ValueError, IndexError) as e:
                sys.exit(f"Error parsing row in {path}: {row!r}  ({e})")
            data[song_id] = (cluster_id, features)
    return data


# ---------------------------------------------------------------------------
# Centroid computation
# ---------------------------------------------------------------------------
def compute_centroids(data: Dataset, k: int) -> Dict[int, List[float]]:
    """Return mean feature vector per cluster label."""
    sums: Dict[int, List[float]] = {}
    counts: Dict[int, int] = {}
    for cluster_id, features in data.values():
        if cluster_id not in sums:
            n = len(features)
            sums[cluster_id] = [0.0] * n
            counts[cluster_id] = 0
        for i, v in enumerate(features):
            sums[cluster_id][i] += v
        counts[cluster_id] += 1
    centroids = {}
    for c, s in sums.items():
        n = counts[c]
        centroids[c] = [v / n for v in s]
    return centroids


def euclidean(a: List[float], b: List[float]) -> float:
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


# ---------------------------------------------------------------------------
# Greedy centroid-to-centroid label remapping
# For each parallel cluster, find the closest serial cluster centroid (by
# Euclidean distance).  Uses a greedy one-to-one matching — sufficient for
# class purposes; Hungarian algorithm not required.
# ---------------------------------------------------------------------------
def build_label_map(
    serial_centroids: Dict[int, List[float]],
    parallel_centroids: Dict[int, List[float]],
) -> Dict[int, int]:
    """Return mapping: parallel_label -> serial_label."""
    serial_labels = list(serial_centroids.keys())
    parallel_labels = list(parallel_centroids.keys())

    used_serial: set = set()
    mapping: Dict[int, int] = {}

    # Sort parallel labels for determinism
    for pl in sorted(parallel_labels):
        pc = parallel_centroids[pl]
        best_sl, best_dist = -1, float("inf")
        for sl in serial_labels:
            if sl in used_serial:
                continue
            d = euclidean(pc, serial_centroids[sl])
            if d < best_dist:
                best_dist = d
                best_sl = sl
        if best_sl == -1:
            # All serial labels used (more parallel clusters than serial) — map to self
            mapping[pl] = pl
        else:
            mapping[pl] = best_sl
            used_serial.add(best_sl)

    return mapping


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate parallel K-Means output against the serial baseline."
    )
    parser.add_argument("--serial", required=True, help="Path to serial output CSV.")
    parser.add_argument("--parallel", required=True, help="Path to parallel output CSV.")
    parser.add_argument(
        "--tolerance",
        type=float,
        default=0.0,
        help="Max allowed mismatch fraction (0.0 = exact match required). Default: 0.0",
    )
    args = parser.parse_args()

    print(f"Loading serial:   {args.serial}")
    serial = load_csv(args.serial)
    print(f"Loading parallel: {args.parallel}")
    parallel = load_csv(args.parallel)

    # --- Basic sanity checks ---
    if len(serial) != len(parallel):
        print(
            f"FAIL: row count mismatch — serial={len(serial)}, parallel={len(parallel)}"
        )
        sys.exit(1)

    serial_ids = set(serial.keys())
    parallel_ids = set(parallel.keys())
    if serial_ids != parallel_ids:
        missing = serial_ids - parallel_ids
        extra = parallel_ids - serial_ids
        print(f"FAIL: song_id sets differ. missing={len(missing)} extra={len(extra)}")
        sys.exit(1)

    total = len(serial)
    print(f"Total songs: {total}")

    # --- Direct comparison ---
    direct_mismatches = sum(
        1 for sid in serial_ids if serial[sid][0] != parallel[sid][0]
    )
    print(f"\nDirect comparison (same label space):")
    print(f"  Mismatches: {direct_mismatches}/{total}")

    # Infer k from cluster IDs present
    serial_k = len({v[0] for v in serial.values()})
    parallel_k = len({v[0] for v in parallel.values()})
    print(f"  Serial   K (observed): {serial_k}")
    print(f"  Parallel K (observed): {parallel_k}")

    # --- Centroid-remapping comparison ---
    serial_centroids = compute_centroids(serial, serial_k)
    parallel_centroids = compute_centroids(parallel, parallel_k)
    label_map = build_label_map(serial_centroids, parallel_centroids)

    remapped_mismatches = sum(
        1
        for sid in serial_ids
        if label_map.get(parallel[sid][0], parallel[sid][0]) != serial[sid][0]
    )
    print(f"\nRemapped comparison (centroid-based label alignment):")
    print(f"  Mismatches: {remapped_mismatches}/{total}")

    # --- PASS / FAIL ---
    allowed = int(math.floor(args.tolerance * total))
    print(f"\nTolerance: {args.tolerance:.2%} → allowed mismatches ≤ {allowed}")

    if remapped_mismatches <= allowed:
        print("PASS")
        sys.exit(0)
    else:
        print("FAIL")
        sys.exit(1)


if __name__ == "__main__":
    main()

