#!/usr/bin/env python3

import argparse
import csv
import sys
import random

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

FEATURE_NAMES = [
    "danceability", "energy", "acousticness",
    "instrumentalness", "valence", "tempo"
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize K-Means clustering results.")
    parser.add_argument("--input", required=True, help="Input output CSV from a K-Means implementation.")
    parser.add_argument("--k", type=int, required=True, help="Number of clusters.")
    parser.add_argument(
        "--features",
        nargs="+",
        required=True,
        help="Exactly 3 feature names to plot (e.g., valence danceability energy).",
    )
    parser.add_argument("--output", default="plot.png", help="Output PNG filename.")
    parser.add_argument("--sample", type=int, default=None, help="Subsample to N random points (default: all).")
    args = parser.parse_args()

    # Validate exactly 3 features
    if len(args.features) != 3:
        print("Error: must specify exactly 3 features.", file=sys.stderr)
        sys.exit(1)

    # Map feature names to column indices
    try:
        feat_indices = [FEATURE_NAMES.index(f) for f in args.features]
    except ValueError as e:
        print(f"Error: invalid feature name. Choose from: {', '.join(FEATURE_NAMES)}", file=sys.stderr)
        sys.exit(1)

    # Read CSV: columns are [song_id, cluster_id, f0, f1, f2, f3, f4, f5]
    xs, ys, zs, colors = [], [], [], []

    try:
        with open(args.input) as f:
            reader = csv.reader(f)
            next(reader)  # skip header
            rows = list(reader)
    except Exception as e:
        print(f"Error reading {args.input}: {e}", file=sys.stderr)
        sys.exit(1)

    # Subsample if requested
    if args.sample is not None and args.sample < len(rows):
        rows = random.sample(rows, args.sample)

    # Extract data
    for row in rows:
        try:
            cluster_id = int(row[1])
            # Feature columns start at index 2
            feats = [float(row[2 + fi]) for fi in feat_indices]
            xs.append(feats[0])
            ys.append(feats[1])
            zs.append(feats[2])
            colors.append(cluster_id)
        except (IndexError, ValueError) as e:
            print(f"Warning: skipping malformed row: {row}", file=sys.stderr)
            continue

    if not xs:
        print("Error: no valid data points to plot.", file=sys.stderr)
        sys.exit(1)

    # Create 3D scatter plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    sc = ax.scatter(xs, ys, zs, c=colors, cmap='tab20', s=0.5, alpha=0.4)
    ax.set_xlabel(args.features[0])
    ax.set_ylabel(args.features[1])
    ax.set_zlabel(args.features[2])
    ax.set_title(f"K-Means k={args.k}: {', '.join(args.features)}")
    plt.colorbar(sc, ax=ax, label='cluster', shrink=0.5)

    try:
        plt.savefig(args.output, dpi=150, bbox_inches='tight')
        print(f"Saved plot to {args.output}")
    except Exception as e:
        print(f"Error saving plot: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

