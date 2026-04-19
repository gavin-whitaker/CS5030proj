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

def validate_feature_count(features):
    if len(features) != 3:
        print("Error: must specify exactly 3 features.", file=sys.stderr)
        sys.exit(1)

def get_feature_indices(features):
    indices = []
    for f in features:
        try:
            indices.append(FEATURE_NAMES.index(f))
        except ValueError:
            print(f"Error: invalid feature name. Choose from: {', '.join(FEATURE_NAMES)}", file=sys.stderr)
            sys.exit(1)
    return indices

def read_csv(filepath):
    try:
        with open(filepath) as f:
            reader = csv.reader(f)
            next(reader)
            rows = list(reader)
        return rows
    except Exception as e:
        print(f"Error reading {filepath}: {e}", file=sys.stderr)
        sys.exit(1)

def subsample_rows(rows, sample_size):
    if sample_size is not None and sample_size < len(rows):
        return random.sample(rows, sample_size)
    return rows

def extract_point_data(rows, feat_indices):
    xs = []
    ys = []
    zs = []
    colors = []

    for row in rows:
        try:
            cluster_id = int(row[1])
            feats = [float(row[2 + fi]) for fi in feat_indices]
            xs.append(feats[0])
            ys.append(feats[1])
            zs.append(feats[2])
            colors.append(cluster_id)
        except (IndexError, ValueError):
            print(f"Warning: skipping malformed row: {row}", file=sys.stderr)
            continue

    if not xs:
        print("Error: no valid data points to plot.", file=sys.stderr)
        sys.exit(1)

    return xs, ys, zs, colors

def create_plot(xs, ys, zs, colors, feature_names, k):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    sc = ax.scatter(xs, ys, zs, c=colors, cmap='tab20', s=0.5, alpha=0.4)
    ax.set_xlabel(feature_names[0])
    ax.set_ylabel(feature_names[1])
    ax.set_zlabel(feature_names[2])
    ax.set_title(f"K-Means k={k}: {', '.join(feature_names)}")
    plt.colorbar(sc, ax=ax, label='cluster', shrink=0.5)
    return fig

def save_plot(fig, output):
    try:
        fig.savefig(output, dpi=150, bbox_inches='tight')
        print(f"Saved plot to {output}")
    except Exception as e:
        print(f"Error saving plot: {e}", file=sys.stderr)
        sys.exit(1)

def main():
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

    validate_feature_count(args.features)
    feat_indices = get_feature_indices(args.features)
    rows = read_csv(args.input)
    rows = subsample_rows(rows, args.sample)
    xs, ys, zs, colors = extract_point_data(rows, feat_indices)
    fig = create_plot(xs, ys, zs, colors, args.features, args.k)
    save_plot(fig, args.output)

if __name__ == "__main__":
    main()

