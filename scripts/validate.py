#!/usr/bin/env python3
"""validate.py — Validate K-Means clustering output against a reference."""
# TODO: implement validation logic (e.g., compare cluster assignments, inertia)

import argparse


def parse_args():
    parser = argparse.ArgumentParser(
        description="Validate K-Means clustering output."
    )
    parser.add_argument("--output", required=True,
                        help="Path to the clustering output CSV file.")
    parser.add_argument("--reference", required=True,
                        help="Path to the reference/ground-truth CSV file.")
    parser.add_argument("--tol", type=float, default=1e-4,
                        help="Tolerance for centroid comparison (default: 1e-4).")
    return parser.parse_args()


def main():
    args = parse_args()
    # TODO: load output and reference files
    # TODO: compare cluster assignments or centroids within args.tol
    # TODO: print PASS / FAIL with diagnostic information
    print("Validation: not yet implemented.")


if __name__ == "__main__":
    main()
