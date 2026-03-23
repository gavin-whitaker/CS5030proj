#!/usr/bin/env python3
"""visualize.py — Visualize K-Means clustering results."""
# TODO: implement visualization (e.g., scatter plot colored by cluster)

import argparse


def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize K-Means clustering results."
    )
    parser.add_argument("--output", required=True,
                        help="Path to the clustering output CSV file.")
    parser.add_argument("--dims", type=int, default=2,
                        help="Number of dimensions to plot (default: 2).")
    parser.add_argument("--save", default=None,
                        help="If set, save the plot to this file instead of displaying.")
    return parser.parse_args()


def main():
    args = parse_args()
    # TODO: load output CSV with point coordinates and cluster assignments
    # TODO: create scatter plot with points colored by cluster ID
    # TODO: overlay centroid markers
    # TODO: save or display the figure
    print("Visualization: not yet implemented.")


if __name__ == "__main__":
    main()
