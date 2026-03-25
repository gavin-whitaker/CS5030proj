#!/usr/bin/env python3

import argparse


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
    args = parser.parse_args()

    # TODO: implement loading CSV, extracting the requested feature columns,
    # and producing a 3D scatter plot colored by cluster id.
    print(
        f"TODO (stub): visualize.py would plot {args.features} from {args.input} into {args.output}."
    )


if __name__ == "__main__":
    main()

