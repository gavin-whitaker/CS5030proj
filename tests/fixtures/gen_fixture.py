#!/usr/bin/env python3
"""
Generate synthetic Spotify-format CSV for testing.

Output columns (must match Kaggle dataset):
id, name, album, album_id, artists, artist_ids, track_number, disc_number,
explicit, danceability, energy, key, loudness, mode, speechiness,
acousticness, instrumentalness, liveness, valence, tempo, ...

The K-Means code uses hardcoded column indices:
  9: danceability
 10: energy
 15: acousticness
 16: instrumentalness
 18: valence
 19: tempo
"""

import sys
import csv

def gen_fixture(n_rows, output_path):
    """Generate n_rows of synthetic data."""
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        # Header: must have at least 20 columns (up to tempo at index 19)
        writer.writerow([
            'id', 'name', 'album', 'album_id', 'artists', 'artist_ids',
            'track_number', 'disc_number', 'explicit',
            'danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness',
            'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo',
            'duration_ms', 'time_signature', 'year', 'release_date'
        ])

        # Generate n_rows of data
        for i in range(n_rows):
            row = [
                f'id_{i}',                          # id
                f'track_{i}',                       # name
                'Test Album',                       # album
                'album_id_1',                       # album_id
                "['Artist']",                       # artists (quoted)
                "['artist_id_1']",                  # artist_ids
                '1',                                # track_number
                '1',                                # disc_number
                'False',                            # explicit
                0.5 + (i * 0.01) % 0.5,            # danceability [0,1]
                0.6 + (i * 0.02) % 0.4,            # energy [0,1]
                '5',                                # key
                '-5.0',                             # loudness
                '1',                                # mode
                '0.05',                             # speechiness
                0.1 + (i * 0.005) % 0.9,           # acousticness [0,1]
                0.0 + (i * 0.001) % 1.0,           # instrumentalness [0,1]
                '0.2',                              # liveness
                0.5 + (i * 0.015) % 0.5,           # valence [0,1]
                '120.0',                            # tempo
                '240000',                           # duration_ms
                '4',                                # time_signature
                '2020',                             # year
                '2020-01-01'                        # release_date
            ]
            writer.writerow(row)

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print('Usage: gen_fixture.py <n_rows> <output.csv>')
        sys.exit(1)

    n = int(sys.argv[1])
    out = sys.argv[2]
    gen_fixture(n, out)
    print(f'Generated {n} rows to {out}')
