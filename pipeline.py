"""Root entrypoint for the IA pipeline.

Usage:
    python pipeline.py --path_data data/test --path_output output
"""

import argparse

from Projet.src.pipeline import pipeline as run_pipeline


def parse_args():
    parser = argparse.ArgumentParser(description="Pipeline IA de clustering")
    parser.add_argument(
        "--path_data",
        type=str,
        default=None,
        help="Chemin vers le dossier des images (defaut: data/test)",
    )
    parser.add_argument(
        "--path_output",
        type=str,
        default=None,
        help="Chemin vers le dossier de sortie (defaut: output)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_pipeline(path_data=args.path_data, path_output=args.path_output)
