"""Root entrypoint to run the Streamlit dashboard.

Usage:
    python dashboard.py --path_data output
"""

import argparse
import os
import subprocess
import sys


def parse_args():
    parser = argparse.ArgumentParser(description="Lancer le dashboard de clustering")
    parser.add_argument(
        "--path_data",
        type=str,
        default=None,
        help="Chemin vers le dossier des sorties IA (save_metric, save_clustering_*)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port du serveur Streamlit",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    env = os.environ.copy()
    if args.path_data:
        env["PATH_ANALYSIS"] = os.path.abspath(args.path_data)

    app_path = os.path.join("Projet", "src", "dashboard_clustering.py")
    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        app_path,
        "--server.port",
        str(args.port),
        "--server.address",
        "0.0.0.0",
    ]
    subprocess.run(cmd, check=True, env=env)


if __name__ == "__main__":
    main()
