import os

PATH_OUTPUT = "output"
MODEL_CLUSTERING = "kmeans"
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

# Default location expected by delivery requirements.
PATH_DATA = os.path.join(REPO_ROOT, "data", "test")

# Backward compatible fallback while migrating old folder layout.
LEGACY_PATH_DATA = os.path.join(REPO_ROOT, "Projet", "donnees", "test")