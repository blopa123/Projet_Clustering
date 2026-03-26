import os

PATH_OUTPUT = "output"
MODEL_CLUSTERING = "kmeans"
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

# Emplacement par défaut attendu selon les exigences de livraison.
PATH_DATA = os.path.join(REPO_ROOT, "data", "test")

# Solution de repli rétrocompatible lors de la migration de l'ancienne structure de dossiers.
LEGACY_PATH_DATA = os.path.join(REPO_ROOT, "Projet", "donnees", "test")
