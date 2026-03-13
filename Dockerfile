# Utilise une image Python légère et lance le dashboard Streamlit
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1

ENV PIP_DEFAULT_TIMEOUT=120

# Installer les dépendances système nécessaires pour OpenCV et le rendu
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
	&& apt-get install -y --no-install-recommends \
	   libgl1 \
	   libglib2.0-0 \
	   libsm6 \
	   libxrender1 \
	   libxext6 \
	   ffmpeg \
	&& rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copier le fichier de dépendances
COPY requirements.txt ./

# Installer les dépendances (augmenter le timeout réseau)
RUN pip install --no-cache-dir --timeout 120 -r requirements.txt

# Copier le code source
COPY Projet/ ./Projet/
COPY pipeline.py ./pipeline.py
COPY dashboard.py ./dashboard.py

# Copier les résultats de sortie (fichiers Excel/CSV) pour que le dashboard puisse les lire (optionnel)
COPY output/ ./output/

# Aligner les donnees sur la convention demandee: /app/data/test
COPY Projet/donnees/test/ ./data/test/

# Exposer le port utilisé par Streamlit
EXPOSE 8000

# Variables d'environnement pour les chemins par défaut
ENV PATH_DATA=/app/data/test
ENV PATH_ANALYSIS=/app/output

# Lancer le dashboard via le script racine
CMD ["python", "dashboard.py", "--path_data", "/app/output", "--port", "8000"]
