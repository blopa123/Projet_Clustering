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

# Copier l'ensemble du projet dans /app/Projet pour préserver l'arborescence
COPY Projet/ ./Projet/

# Copier les résultats de sortie (fichiers Excel) pour que Streamlit puisse les lire (optionnel)
COPY output/ ./output/

# Exposer le port utilisé par Streamlit
EXPOSE 8000

# Définir une variable d'environnement pointant vers les données (conforme à `constant.py`)
ENV PATH_DATA=/app/Projet/donnees/test

# Lancer le dashboard Streamlit depuis le chemin Projet/src
CMD ["streamlit", "run", "Projet/src/dashboard_clustering.py", "--server.port", "8000", "--server.address", "0.0.0.0"]
