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

# Copier le fichier de dépendances (note: le projet utilise 'requierements.txt')
COPY requierements.txt ./

# Installer les dépendances (augmenter le timeout réseau)
RUN pip install --no-cache-dir --timeout 120 -r requierements.txt

# Copier le code source et les données (si vous voulez les monter depuis l'hôte, vous pouvez omettre la copie des données)
COPY src/ ./src/
COPY donnees/ ./donnees/
# Copier les résultats de sortie (fichiers Excel) pour que Streamlit puisse les lire
COPY output/ ./output/

# Exposer le port utilisé par Streamlit
EXPOSE 8000

# Optionnel: définir une variable d'environnement pointant vers les données
ENV PATH_DATA=/app/donnees

# Lancer le dashboard Streamlit (écoute sur 0.0.0.0 pour être accessible depuis l'extérieur)
CMD ["streamlit", "run", "src/dashboard_clustering.py", "--server.port", "8000", "--server.address", "0.0.0.0"]
