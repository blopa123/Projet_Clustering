# Projet Clustering SNACK

## Arborescence attendue

- Donnees d'entree (structure actuelle): `Projet/donnees/test/`
- Donnees d'entree (compatible): `data/test/`
- Sorties IA: `output/`


## 1. Installation des packages

```bash
pip install -r requirements.txt
```

## 2. Run Pipeline IA

```bash
python pipeline.py --path_data "Projet/donnees/test" --path_output "output"
```

Alternative (si vous avez deja un dossier `data/test` a la racine):

```bash
python pipeline.py --path_data "data/test" --path_output "output"
```

Sorties generees pour le dashboard (formats Excel + CSV):
- `save_metric.xlsx` et `save_metric.csv`
- `save_clustering_*_kmeans.xlsx` et `.csv`
- `save_clustering_*_meanshift.xlsx` et `.csv`

## 3. Run Pipeline Dashboard

```bash
python dashboard.py --path_data "output" --port 8000
```

Le serveur ecoute sur `0.0.0.0` (toutes les interfaces) pour etre compatible local + Docker.
Pour ouvrir l'application en local, utilisez: `http://localhost:8000`.

## 4. Lancer avec Docker

Prerequis:
- Docker Desktop installe et demarre

Construire l'image:

```bash
docker build -t snack-dashboard .
```

Demarrer le conteneur:

```bash
docker run -d --name snack-dashboard-container -p 8000:8000 snack-dashboard
```

Verifier qu'il tourne:

```bash
docker ps
```

Afficher les logs (optionnel):

```bash
docker logs snack-dashboard-container
```

Le dashboard est accessible sur `http://localhost:8000`.

Arreter et supprimer le conteneur apres la demo:

```bash
docker stop snack-dashboard-container
docker rm snack-dashboard-container
```