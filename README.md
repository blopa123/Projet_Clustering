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

Alternative si vos donnees sont dans `data/test`:

```bash
python pipeline.py --path_data "data/test" --path_output "output"
```

## 3. Run Pipeline Dashboard

```bash
python dashboard.py --path_data "output" --port 8000
```

Acces application: `http://localhost:8000`

## 4. Lancer avec Docker

Prerequis:
- Docker Desktop installe et demarre

Construire l'image:

```bash
docker build --no-cache -t snack-dashboard .
```

Important: le dashboard lit les resultats du pipeline dans `/app/output`.
Il faut donc que le dossier local `output/` contienne deja les CSV/XLSX (genere via le pipeline local), puis le monter dans le conteneur.

Demarrer le conteneur (Linux/macOS):

```bash
docker run -d --name snack-dashboard-container -p 8000:8000 -v "$(pwd)/output:/app/output" snack-dashboard
```

Demarrer le conteneur (Windows PowerShell):

```powershell
docker run -d --name snack-dashboard-container -p 8000:8000 -v "${PWD}/output:/app/output" snack-dashboard
```

Verifier:

```bash
docker ps
docker logs snack-dashboard-container
```

Acces application: `http://localhost:8000`

Arret et nettoyage:

```bash
docker stop snack-dashboard-container
docker rm snack-dashboard-container
```
