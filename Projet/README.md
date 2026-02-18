# Projet de clustering des immages SNACK pour ET4 info Polytech Paris Saclay

### step 1 : téléchargement des données et installation des packages
    - a. télécharger les données Snack : https://huggingface.co/datasets/Matthijs/snacks/tree/main
        => Utilisez seulement le dossier "test".
    - b. installer les requierements : "pip install -r requierements.txt"
### step 2 : configuration du chemin vers les donnés
    - dans le dossier src/constant.py, modifier la variable "PATH_DATA" par le chemin vers le dossier contenant les données à clusteriser.

### step 3 :  run de la pipeline clustering
    - aller dans le dossier src
    - exécutez la commande : "python pipeline.py"
    
### step 4 : lancement du dashboard
    - aller dans le dossier src 
    - exécutez la commande : "streamlit run dashboard_clustering.py"

### Step 5 : Dockerfile
    - installer Docker
    - créer un dockerfile pour exécuter (seulement) le dashboard