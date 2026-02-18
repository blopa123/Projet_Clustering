# Projet de Clustering d'Images SNACK - ET4 Deep Learning

Projet de clustering d'images utilisant différents descripteurs de features (HOG, Histogrammes) et K-Means.

## 📁 Structure du projet

```
.
├── Projet/
│   ├── donnees/        # Données SNACK (20 catégories d'aliments)
│   └── README.md
│
└── sujet_tp/
    └── src/
        ├── pipeline.py              # Pipeline principal de clustering
        ├── dashboard_clustering.py  # Dashboard Streamlit
        ├── clustering.py            # Implémentation K-Means
        ├── features.py              # Extraction de features (HOG, Histogrammes)
        ├── utils.py                 # Fonctions utilitaires
        └── constant.py              # Constantes et chemins
```

## 🚀 Installation

1. Cloner le dépôt :
```bash
git clone <votre-repo>
cd "deep learning"
```

2. Créer un environnement virtuel (recommandé) :
```bash
python -m venv .venv
.venv\Scripts\activate  # Windows
```

3. Installer les dépendances :
```bash
pip install -r requirements.txt
```

## 📊 Utilisation

### Étape 1 : Télécharger les données
- Télécharger les données SNACK depuis : https://huggingface.co/datasets/Matthijs/snacks/tree/main
- Extraire le dossier `test` dans `Projet/donnees/test/`

### Étape 2 : Exécuter le pipeline de clustering
```bash
cd sujet_tp/src
python pipeline.py
```

Cette étape va :
- Charger les images SNACK (20 catégories)
- Extraire les features (HOG et Histogrammes)
- Appliquer le clustering K-Means
- Sauvegarder les résultats dans `output/`

### Étape 3 : Lancer le dashboard
```bash
cd sujet_tp/src
streamlit run dashboard_clustering.py
```

Le dashboard permet de :
- Visualiser les clusters en 3D
- Comparer les performances des descripteurs
- Analyser les métriques de clustering

## 📈 Métriques de clustering

Le projet évalue les clusterings avec :
- Adjusted Rand Index (ARI)
- Adjusted Mutual Information (AMI)
- Homogeneity, Completeness, V-measure
- Silhouette Score
- Jaccard Index

## 🗂️ Données

Le dataset SNACK contient 20 catégories d'aliments :
- apple, banana, cake, candy, carrot
- cookie, doughnut, grape, hot dog, ice cream
- juice, muffin, orange, pineapple, popcorn
- pretzel, salad, strawberry, waffle, watermelon

## 🛠️ Technologies

- **Python 3.11**
- **Scikit-learn** : Clustering et métriques
- **OpenCV** : Traitement d'images
- **Scikit-image** : Extraction de features HOG
- **Streamlit** : Dashboard interactif
- **Plotly** : Visualisations 3D
- **TensorFlow** : (si nécessaire pour d'autres features)

## 👥 Auteurs

Projet réalisé dans le cadre du cours de Deep Learning - ET4 Polytech Paris-Saclay
