from sklearn.preprocessing import StandardScaler
import os
import pandas as pd
from sklearn import datasets
import cv2

from features import *
from clustering import *
from utils import *
from constant import PATH_OUTPUT, MODEL_CLUSTERING, PATH_DATA


def load_snack_images(data_path, img_size=(64, 64)):
    """
    Charge les images depuis le dossier de données SNACK
    Input : data_path (str) : chemin vers le dossier contenant les images
            img_size (tuple) : taille de redimensionnement des images
    Output : images (list) : liste des images
             labels (list) : liste des labels (noms des catégories)
             label_names (list) : noms des catégories
    """
    images = []
    labels = []
    label_names = []
    
    # Parcourir tous les sous-dossiers (catégories)
    categories = sorted([d for d in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, d))])
    
    for label_idx, category in enumerate(categories):
        category_path = os.path.join(data_path, category)
        label_names.append(category)
        
        # Parcourir toutes les images dans la catégorie
        for img_name in os.listdir(category_path):
            img_path = os.path.join(category_path, img_name)
            
            # Charger l'image en couleur puis convertir en niveaux de gris
            img = cv2.imread(img_path)
            if img is not None:
                # Convertir en niveaux de gris
                img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                # Redimensionner
                img_resized = cv2.resize(img_gray, img_size)
                
                images.append(img_resized)
                labels.append(label_idx)
    
    return np.array(images), np.array(labels), label_names


def pipeline():
   
    # Chargement des données SNACK
    print("\n\n ##### Chargement des données SNACK ######")
    images, labels_true, label_names = load_snack_images(PATH_DATA)
    print(f"Nombre d'images chargées : {len(images)}")
    print(f"Catégories : {label_names}")
   
    print("\n\n ##### Extraction de Features ######")
    print("- calcul features hog...")
    descriptors_hog = compute_hog_descriptors(images)
    print("- calcul features Histogram...")
    descriptors_hist = compute_gray_histograms(images)


    print("\n\n ##### Clustering ######")
    number_cluster = len(label_names)  # Nombre de catégories
    kmeans_hog = KMeans(n_clusters=number_cluster)
    kmeans_hist = KMeans(n_clusters=number_cluster)

    print("- calcul kmeans avec features HOG ...")
    kmeans_hog.fit(np.array(descriptors_hog))
    print("- calcul kmeans avec features Histogram...")
    kmeans_hist.fit(np.array(descriptors_hist))


    print("\n\n ##### Résultat ######")
    metric_hist = show_metric(labels_true, kmeans_hist.labels_, descriptors_hist, bool_show=True, name_descriptor="HISTOGRAM", bool_return=True)
    metric_hog = show_metric(labels_true, kmeans_hog.labels_, descriptors_hog,bool_show=True, name_descriptor="HOG", bool_return=True)


    print("- export des données vers le dashboard")
    # conversion des données vers le format du dashboard
    list_dict = [metric_hist,metric_hog]
    df_metric = pd.DataFrame(list_dict)
    
    # Normalisation des données
    scaler = StandardScaler()
    descriptors_hist_norm = scaler.fit_transform(descriptors_hist)
    descriptors_hog_norm = scaler.fit_transform(descriptors_hog)

    #conversion vers un format 3D pour la visualisation
    x_3d_hist = conversion_3d(descriptors_hist_norm)
    x_3d_hog = conversion_3d(descriptors_hog_norm)

    # création des dataframe pour la sauvegarde des données pour la visualisation
    df_hist = create_df_to_export(x_3d_hist, labels_true, kmeans_hist.labels_)
    df_hog = create_df_to_export(x_3d_hog, labels_true, kmeans_hog.labels_)

    # Vérifie si le dossier existe déjà
    if not os.path.exists(PATH_OUTPUT):
        # Crée le dossier
        os.makedirs(PATH_OUTPUT)

    # sauvegarde des données
    df_hist.to_excel(PATH_OUTPUT+"/save_clustering_hist_kmeans.xlsx")
    df_hog.to_excel(PATH_OUTPUT+"/save_clustering_hog_kmeans.xlsx")
    df_metric.to_excel(PATH_OUTPUT+"/save_metric.xlsx")
    print("Fin. \n\n Pour avoir la visualisation dashboard, veuillez lancer la commande : streamlit run dashboard_clustering.py")


if __name__ == "__main__":
    pipeline()