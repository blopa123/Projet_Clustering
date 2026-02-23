from sklearn.metrics import (
    adjusted_mutual_info_score,
    homogeneity_completeness_v_measure,
    jaccard_score,
    silhouette_score,
    adjusted_rand_score,
)
import pandas as pd


def show_metric(labels_true, labels_pred, descriptors=None, bool_show=True, name_descriptor="", bool_return=False, name_model="kmeans"):
    """
    Fonction d'affichage et création des métrique pour le clustering.
    Input :
    - labels_true : étiquettes réelles des données
    - labels_pred : étiquettes prédites des données
    - descriptors : ensemble de descripteurs utilisé pour le clustering
    - bool_return : booléen indiquant si les métriques doivent être retournées ou affichées
    - name_descriptor : nom de l'ensemble de descripteurs utilisé pour le clustering
    - name_model : nom du modèle de clustering utilisé
    - bool_show : booléen indiquant si les métriques doivent être affichées ou non

    Output :
    - dictionnaire contenant les métriques d'évaluation des clusters
    """
    homogeneity, completeness, v_measure = homogeneity_completeness_v_measure(labels_true, labels_pred)
    jaccard = jaccard_score(labels_true, labels_pred, average='macro')
    ami = adjusted_mutual_info_score(labels_true, labels_pred)
    silhouette = None
    try:
        if descriptors is not None:
            silhouette = float(silhouette_score(descriptors, labels_pred))
    except Exception:
        silhouette = None
    ari = adjusted_rand_score(labels_true, labels_pred)
    # Affichons les résultats
    if bool_show :
        print(f"########## Métrique descripteur : {name_descriptor}")
        print(f"Adjusted Rand Index: {ari}")
        print(f"Jaccard Index: {jaccard}")
        print(f"Homogeneity: {homogeneity}")
        print(f"Completeness: {completeness}")
        print(f"V-measure: {v_measure}")
        print(f"Silhouette Score: {silhouette}")
        print(f"Adjusted Mutual Information: {ami}")
    if bool_return:
        return {"ami":ami,
                "ari":ari, 
                "silhouette":silhouette,
                "homogeneity":homogeneity,
                "completeness":completeness,
                "v_measure":v_measure, 
                "jaccard":jaccard,
               "descriptor":name_descriptor,
               "name_model":name_model}