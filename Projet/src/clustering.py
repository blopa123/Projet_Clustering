import numpy as np
from sklearn.cluster import KMeans as SKLearnKMeans
from sklearn.cluster import MeanShift as SKLearnMeanShift
from sklearn.cluster import estimate_bandwidth
from sklearn.metrics import (
    adjusted_mutual_info_score,
    homogeneity_completeness_v_measure,
    jaccard_score,
    silhouette_score,
    davies_bouldin_score,
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
    davies_bouldin = None
    try:
        n_labels = len(set(labels_pred))
        n_samples = len(labels_pred)
        if descriptors is not None and 1 < n_labels < n_samples:
            silhouette = float(silhouette_score(descriptors, labels_pred))
            davies_bouldin = float(davies_bouldin_score(descriptors, labels_pred))
    except Exception:
        silhouette = None
        davies_bouldin = None
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
        print(f"Davies-Bouldin Index: {davies_bouldin}")
        print(f"Adjusted Mutual Information: {ami}")
    if bool_return:
        return {"ami":ami,
                "ari":ari, 
                "silhouette":silhouette,
            "davies_bouldin":davies_bouldin,
                "homogeneity":homogeneity,
                "completeness":completeness,
                "v_measure":v_measure, 
                "jaccard":jaccard,
               "descriptor":name_descriptor,
               "name_model":name_model}


def select_best_kmeans(X, candidates, random_state=0):
    """
    Sélectionne le meilleur modèle KMeans en optimisant le score de silhouette.
    
    Parcourt une liste de nombres de clusters (k) candidats, entraîne un modèle 
    pour chaque k, et retient celui qui offre la meilleure séparation (silhouette).
    
    Args:
        X (np.ndarray): Matrice de caractéristiques (samples, features).
        candidates (list[int]): Liste des valeurs de k à tester.
        random_state (int): Graine pour la reproductibilité.
        
    Returns:
        tuple: (best_model, best_k, best_sil)
            - best_model: L'instance de SKLearnKMeans la plus performante.
            - best_k (int): Le nombre de clusters optimal trouvé.
            - best_sil (float): Le score de silhouette associé (ou None si fallback).
    """
    best_model = None
    best_k = None
    best_sil = -np.inf
    n_samples = X.shape[0]

    for k in candidates:
        # Validation : k doit être cohérent avec la taille du dataset
        if k <= 1 or k >= n_samples:
            continue
            
        model = SKLearnKMeans(
            n_clusters=k,
            init="k-means++",
            n_init=50,      # Augmenté pour éviter les minima locaux
            max_iter=1000,
            algorithm="elkan", # Plus efficace sur les datasets denses
            random_state=random_state,
        )
        
        labels = model.fit_predict(X)
        
        # Le score de silhouette nécessite au moins 2 clusters distincts
        if len(np.unique(labels)) < 2:
            continue
            
        score = silhouette_score(X, labels)
        
        if score > best_sil:
            best_sil = score
            best_k = k
            best_model = model

    # --- Stratégie de Fallback ---
    # Si aucun modèle n'a pu être validé, on force un modèle par défaut
    if best_model is None:
        fallback_k = min(20, max(2, n_samples - 1))
        best_model = SKLearnKMeans(
            n_clusters=fallback_k,
            init="k-means++",
            n_init=50,
            max_iter=1000,
            algorithm="elkan",
            random_state=random_state,
        )
        best_model.fit(X)
        best_k = fallback_k
        best_sil = None

    return best_model, best_k, best_sil


def compute_kmeans_silhouette_tracking(X, candidates, random_state=0):
    """
    Calcule l'evolution du score silhouette selon plusieurs valeurs de k pour KMeans.

    Args:
        X (np.ndarray): Matrice de caracteristiques (samples, features).
        candidates (list[int]): Valeurs de k a evaluer.
        random_state (int): Graine pour la reproductibilite.

    Returns:
        pd.DataFrame: Colonnes [k, silhouette]. Les cas invalides sont notes en NaN.
    """
    data = []
    n_samples = X.shape[0]

    for k in sorted(set(candidates)):
        if k <= 1 or k >= n_samples:
            data.append({"k": int(k), "silhouette": np.nan})
            continue

        try:
            model = SKLearnKMeans(
                n_clusters=int(k),
                init="k-means++",
                n_init=50,
                max_iter=1000,
                algorithm="elkan",
                random_state=random_state,
            )
            labels = model.fit_predict(X)
            n_labels = len(np.unique(labels))
            if n_labels <= 1 or n_labels >= n_samples:
                score = np.nan
            else:
                score = float(silhouette_score(X, labels))
            data.append({"k": int(k), "silhouette": score})
        except Exception:
            data.append({"k": int(k), "silhouette": np.nan})

    return pd.DataFrame(data)


def compute_silhouette_at_fixed_k(X, k_values=(5, 10, 15, 20, 25), random_state=0):
    """
    Calcule le silhouette score exactement pour k = 5, 10, 15, 20, 25.
    Utilise pour le graphe de suivi standardise du dashboard.

    Args:
        X (np.ndarray): Matrice de features normalisee.
        k_values (tuple): Valeurs de k a evaluer.
        random_state (int): Graine pour la reproductibilite.

    Returns:
        pd.DataFrame: Colonnes [k, silhouette].
    """
    data = []
    n_samples = X.shape[0]

    for k in k_values:
        if k <= 1 or k >= n_samples:
            data.append({"k": int(k), "silhouette": np.nan})
            continue

        try:
            model = SKLearnKMeans(
                n_clusters=int(k),
                init="k-means++",
                n_init=10,
                max_iter=300,
                random_state=random_state,
            )
            labels = model.fit_predict(X)
            n_labels = len(np.unique(labels))
            if 1 < n_labels < n_samples:
                score = float(silhouette_score(X, labels))
            else:
                score = np.nan
        except Exception:
            score = np.nan

        data.append({"k": int(k), "silhouette": score})

    return pd.DataFrame(data)


def _choose_best_bandwidth_result(results, target):
    """
    Fonction utilitaire pour sélectionner la meilleure bande passante (bandwidth).
    
    Priorité 1 : Le résultat qui se rapproche le plus de 'target' sans le dépasser.
    Priorité 2 : Si tous dépassent, celui qui est le plus proche de la cible.
    
    Args:
        results (list[tuple]): Liste de (quantile, bandwidth, n_clusters, silhouette).
        target (int): Nombre de clusters idéal recherché.
        
    Returns:
        tuple: Le meilleur triplet (quantile, bandwidth, n_clusters) trouvé.
    """
    # Nettoyage des résultats invalides (erreurs lors de l'estimation)
    filtered = [r for r in results if r[1] is not None and r[2] is not None]
    if not filtered:
        return None, None, None
    
    # Cas A : On cherche le maximum de clusters qui reste <= à notre cible
    le = [r for r in filtered if r[2] <= target]
    if le:
        # On trie par n_clusters (max) puis par bandwidth (pour stabilité)
        return max(le, key=lambda x: (x[2], -x[1] if x[1] is not None else 0))
    
    # Cas B : Aucun résultat n'est inférieur à la cible, on prend le plus proche
    return min(filtered, key=lambda x: (abs(x[2] - target), x[2]))


def tune_meanshift_bandwidth(
    descriptors_hog_pca,
    descriptors_hist_pca,
    descriptors_hsv_pca,
    descriptors_lbp_pca,
    descriptors_resnet_pca,
    target,
    n_samples_cap=952,
    verbose=False,
):
    """
    Optimise les paramètres de MeanShift pour chaque type de descripteur.
    
    Cette fonction effectue une recherche par grille (grid search) sur les quantiles 
    utilisés pour estimer la 'bandwidth'. Chaque descripteur possède sa propre 
    plage de quantiles optimisée expérimentalement.
    
    Args:
        descriptors_..._pca (np.ndarray): Caractéristiques réduites par PCA.
        target (int): Nombre de clusters visé (généralement len(categories)).
        n_samples_cap (int): Limite de samples pour l'estimation de bandwidth (vitesse).
        verbose (bool): Si True, affiche la progression dans la console.
        
    Returns:
        dict: Contient les meilleurs résultats, bandwidths et traces de tuning par descripteur.
    """
    # Définition des plages de recherche spécifiques à la nature des descripteurs
    quantiles_hog = list(np.linspace(0.0015, 0.0045, 30))
    quantiles_hist = list(np.linspace(0.001, 0.01, 30))
    quantiles_hsv = list(np.linspace(0.001, 0.01, 30))
    quantiles_resnet = list(np.linspace(0.005, 0.3, 30))
    quantiles_lbp = list(np.linspace(0.1115, 0.1125, 20))

    results_hog, results_hist, results_hsv, results_lbp, results_resnet = [], [], [], [], []

    # --- Section LBP (Traitement séparé car plage étroite) ---
    if verbose: print("Recherche bandwidth LBP...")
    for q_lbp in quantiles_lbp:
        try:
            bw_lbp_candidate = estimate_bandwidth(
                descriptors_lbp_pca,
                quantile=q_lbp,
                n_samples=min(n_samples_cap, len(descriptors_lbp_pca)),
            )
            if bw_lbp_candidate is None or bw_lbp_candidate <= 0:
                n_lbp_candidate = None
                sil_lbp_candidate = np.nan
            else:
                # Test réel du clustering pour compter les clusters générés
                ms = SKLearnMeanShift(bandwidth=bw_lbp_candidate, bin_seeding=False, cluster_all=True)
                labels = ms.fit(descriptors_lbp_pca).labels_
                n_lbp_candidate = len(np.unique(labels))
                if 1 < n_lbp_candidate < len(descriptors_lbp_pca):
                    sil_lbp_candidate = float(silhouette_score(descriptors_lbp_pca, labels))
                else:
                    sil_lbp_candidate = np.nan
        except Exception:
            bw_lbp_candidate, n_lbp_candidate, sil_lbp_candidate = None, None, np.nan

        results_lbp.append((q_lbp, bw_lbp_candidate, n_lbp_candidate, sil_lbp_candidate))

    # --- Section Autres Descripteurs (Boucle groupée sur quantiles triés) ---
    all_quantiles = sorted(set(quantiles_hog + quantiles_hist + quantiles_hsv + quantiles_resnet))

    for q in all_quantiles:
        # Traitement HOG
        if q in quantiles_hog:
            try:
                bw = estimate_bandwidth(descriptors_hog_pca, quantile=q, n_samples=min(n_samples_cap, len(descriptors_hog_pca)))
                ms = SKLearnMeanShift(bandwidth=bw, bin_seeding=True, min_bin_freq=5, cluster_all=True)
                labels = ms.fit(descriptors_hog_pca).labels_ if bw and bw > 0 else None
                n_hog = len(np.unique(labels)) if labels is not None else None
                sil_hog = float(silhouette_score(descriptors_hog_pca, labels)) if labels is not None and 1 < n_hog < len(descriptors_hog_pca) else np.nan
            except Exception:
                bw, n_hog, sil_hog = None, None, np.nan
            results_hog.append((q, bw, n_hog, sil_hog))

        # Traitement Histogrammes Gris
        if q in quantiles_hist:
            try:
                bw2 = estimate_bandwidth(descriptors_hist_pca, quantile=q, n_samples=min(n_samples_cap, len(descriptors_hist_pca)))
                ms = SKLearnMeanShift(bandwidth=bw2, bin_seeding=True, min_bin_freq=3, cluster_all=False)
                labels = ms.fit(descriptors_hist_pca).labels_ if bw2 and bw2 > 0 else None
                n_hist = len(np.unique(labels)) if labels is not None else None
                sil_hist = float(silhouette_score(descriptors_hist_pca, labels)) if labels is not None and 1 < n_hist < len(descriptors_hist_pca) else np.nan
            except Exception:
                bw2, n_hist, sil_hist = None, None, np.nan
            results_hist.append((q, bw2, n_hist, sil_hist))

        # Traitement Histogrammes HSV
        if q in quantiles_hsv:
            try:
                bw_hsv = estimate_bandwidth(descriptors_hsv_pca, quantile=q, n_samples=min(n_samples_cap, len(descriptors_hsv_pca)))
                ms = SKLearnMeanShift(bandwidth=bw_hsv, bin_seeding=True, min_bin_freq=3, cluster_all=False)
                labels = ms.fit(descriptors_hsv_pca).labels_ if bw_hsv and bw_hsv > 0 else None
                n_hsv = len(np.unique(labels)) if labels is not None else None
                sil_hsv = float(silhouette_score(descriptors_hsv_pca, labels)) if labels is not None and 1 < n_hsv < len(descriptors_hsv_pca) else np.nan
            except Exception:
                bw_hsv, n_hsv, sil_hsv = None, None, np.nan
            results_hsv.append((q, bw_hsv, n_hsv, sil_hsv))

        # Traitement ResNet50
        if q in quantiles_resnet:
            try:
                bw3 = estimate_bandwidth(descriptors_resnet_pca, quantile=q, n_samples=min(n_samples_cap, len(descriptors_resnet_pca)))
                ms = SKLearnMeanShift(bandwidth=bw3, bin_seeding=True, min_bin_freq=5, cluster_all=False)
                labels = ms.fit(descriptors_resnet_pca).labels_ if bw3 and bw3 > 0 else None
                n_resnet = len(np.unique(labels)) if labels is not None else None
                sil_resnet = float(silhouette_score(descriptors_resnet_pca, labels)) if labels is not None and 1 < n_resnet < len(descriptors_resnet_pca) else np.nan
            except Exception:
                bw3, n_resnet, sil_resnet = None, None, np.nan
            results_resnet.append((q, bw3, n_resnet, sil_resnet))

    # --- Sélection finale des meilleurs choix ---
    best_hog = _choose_best_bandwidth_result(results_hog, target)
    best_hist = _choose_best_bandwidth_result(results_hist, target)
    best_hsv = _choose_best_bandwidth_result(results_hsv, target)
    best_lbp = _choose_best_bandwidth_result(results_lbp, target)
    best_resnet = _choose_best_bandwidth_result(results_resnet, target)

    return {
        "best_hog": best_hog,
        "best_hist": best_hist,
        "best_hsv": best_hsv,
        "best_lbp": best_lbp,
        "best_resnet": best_resnet,
        "best_bw": {
            "hog": best_hog[1],
            "hist": best_hist[1],
            "hsv": best_hsv[1],
            "lbp": best_lbp[1],
            "resnet": best_resnet[1],
        },
        "results": {
            "hog": results_hog,
            "hist": results_hist,
            "hsv": results_hsv,
            "lbp": results_lbp,
            "resnet": results_resnet,
        },
    }
