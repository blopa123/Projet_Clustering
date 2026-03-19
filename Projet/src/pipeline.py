import argparse
from sklearn.preprocessing import StandardScaler, Normalizer
import numpy as np
import os
import pandas as pd
import cv2
from sklearn.metrics import silhouette_score

try:
    from .features import (
        compute_color_histograms_hsv,
        compute_gray_histograms,
        compute_hog_descriptors,
        compute_lbp_descriptors,
        compute_resnet50_descriptors,
    )
    from .clustering import show_metric
    from .utils import conversion_3d, create_df_to_export
    from .constant import PATH_OUTPUT, PATH_DATA, REPO_ROOT, LEGACY_PATH_DATA
except ImportError:
    # Support direct execution from Projet/src.
    from features import (
        compute_color_histograms_hsv,
        compute_gray_histograms,
        compute_hog_descriptors,
        compute_lbp_descriptors,
        compute_resnet50_descriptors,
    )
    from clustering import show_metric
    from utils import conversion_3d, create_df_to_export
    from constant import PATH_OUTPUT, PATH_DATA, REPO_ROOT, LEGACY_PATH_DATA
from sklearn.cluster import KMeans as SKLearnKMeans
from sklearn.cluster import MeanShift as SKLearnMeanShift
from sklearn.cluster import estimate_bandwidth
from sklearn.cluster import SpectralClustering as SKLearnSpectralClustering
from sklearn.decomposition import PCA



def load_snack_images(data_path, img_size=(128, 128)):
    """
    Charge les images depuis le dossier de données SNACK
    Input : data_path (str) : chemin vers le dossier contenant les images
            img_size (tuple) : taille de redimensionnement des images
    Output : images_gray (list) : liste des images en niveaux de gris
             images_bgr (list) : liste des images couleur au format BGR
             labels (list) : liste des labels (noms des catégories)
             label_names (list) : noms des catégories
    """
    images_gray = []
    images_bgr = []
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
                img_bgr_resized = cv2.resize(img, img_size)
                
                images_gray.append(img_resized)
                images_bgr.append(img_bgr_resized)
                labels.append(label_idx)
    
    return np.array(images_gray), np.array(images_bgr), np.array(labels), label_names


def resolve_data_path(path_data=None):
    """Resolve data path from CLI/env/defaults with backward compatibility."""
    if path_data:
        return os.path.abspath(path_data)
    env_path = os.getenv("PATH_DATA")
    if env_path:
        return os.path.abspath(env_path)
    if os.path.isdir(PATH_DATA):
        return PATH_DATA
    if os.path.isdir(LEGACY_PATH_DATA):
        return LEGACY_PATH_DATA
    return PATH_DATA


def resolve_output_path(path_output=None):
    """Resolve output folder path from CLI/env/defaults."""
    if path_output:
        return os.path.abspath(path_output)
    env_path = os.getenv("PATH_OUTPUT")
    if env_path:
        return os.path.abspath(env_path)
    if os.path.isabs(PATH_OUTPUT):
        return PATH_OUTPUT
    return os.path.join(REPO_ROOT, PATH_OUTPUT)


def save_dataframe_multi_format(df, out_dir, base_name):
    """Export dashboard inputs to both Excel and CSV for delivery."""
    excel_path = os.path.join(out_dir, f"{base_name}.xlsx")
    csv_path = os.path.join(out_dir, f"{base_name}.csv")
    df.to_excel(excel_path, index=False)
    df.to_csv(csv_path, index=False)


def pipeline(path_data=None, path_output=None):
    data_path = resolve_data_path(path_data)
    output_path = resolve_output_path(path_output)
   
    # Chargement des données SNACK
    print("\n\n ##### Chargement des données SNACK ######")
    print(f"Chemin données: {data_path}")
    images_gray, images_bgr, labels_true, label_names = load_snack_images(data_path)
    print(f"Nombre d'images chargées : {len(images_gray)}")
    print(f"Catégories : {label_names}")
   
    print("\n\n ##### Extraction de Features ######")
    print("- calcul features hog...")
    descriptors_hog = compute_hog_descriptors(images_gray)
    print("- calcul features Histogram...")
    descriptors_hist = compute_gray_histograms(images_gray)
    print("- calcul features HSV...")
    descriptors_hsv = compute_color_histograms_hsv(images_bgr)
    print("- calcul features LBP...")
    descriptors_lbp = compute_lbp_descriptors(images_gray)
    print("- calcul features ResNet50...")
    descriptors_resnet = compute_resnet50_descriptors(images_bgr)

    # Normalisation pour KMeans : met toutes les dimensions sur une echelle comparable.
    scaler_km_hog = StandardScaler()
    scaler_km_hist = StandardScaler()
    scaler_km_hsv = StandardScaler()
    normalizer_km_resnet = Normalizer(norm="l2")
    scaler_km_resnet = StandardScaler()
    descriptors_hog_km = scaler_km_hog.fit_transform(np.array(descriptors_hog))
    descriptors_hist_km = scaler_km_hist.fit_transform(np.array(descriptors_hist))
    descriptors_hsv_km = scaler_km_hsv.fit_transform(np.array(descriptors_hsv))
    descriptors_resnet_l2 = normalizer_km_resnet.fit_transform(np.array(descriptors_resnet))
    descriptors_resnet_km_scaled = scaler_km_resnet.fit_transform(descriptors_resnet_l2)
    pca_km_resnet = PCA(n_components=0.90, svd_solver="full", random_state=0)
    descriptors_resnet_km = pca_km_resnet.fit_transform(descriptors_resnet_km_scaled)
    print(
        "ResNet KMeans preprocessing: "
        f"L2 + StandardScaler + PCA(90%) -> {descriptors_resnet_km.shape[1]} dims"
    )

    # Tester un petit voisinage autour de 20 pour KMeans.
    kmeans_candidates = sorted({18, 19, 20, 21, 22, len(label_names)})

    def _select_best_kmeans_model(X, candidates, random_state=0):
        best_model = None
        best_k = None
        best_sil = -np.inf
        n_samples = X.shape[0]

        for k in candidates:
            if k <= 1 or k >= n_samples:
                continue
            model = SKLearnKMeans(
                n_clusters=k,
                init="k-means++",
                n_init=50,
                max_iter=1000,
                algorithm="elkan",
                random_state=random_state,
            )
            labels = model.fit_predict(X)
            if len(np.unique(labels)) < 2:
                continue
            score = silhouette_score(X, labels)
            if score > best_sil:
                best_sil = score
                best_k = k
                best_model = model

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

    # Réduction de dimension pour MeanShift: forcer un espace latent compact de 20 dimensions.
    pca_target_dims = 20

    # Normalisation uniquement pour MeanShift (pas pour KMeans)
    scaler_ms_hog = StandardScaler()
    scaler_ms_hist = StandardScaler()
    scaler_ms_hsv = StandardScaler()
    scaler_ms_lbp = StandardScaler()
    scaler_ms_resnet = StandardScaler()
    # L2 pour tous les descripteurs afin d'homogeneiser les distances.
    scaler_ms_hog = Normalizer(norm="l2")
    scaler_ms_hist = Normalizer(norm="l2")
    scaler_ms_hsv = Normalizer(norm="l2")
    scaler_ms_resnet = Normalizer(norm="l2")
    descriptors_hog_ms = scaler_ms_hog.fit_transform(np.array(descriptors_hog))
    descriptors_hist_ms = scaler_ms_hist.fit_transform(np.array(descriptors_hist))
    descriptors_hsv_ms = scaler_ms_hsv.fit_transform(np.array(descriptors_hsv))
    descriptors_lbp_ms = scaler_ms_lbp.fit_transform(np.array(descriptors_lbp))
    descriptors_resnet_ms = scaler_ms_resnet.fit_transform(np.array(descriptors_resnet))

    n_comp_hog = _safe_n_components(descriptors_hog_ms, target=10)
    n_comp_hist = _safe_n_components(descriptors_hist_ms, target=10)
    n_comp_hsv = _safe_n_components(descriptors_hsv_ms, target=10)
    n_comp_lbp = _safe_n_components(descriptors_lbp_ms, target=10)
    n_comp_resnet = _safe_n_components(descriptors_resnet_ms, target=10)
    pca_hog = PCA(n_components=n_comp_hog)
    pca_hist = PCA(n_components=n_comp_hist)
    pca_hsv = PCA(n_components=n_comp_hsv)
    pca_lbp = PCA(n_components=n_comp_lbp)
    pca_resnet = PCA(n_components=n_comp_resnet)
    descriptors_hog_pca = pca_hog.fit_transform(descriptors_hog_ms)
    descriptors_hist_pca = pca_hist.fit_transform(descriptors_hist_ms)
    descriptors_hsv_pca = pca_hsv.fit_transform(descriptors_hsv_ms)
    descriptors_lbp_pca = pca_lbp.fit_transform(descriptors_lbp_ms)
    descriptors_resnet_pca = pca_resnet.fit_transform(descriptors_resnet_ms)
    def _pca_fixed_dims_for_ms(X, target_dims):
        n_components = min(target_dims, X.shape[1], max(1, X.shape[0] - 1))
        return PCA(n_components=n_components, svd_solver="full").fit_transform(X)

    descriptors_hog_pca = _pca_fixed_dims_for_ms(descriptors_hog_ms, pca_target_dims)
    descriptors_hist_pca = _pca_fixed_dims_for_ms(descriptors_hist_ms, pca_target_dims)
    descriptors_hsv_pca = _pca_fixed_dims_for_ms(descriptors_hsv_ms, pca_target_dims)
    descriptors_resnet_pca = _pca_fixed_dims_for_ms(descriptors_resnet_ms, pca_target_dims)

    n_comp_hog = descriptors_hog_pca.shape[1]
    n_comp_hist = descriptors_hist_pca.shape[1]
    n_comp_hsv = descriptors_hsv_pca.shape[1]
    n_comp_resnet = descriptors_resnet_pca.shape[1]

    print(
        "Applied PCA: "
        f"target_dims={pca_target_dims}, "
        f"HOG -> {n_comp_hog} dims, "
        f"HIST -> {n_comp_hist} dims, "
        f"HSV -> {n_comp_hsv} dims, "
        f"LBP -> {n_comp_lbp} dims, "
        f"RESNET50 -> {n_comp_resnet} dims"
    )

    number_cluster = len(label_names)  # Nombre de catégories

    # Recherche automatique de bandwidth via estimate_bandwidth (grid de quantiles)
    print("Recherche automatique de bandwidth via estimate_bandwidth (grid de quantiles)...")
    quantiles = list(np.linspace(0.005, 0.3, 30))
    results_hog = []
    results_hist = []
    results_hsv = []
    results_lbp = []
    results_resnet = []
    target = number_cluster
    for q in quantiles:
        # HOG
        try:
            bw = estimate_bandwidth(descriptors_hog_pca, quantile=q, n_samples=min(952, len(descriptors_hog_pca)))
            if bw is None or bw <= 0:
                n_hog = None
            else:
                n_hog = len(np.unique(SKLearnMeanShift(bandwidth=bw, bin_seeding=True, min_bin_freq=5, cluster_all=False).fit(descriptors_hog_pca).labels_))
        except Exception:
            bw = None
            n_hog = None
        results_hog.append((q, bw, n_hog))

        # HIST
        try:
            bw2 = estimate_bandwidth(descriptors_hist_pca, quantile=q,n_samples=len(descriptors_hog_pca))
            if bw2 is None or bw2 <= 0:
                n_hist = None
            else:
                n_hist = len(np.unique(SKLearnMeanShift(bandwidth=bw2, bin_seeding=True, min_bin_freq=5, cluster_all=False).fit(descriptors_hist_pca).labels_))
        except Exception:
            bw2 = None
            n_hist = None
        results_hist.append((q, bw2, n_hist))

        # HSV
        try:
            bw_hsv = estimate_bandwidth(descriptors_hsv_pca, quantile=q,n_samples=len(descriptors_hog_pca))
            if bw_hsv is None or bw_hsv <= 0:
                n_hsv = None
            else:
                n_hsv = len(np.unique(SKLearnMeanShift(bandwidth=bw_hsv, bin_seeding=True, min_bin_freq=5, cluster_all=False).fit(descriptors_hsv_pca).labels_))
        except Exception:
            bw_hsv = None
            n_hsv = None
        results_hsv.append((q, bw_hsv, n_hsv))

        # RESNET50
        try:
            bw3 = estimate_bandwidth(descriptors_resnet_pca, quantile=q,n_samples=len(descriptors_hog_pca))
            if bw3 is None or bw3 <= 0:
                n_resnet = None
            else:
                n_resnet = len(np.unique(SKLearnMeanShift(bandwidth=bw3, bin_seeding=True, min_bin_freq=5, cluster_all=False).fit(descriptors_resnet_pca).labels_))
        except Exception:
            bw3 = None
            n_resnet = None
        results_resnet.append((q, bw3, n_resnet))

        # LBP
        try:
            bw_lbp = estimate_bandwidth(descriptors_lbp_pca, quantile=q, n_samples=min(500, len(descriptors_lbp_pca)))
            if bw_lbp is None or bw_lbp <= 0:
                n_lbp = None
            else:
                n_lbp = len(np.unique(SKLearnMeanShift(bandwidth=bw_lbp, bin_seeding=True).fit(descriptors_lbp_pca).labels_))
        except Exception:
            bw_lbp = None
            n_lbp = None
        results_lbp.append((q, bw_lbp, n_lbp))

        print(
            f"quantile={q:.3f} -> "
            f"HOG: bw={bw}, n_clusters={n_hog} | "
            f"HIST: bw={bw2}, n_clusters={n_hist} | "
            f"HSV: bw={bw_hsv}, n_clusters={n_hsv} | "
            f"LBP: bw={bw_lbp}, n_clusters={n_lbp} | "
            f"RESNET50: bw={bw3}, n_clusters={n_resnet}"
        )

    # Choisir le bandwidth produisant un nombre de clusters le plus proche de la cible
    def _choose_best(results, target):
        # results: list of (q,bw,n)
        filtered = [r for r in results if r[1] is not None and r[2] is not None]
        if not filtered:
            return None, None, None
        # Prefer candidates with n_clusters <= target (at most target)
        le = [r for r in filtered if r[2] <= target]
        if le:
            # choose the one with largest clusters (closest to target), tie-breaker: smaller bandwidth
            best = max(le, key=lambda x: (x[2], -x[1] if x[1] is not None else 0))
            return best
        # If none <= target, fall back to the candidate closest to target
        best = min(filtered, key=lambda x: (abs(x[2] - target), x[2]))
        return best  # (q,bw,n)

    best_hog = _choose_best(results_hog, target)
    best_hist = _choose_best(results_hist, target)
    best_hsv = _choose_best(results_hsv, target)
    best_lbp = _choose_best(results_lbp, target)
    best_resnet = _choose_best(results_resnet, target)

    if best_hog[1] is None:
        print("Aucun bandwidth utile trouvé pour HOG via estimate_bandwidth; utilisation du bandwidth par défaut.")
        best_bw_hog = None
    else:
        best_bw_hog = best_hog[1]
        print(f"Choix HOG -> quantile={best_hog[0]:.3f}, bandwidth={best_bw_hog}, clusters={best_hog[2]}")

    if best_hist[1] is None:
        print("Aucun bandwidth utile trouvé pour HIST via estimate_bandwidth; utilisation du bandwidth par défaut.")
        best_bw_hist = None
    else:
        best_bw_hist = best_hist[1]
        print(f"Choix HIST -> quantile={best_hist[0]:.3f}, bandwidth={best_bw_hist}, clusters={best_hist[2]}")

    if best_hsv[1] is None:
        print("Aucun bandwidth utile trouvé pour HSV via estimate_bandwidth; utilisation du bandwidth par défaut.")
        best_bw_hsv = None
    else:
        best_bw_hsv = best_hsv[1]
        print(f"Choix HSV -> quantile={best_hsv[0]:.3f}, bandwidth={best_bw_hsv}, clusters={best_hsv[2]}")

    if best_lbp[1] is None:
        print("Aucun bandwidth utile trouvé pour LBP via estimate_bandwidth; utilisation du bandwidth par défaut.")
        best_bw_lbp = None
    else:
        best_bw_lbp = best_lbp[1]
        print(f"Choix LBP -> quantile={best_lbp[0]:.3f}, bandwidth={best_bw_lbp}, clusters={best_lbp[2]}")

    if best_resnet[1] is None:
        print("Aucun bandwidth utile trouvé pour RESNET50 via estimate_bandwidth; utilisation du bandwidth par défaut.")
        best_bw_resnet = None
    else:
        best_bw_resnet = best_resnet[1]
        print(f"Choix RESNET50 -> quantile={best_resnet[0]:.3f}, bandwidth={best_bw_resnet}, clusters={best_resnet[2]}")

    print("\n\n ##### Clustering ######")
    print(f"- test KMeans n_clusters candidats: {kmeans_candidates}")
    print("- calcul + sélection kmeans avec features HOG ...")
    kmeans_hog, k_hog, sil_hog = _select_best_kmeans_model(descriptors_hog_km, kmeans_candidates)
    print(f"  -> HOG: k={k_hog}, silhouette={sil_hog}")
    print("- calcul + sélection kmeans avec features Histogram...")
    kmeans_hist, k_hist, sil_hist = _select_best_kmeans_model(descriptors_hist_km, kmeans_candidates)
    print(f"  -> HIST: k={k_hist}, silhouette={sil_hist}")
    print("- calcul + sélection kmeans avec features HSV...")
    kmeans_hsv, k_hsv, sil_hsv = _select_best_kmeans_model(descriptors_hsv_km, kmeans_candidates)
    print(f"  -> HSV: k={k_hsv}, silhouette={sil_hsv}")
    print("- calcul + sélection kmeans avec features ResNet50...")
    kmeans_resnet, k_resnet, sil_resnet = _select_best_kmeans_model(descriptors_resnet_km, kmeans_candidates)
    print(f"  -> RESNET50: k={k_resnet}, silhouette={sil_resnet}")
    kmeans_hog = SKLearnKMeans(n_clusters=number_cluster, random_state=0)
    kmeans_hist = SKLearnKMeans(n_clusters=number_cluster, random_state=0)
    kmeans_hsv = SKLearnKMeans(n_clusters=number_cluster, random_state=0)
    kmeans_lbp = SKLearnKMeans(n_clusters=number_cluster, random_state=0)
    kmeans_resnet = SKLearnKMeans(n_clusters=number_cluster, random_state=0)

    print("- calcul kmeans avec features HOG ...")
    kmeans_hog.fit(np.array(descriptors_hog))
    print("- calcul kmeans avec features Histogram...")
    kmeans_hist.fit(np.array(descriptors_hist))
    print("- calcul kmeans avec features HSV...")
    kmeans_hsv.fit(np.array(descriptors_hsv))
    print("- calcul kmeans avec features LBP...")
    kmeans_lbp.fit(np.array(descriptors_lbp))
    print("- calcul kmeans avec features ResNet50...")
    kmeans_resnet.fit(np.array(descriptors_resnet))

    # MeanShift clustering (sur données réduites par PCA) avec les bandwidth choisis
    if best_bw_hog is not None:
        meanshift_hog = SKLearnMeanShift(bandwidth=best_bw_hog, bin_seeding=True, min_bin_freq=5, cluster_all=False)
    else:
        meanshift_hog = SKLearnMeanShift(cluster_all=False)

    if best_bw_hist is not None:
        meanshift_hist = SKLearnMeanShift(bandwidth=best_bw_hist, bin_seeding=True, min_bin_freq=5, cluster_all=False)
    else:
        meanshift_hist = SKLearnMeanShift(cluster_all=False)

    if best_bw_hsv is not None:
        meanshift_hsv = SKLearnMeanShift(bandwidth=best_bw_hsv, bin_seeding=True, min_bin_freq=5, cluster_all=False)
    else:
        meanshift_hsv = SKLearnMeanShift(cluster_all=False)

    if best_bw_lbp is not None:
        meanshift_lbp = SKLearnMeanShift(bandwidth=best_bw_lbp, bin_seeding=True)
    else:
        meanshift_lbp = SKLearnMeanShift()

    if best_bw_resnet is not None:
        meanshift_resnet = SKLearnMeanShift(bandwidth=best_bw_resnet, bin_seeding=True, min_bin_freq=5, cluster_all=False)
    else:
        meanshift_resnet = SKLearnMeanShift(cluster_all=False)

    print("- calcul meanshift avec features HOG (PCA réduit)...")
    meanshift_hog.fit(descriptors_hog_pca)
    print("- calcul meanshift avec features Histogram (PCA réduit)...")
    meanshift_hist.fit(descriptors_hist_pca)
    print("- calcul meanshift avec features HSV (PCA réduit)...")
    meanshift_hsv.fit(descriptors_hsv_pca)
    print("- calcul meanshift avec features LBP (PCA réduit)...")
    meanshift_lbp.fit(descriptors_lbp_pca)
    print("- calcul meanshift avec features ResNet50 (PCA réduit)...")
    meanshift_resnet.fit(descriptors_resnet_pca)

    # Spectral clustering (sur données réduites par PCA), même approche que MeanShift.
    spectral_hog = SKLearnSpectralClustering(n_clusters=number_cluster, affinity='nearest_neighbors', n_neighbors=10, random_state=0)
    spectral_hist = SKLearnSpectralClustering(n_clusters=number_cluster, affinity='nearest_neighbors', n_neighbors=10, random_state=0)
    spectral_hsv = SKLearnSpectralClustering(n_clusters=number_cluster, affinity='nearest_neighbors', n_neighbors=10, random_state=0)
    spectral_lbp = SKLearnSpectralClustering(n_clusters=number_cluster, affinity='nearest_neighbors', n_neighbors=10, random_state=0)
    spectral_resnet = SKLearnSpectralClustering(n_clusters=number_cluster, affinity='nearest_neighbors', n_neighbors=10, random_state=0)

    print("- calcul spectral clustering avec features HOG (PCA réduit)...")
    spectral_hog.fit(descriptors_hog_pca)
    print("- calcul spectral clustering avec features Histogram (PCA réduit)...")
    spectral_hist.fit(descriptors_hist_pca)
    print("- calcul spectral clustering avec features HSV (PCA réduit)...")
    spectral_hsv.fit(descriptors_hsv_pca)
    print("- calcul spectral clustering avec features LBP (PCA réduit)...")
    spectral_lbp.fit(descriptors_lbp_pca)
    print("- calcul spectral clustering avec features ResNet50 (PCA réduit)...")
    spectral_resnet.fit(descriptors_resnet_pca)


    print("\n\n ##### Résultat ######")
    metric_hist = show_metric(labels_true, kmeans_hist.labels_, descriptors_hist_km, bool_show=True, name_descriptor="HISTOGRAM", bool_return=True, name_model="kmeans")
    metric_hog = show_metric(labels_true, kmeans_hog.labels_, descriptors_hog_km,bool_show=True, name_descriptor="HOG", bool_return=True, name_model="kmeans")
    metric_hsv = show_metric(labels_true, kmeans_hsv.labels_, descriptors_hsv_km, bool_show=True, name_descriptor="HSV", bool_return=True, name_model="kmeans")
    metric_resnet = show_metric(labels_true, kmeans_resnet.labels_, descriptors_resnet_km, bool_show=True, name_descriptor="RESNET50", bool_return=True, name_model="kmeans")
    metric_hist = show_metric(labels_true, kmeans_hist.labels_, descriptors_hist, bool_show=True, name_descriptor="HISTOGRAM", bool_return=True, name_model="kmeans")
    metric_hog = show_metric(labels_true, kmeans_hog.labels_, descriptors_hog,bool_show=True, name_descriptor="HOG", bool_return=True, name_model="kmeans")
    metric_hsv = show_metric(labels_true, kmeans_hsv.labels_, descriptors_hsv, bool_show=True, name_descriptor="HSV", bool_return=True, name_model="kmeans")
    metric_lbp = show_metric(labels_true, kmeans_lbp.labels_, descriptors_lbp, bool_show=True, name_descriptor="LBP", bool_return=True, name_model="kmeans")
    metric_resnet = show_metric(labels_true, kmeans_resnet.labels_, descriptors_resnet, bool_show=True, name_descriptor="RESNET50", bool_return=True, name_model="kmeans")

    metric_hist_ms = show_metric(labels_true, meanshift_hist.labels_, descriptors_hist, bool_show=True, name_descriptor="HISTOGRAM", bool_return=True, name_model="meanshift")
    metric_hog_ms = show_metric(labels_true, meanshift_hog.labels_, descriptors_hog, bool_show=True, name_descriptor="HOG", bool_return=True, name_model="meanshift")
    metric_hsv_ms = show_metric(labels_true, meanshift_hsv.labels_, descriptors_hsv, bool_show=True, name_descriptor="HSV", bool_return=True, name_model="meanshift")
    metric_lbp_ms = show_metric(labels_true, meanshift_lbp.labels_, descriptors_lbp, bool_show=True, name_descriptor="LBP", bool_return=True, name_model="meanshift")
    metric_resnet_ms = show_metric(labels_true, meanshift_resnet.labels_, descriptors_resnet, bool_show=True, name_descriptor="RESNET50", bool_return=True, name_model="meanshift")

    metric_hist_sc = show_metric(labels_true, spectral_hist.labels_, descriptors_hist, bool_show=True, name_descriptor="HISTOGRAM", bool_return=True, name_model="spectralclustering")
    metric_hog_sc = show_metric(labels_true, spectral_hog.labels_, descriptors_hog, bool_show=True, name_descriptor="HOG", bool_return=True, name_model="spectralclustering")
    metric_hsv_sc = show_metric(labels_true, spectral_hsv.labels_, descriptors_hsv, bool_show=True, name_descriptor="HSV", bool_return=True, name_model="spectralclustering")
    metric_lbp_sc = show_metric(labels_true, spectral_lbp.labels_, descriptors_lbp, bool_show=True, name_descriptor="LBP", bool_return=True, name_model="spectralclustering")
    metric_resnet_sc = show_metric(labels_true, spectral_resnet.labels_, descriptors_resnet, bool_show=True, name_descriptor="RESNET50", bool_return=True, name_model="spectralclustering")


    print("- export des données vers le dashboard")
    # conversion des données vers le format du dashboard
    list_dict = [
        metric_hist,
        metric_hog,
        metric_hsv,
        metric_lbp,
        metric_resnet,
        metric_hist_ms,
        metric_hog_ms,
        metric_hsv_ms,
        metric_lbp_ms,
        metric_resnet_ms,
        metric_hist_sc,
        metric_hog_sc,
        metric_hsv_sc,
        metric_lbp_sc,
        metric_resnet_sc,
    ]
    df_metric = pd.DataFrame(list_dict)
    
    # Normalisation des données
    scaler = StandardScaler()
    descriptors_hist_norm = scaler.fit_transform(descriptors_hist)
    descriptors_hog_norm = scaler.fit_transform(descriptors_hog)
    descriptors_hsv_norm = scaler.fit_transform(descriptors_hsv)
    descriptors_lbp_norm = scaler.fit_transform(descriptors_lbp)
    descriptors_resnet_norm = scaler.fit_transform(descriptors_resnet)

    #conversion vers un format 3D pour la visualisation
    x_3d_hist = conversion_3d(descriptors_hist_norm)
    x_3d_hog = conversion_3d(descriptors_hog_norm)
    x_3d_hsv = conversion_3d(descriptors_hsv_norm)
    x_3d_lbp = conversion_3d(descriptors_lbp_norm)
    x_3d_resnet = conversion_3d(descriptors_resnet_norm)

    # création des dataframe pour la sauvegarde des données pour la visualisation
    df_hist = create_df_to_export(x_3d_hist, labels_true, kmeans_hist.labels_)
    df_hog = create_df_to_export(x_3d_hog, labels_true, kmeans_hog.labels_)
    df_hsv = create_df_to_export(x_3d_hsv, labels_true, kmeans_hsv.labels_)
    df_lbp = create_df_to_export(x_3d_lbp, labels_true, kmeans_lbp.labels_)
    df_resnet = create_df_to_export(x_3d_resnet, labels_true, kmeans_resnet.labels_)

    # Dataframes for MeanShift
    df_hist_meanshift = create_df_to_export(x_3d_hist, labels_true, meanshift_hist.labels_)
    df_hog_meanshift = create_df_to_export(x_3d_hog, labels_true, meanshift_hog.labels_)
    df_hsv_meanshift = create_df_to_export(x_3d_hsv, labels_true, meanshift_hsv.labels_)
    df_lbp_meanshift = create_df_to_export(x_3d_lbp, labels_true, meanshift_lbp.labels_)
    df_resnet_meanshift = create_df_to_export(x_3d_resnet, labels_true, meanshift_resnet.labels_)

    # Dataframes for SpectralClustering
    df_hist_spectral = create_df_to_export(x_3d_hist, labels_true, spectral_hist.labels_)
    df_hog_spectral = create_df_to_export(x_3d_hog, labels_true, spectral_hog.labels_)
    df_hsv_spectral = create_df_to_export(x_3d_hsv, labels_true, spectral_hsv.labels_)
    df_lbp_spectral = create_df_to_export(x_3d_lbp, labels_true, spectral_lbp.labels_)
    df_resnet_spectral = create_df_to_export(x_3d_resnet, labels_true, spectral_resnet.labels_)

    # Vérifie si le dossier existe déjà
    if not os.path.exists(output_path):
        # Crée le dossier
        os.makedirs(output_path)

    # sauvegarde des données
    save_dataframe_multi_format(df_hist, output_path, "save_clustering_hist_kmeans")
    save_dataframe_multi_format(df_hog, output_path, "save_clustering_hog_kmeans")
    save_dataframe_multi_format(df_hsv, output_path, "save_clustering_hsv_kmeans")
    save_dataframe_multi_format(df_lbp, output_path, "save_clustering_lbp_kmeans")
    save_dataframe_multi_format(df_resnet, output_path, "save_clustering_resnet_kmeans")
    save_dataframe_multi_format(df_hist_meanshift, output_path, "save_clustering_hist_meanshift")
    save_dataframe_multi_format(df_hog_meanshift, output_path, "save_clustering_hog_meanshift")
    save_dataframe_multi_format(df_hsv_meanshift, output_path, "save_clustering_hsv_meanshift")
    save_dataframe_multi_format(df_lbp_meanshift, output_path, "save_clustering_lbp_meanshift")
    save_dataframe_multi_format(df_resnet_meanshift, output_path, "save_clustering_resnet_meanshift")
    save_dataframe_multi_format(df_hist_spectral, output_path, "save_clustering_hist_spectralclustering")
    save_dataframe_multi_format(df_hog_spectral, output_path, "save_clustering_hog_spectralclustering")
    save_dataframe_multi_format(df_hsv_spectral, output_path, "save_clustering_hsv_spectralclustering")
    save_dataframe_multi_format(df_lbp_spectral, output_path, "save_clustering_lbp_spectralclustering")
    save_dataframe_multi_format(df_resnet_spectral, output_path, "save_clustering_resnet_spectralclustering")
    save_dataframe_multi_format(df_metric, output_path, "save_metric")
    print(f"Résultats exportés dans: {output_path}")
    print("Fin. \n\n Pour avoir la visualisation dashboard, veuillez lancer la commande : python dashboard.py --path_data chemin_vers_les_analyse_ia")


def parse_args():
    parser = argparse.ArgumentParser(description="Pipeline IA de clustering SNACK")
    parser.add_argument(
        "--path_data",
        type=str,
        default=None,
        help="Chemin vers le dossier de données (par défaut: data/test)",
    )
    parser.add_argument(
        "--path_output",
        type=str,
        default=None,
        help="Chemin vers le dossier de sortie (par défaut: output)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    pipeline(path_data=args.path_data, path_output=args.path_output)