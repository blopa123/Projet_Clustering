import argparse
from sklearn.preprocessing import StandardScaler, Normalizer, RobustScaler
import numpy as np
import os
import pandas as pd
import cv2

try:
    from .features import (
        compute_color_histograms_hsv,
        compute_gray_histograms,
        compute_hog_descriptors,
        compute_lbp_descriptors,
        compute_resnet50_descriptors,
    )
    from .clustering import (
        show_metric,
        select_best_kmeans,
        tune_meanshift_bandwidth,
        compute_silhouette_at_fixed_k,
    )
    from .utils import (
        conversion_3d,
        create_df_to_export,
        resolve_data_path,
        resolve_output_path,
        save_dataframe_multi_format,
    )
except ImportError:
    from features import (
        compute_color_histograms_hsv,
        compute_gray_histograms,
        compute_hog_descriptors,
        compute_lbp_descriptors,
        compute_resnet50_descriptors,
    )
    from clustering import (
        show_metric,
        select_best_kmeans,
        tune_meanshift_bandwidth,
        compute_silhouette_at_fixed_k,
    )
    from utils import (
        conversion_3d,
        create_df_to_export,
        resolve_data_path,
        resolve_output_path,
        save_dataframe_multi_format,
    )

from sklearn.cluster import MeanShift as SKLearnMeanShift
from sklearn.cluster import SpectralClustering as SKLearnSpectralClustering
from sklearn.decomposition import PCA


def load_snack_images(data_path, img_size=(128, 128)):
    """
    Charge les images depuis le dossier de données SNACK.
    Input : data_path (str) : chemin vers le dossier contenant les images
            img_size (tuple) : taille de redimensionnement des images
    Output : images_gray (array) : images en niveaux de gris
             images_bgr (array)  : images couleur BGR
             labels (array)      : labels numériques des catégories
             label_names (list)  : noms des catégories
    """
    images_gray = []
    images_bgr = []
    labels = []
    label_names = []

    categories = sorted([
        d for d in os.listdir(data_path)
        if os.path.isdir(os.path.join(data_path, d))
    ])

    for label_idx, category in enumerate(categories):
        category_path = os.path.join(data_path, category)
        label_names.append(category)

        for img_name in os.listdir(category_path):
            img_path = os.path.join(category_path, img_name)
            img = cv2.imread(img_path)
            if img is not None:
                img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                img_resized = cv2.resize(img_gray, img_size)
                img_bgr_resized = cv2.resize(img, img_size)
                images_gray.append(img_resized)
                images_bgr.append(img_bgr_resized)
                labels.append(label_idx)

    return np.array(images_gray), np.array(images_bgr), np.array(labels), label_names


def pipeline(path_data=None, path_output=None):
    data_path = resolve_data_path(path_data)
    output_path = resolve_output_path(path_output)

    # ──────────────────────────────────────────────
    # 1. CHARGEMENT DES DONNÉES
    # ──────────────────────────────────────────────
    print("\n\n ##### Chargement des données SNACK ######")
    print(f"Chemin données: {data_path}")
    images_gray, images_bgr, labels_true, label_names = load_snack_images(data_path)
    print(f"Nombre d'images chargées : {len(images_gray)}")
    print(f"Catégories : {label_names}")
    number_cluster = len(label_names)

    # ──────────────────────────────────────────────
    # 2. EXTRACTION DE FEATURES
    # ──────────────────────────────────────────────
    print("\n\n ##### Extraction de Features ######")
    print("- calcul features HOG...")
    descriptors_hog = compute_hog_descriptors(images_gray)
    print("- calcul features Histogram...")
    descriptors_hist = compute_gray_histograms(images_gray)
    print("- calcul features HSV...")
    descriptors_hsv = compute_color_histograms_hsv(images_bgr)
    print("- calcul features LBP...")
    descriptors_lbp = compute_lbp_descriptors(images_gray)
    print("- calcul features ResNet50...")
    descriptors_resnet = compute_resnet50_descriptors(images_bgr, layer_name="conv5_block3_out")

    # ──────────────────────────────────────────────
    # 3. NORMALISATION POUR KMEANS
    # ──────────────────────────────────────────────
    scaler_km_hog = Normalizer(norm="l2")
    scaler_km_hist = StandardScaler()
    scaler_km_hsv = StandardScaler()
    scaler_km_lbp = StandardScaler()
    normalizer_km_resnet = Normalizer(norm="l2")
    scaler_km_resnet = StandardScaler()

    descriptors_hog_km = scaler_km_hog.fit_transform(np.array(descriptors_hog))
    descriptors_hist_km = scaler_km_hist.fit_transform(np.array(descriptors_hist))
    descriptors_hsv_km = scaler_km_hsv.fit_transform(np.array(descriptors_hsv))
    descriptors_lbp_km = scaler_km_lbp.fit_transform(np.array(descriptors_lbp))
    descriptors_resnet_l2 = normalizer_km_resnet.fit_transform(np.array(descriptors_resnet))
    descriptors_resnet_km_scaled = scaler_km_resnet.fit_transform(descriptors_resnet_l2)
    pca_km_resnet = PCA(n_components=0.90, svd_solver="full", random_state=0)
    descriptors_resnet_km = pca_km_resnet.fit_transform(descriptors_resnet_km_scaled)
    print(
        "ResNet KMeans preprocessing: "
        f"L2 + StandardScaler + PCA(90%) -> {descriptors_resnet_km.shape[1]} dims"
    )

    # ──────────────────────────────────────────────
    # 4. NORMALISATION + PCA POUR MEANSHIFT
    # ──────────────────────────────────────────────
    pca_target_dims = 20

    scaler_ms_hog = StandardScaler()
    scaler_ms_hist = Normalizer(norm="l2")
    scaler_ms_hsv = Normalizer(norm="l2")
    scaler_ms_lbp = Normalizer(norm="l2")
    scaler_ms_resnet = Normalizer(norm="l2")

    descriptors_hog_ms = scaler_ms_hog.fit_transform(np.array(descriptors_hog))
    descriptors_hist_ms = scaler_ms_hist.fit_transform(np.array(descriptors_hist))
    descriptors_hsv_ms = scaler_ms_hsv.fit_transform(np.array(descriptors_hsv))
    descriptors_lbp_ms = scaler_ms_lbp.fit_transform(np.array(descriptors_lbp))
    descriptors_resnet_ms = scaler_ms_resnet.fit_transform(np.array(descriptors_resnet))
    descriptors_resnet_sc = PCA(n_components=0.95, svd_solver="full", random_state=0).fit_transform(descriptors_resnet_ms)

    def _pca_fixed_dims_for_ms(X, target_dims):
        n_components = min(target_dims, X.shape[1], max(1, X.shape[0] - 1))
        return PCA(n_components=n_components, svd_solver="full").fit_transform(X)

    # PCA individuels par descripteur pour MeanShift
    pca_dims_ms = {
        "hog":    10,
        "hist":   13,
        "hsv":    15,
        "lbp":    10,
        "resnet": 20,
    }

    descriptors_hog_pca    = _pca_fixed_dims_for_ms(descriptors_hog_ms,    pca_dims_ms["hog"])
    descriptors_hist_pca   = _pca_fixed_dims_for_ms(descriptors_hist_ms,   pca_dims_ms["hist"])
    descriptors_hsv_pca    = _pca_fixed_dims_for_ms(descriptors_hsv_ms,    pca_dims_ms["hsv"])
    descriptors_lbp_pca    = _pca_fixed_dims_for_ms(descriptors_lbp_ms,    pca_dims_ms["lbp"])
    descriptors_resnet_pca = _pca_fixed_dims_for_ms(descriptors_resnet_ms, pca_dims_ms["resnet"])

    print(
        "Applied PCA (MeanShift dims): "
        f"HOG -> {descriptors_hog_pca.shape[1]} dims, "
        f"HIST -> {descriptors_hist_pca.shape[1]} dims, "
        f"HSV -> {descriptors_hsv_pca.shape[1]} dims, "
        f"LBP -> {descriptors_lbp_pca.shape[1]} dims, "
        f"RESNET50 -> {descriptors_resnet_pca.shape[1]} dims"
    )

    descriptor_key_map = {
        "HOG": "hog",
        "HISTOGRAM": "hist",
        "HSV": "hsv",
        "LBP": "lbp",
        "RESNET50": "resnet",
    }

    # ──────────────────────────────────────────────
    # 5. KMEANS
    # ──────────────────────────────────────────────
    print("\n\n ##### Clustering KMeans ######")

    # Candidates globaux par défaut
    kmeans_candidates_default = sorted({15, 17, 18, 19, 20, 21, 22, 25, number_cluster})
    # HOG/HIST 
    kmeans_candidates_hog_hist = sorted({15, 17, 18, 19, 20, 21, 22, 25, 30, number_cluster})
    # ResNet 
    kmeans_candidates_resnet = sorted({15, 17, 18, 19, 20, 21, 22, 25, 30, 35, number_cluster})

    print(f"- test KMeans n_clusters candidats (défaut): {kmeans_candidates_default}")
    print(f"- test KMeans candidats HOG/HIST: {kmeans_candidates_hog_hist}")
    print(f"- test KMeans candidats RESNET: {kmeans_candidates_resnet}")
    print("- calcul + sélection kmeans avec features HOG ...")
    kmeans_hog, k_hog, sil_hog = select_best_kmeans(descriptors_hog_km, kmeans_candidates_hog_hist)
    print(f"  -> HOG: k={k_hog}, silhouette={sil_hog}")
    show_metric(labels_true, kmeans_hog.labels_, descriptors_hog_km, bool_show=True, name_descriptor="HOG", bool_return=False, name_model="kmeans")
    print("- calcul + sélection kmeans avec features Histogram...")
    kmeans_hist, k_hist, sil_hist = select_best_kmeans(descriptors_hist_km, kmeans_candidates_hog_hist)
    print(f"  -> HIST: k={k_hist}, silhouette={sil_hist}")
    show_metric(labels_true, kmeans_hist.labels_, descriptors_hist_km, bool_show=True, name_descriptor="HISTOGRAM", bool_return=False, name_model="kmeans")
    print("- calcul + sélection kmeans avec features HSV...")
    kmeans_hsv, k_hsv, sil_hsv = select_best_kmeans(descriptors_hsv_km, kmeans_candidates_default)
    print(f"  -> HSV: k={k_hsv}, silhouette={sil_hsv}")
    show_metric(labels_true, kmeans_hsv.labels_, descriptors_hsv_km, bool_show=True, name_descriptor="HSV", bool_return=False, name_model="kmeans")
    print("- calcul + sélection kmeans avec features LBP...")
    kmeans_lbp, k_lbp, sil_lbp = select_best_kmeans(descriptors_lbp_km, kmeans_candidates_default)
    print(f"  -> LBP: k={k_lbp}, silhouette={sil_lbp}")
    show_metric(labels_true, kmeans_lbp.labels_, descriptors_lbp_km, bool_show=True, name_descriptor="LBP", bool_return=False, name_model="kmeans")
    print("- calcul + sélection kmeans avec features ResNet50...")
    kmeans_resnet, k_resnet, sil_resnet = select_best_kmeans(descriptors_resnet_km, kmeans_candidates_resnet)
    print(f"  -> RESNET50: k={k_resnet}, silhouette={sil_resnet}")
    show_metric(labels_true, kmeans_resnet.labels_, descriptors_resnet_km, bool_show=True, name_descriptor="RESNET50", bool_return=False, name_model="kmeans")

    # ──────────────────────────────────────────────
    # 6. SPECTRAL CLUSTERING
    # ──────────────────────────────────────────────
    print("\n\n ##### Clustering Spectral ######")
    spectral_hog = SKLearnSpectralClustering(
        n_clusters=number_cluster, affinity='nearest_neighbors', n_neighbors=10, random_state=0)
    spectral_hist = SKLearnSpectralClustering(
        n_clusters=number_cluster, affinity='nearest_neighbors', n_neighbors=10, random_state=0)
    spectral_hsv = SKLearnSpectralClustering(
        n_clusters=number_cluster, affinity='nearest_neighbors', n_neighbors=10, random_state=0)
    spectral_lbp = SKLearnSpectralClustering(
        n_clusters=number_cluster, affinity='nearest_neighbors', n_neighbors=10, random_state=0)
    spectral_resnet = SKLearnSpectralClustering(
        n_clusters=22,
        affinity="rbf",
        gamma=0.03,
        assign_labels="discretize",
        random_state=0,
    )

    print("- calcul spectral clustering avec features HOG (PCA réduit)...")
    spectral_hog.fit(descriptors_hog_pca)
    print("- calcul spectral clustering avec features Histogram (PCA réduit)...")
    spectral_hist.fit(descriptors_hist_pca)
    print("- calcul spectral clustering avec features HSV (PCA réduit)...")
    spectral_hsv.fit(descriptors_hsv_pca)
    print("- calcul spectral clustering avec features LBP (PCA réduit)...")
    spectral_lbp.fit(descriptors_lbp_pca)
    print("- calcul spectral clustering avec features ResNet50 (PCA réduit)...")
    spectral_resnet.fit(descriptors_resnet_sc)

    # ──────────────────────────────────────────────
    # 7. MEANSHIFT — recherche de bandwidth
    # ──────────────────────────────────────────────
    print("\n\n ##### Clustering MeanShift ######")
    print("Recherche automatique de bandwidth via estimate_bandwidth (grid de quantiles)...")

    tuned_ms = tune_meanshift_bandwidth(
        descriptors_hog_pca=descriptors_hog_pca,
        descriptors_hist_pca=descriptors_hist_pca,
        descriptors_hsv_pca=descriptors_hsv_pca,
        descriptors_lbp_pca=descriptors_lbp_pca,
        descriptors_resnet_pca=descriptors_resnet_pca,
        target=number_cluster,
        n_samples_cap=952,
        verbose=True,
    )

    best_hog = tuned_ms["best_hog"]
    best_hist = tuned_ms["best_hist"]
    best_hsv = tuned_ms["best_hsv"]
    best_lbp = tuned_ms["best_lbp"]
    best_resnet = tuned_ms["best_resnet"]

    best_bw_hog = tuned_ms["best_bw"]["hog"]
    best_bw_hist = tuned_ms["best_bw"]["hist"]
    best_bw_hsv = tuned_ms["best_bw"]["hsv"]
    best_bw_lbp = tuned_ms["best_bw"]["lbp"]
    best_bw_resnet = tuned_ms["best_bw"]["resnet"]

    print(f"Choix HOG    -> bandwidth={best_bw_hog},    clusters={best_hog[2]}")
    print(f"Choix HIST   -> bandwidth={best_bw_hist},   clusters={best_hist[2]}")
    print(f"Choix HSV    -> bandwidth={best_bw_hsv},    clusters={best_hsv[2]}")
    print(f"Choix LBP    -> bandwidth={best_bw_lbp},    clusters={best_lbp[2]}")
    print(f"Choix RESNET -> bandwidth={best_bw_resnet}, clusters={best_resnet[2]}")

    # MeanShift fit final
    meanshift_hog = SKLearnMeanShift(
        bandwidth=best_bw_hog, bin_seeding=True, min_bin_freq=5, cluster_all=False
    ) if best_bw_hog else SKLearnMeanShift(cluster_all=False)

    meanshift_hist = SKLearnMeanShift(
        bandwidth=best_bw_hist, bin_seeding=True, min_bin_freq=3, cluster_all=False
    ) if best_bw_hist else SKLearnMeanShift(cluster_all=False)

    meanshift_hsv = SKLearnMeanShift(
        bandwidth=best_bw_hsv, bin_seeding=True, min_bin_freq=3, cluster_all=False
    ) if best_bw_hsv else SKLearnMeanShift(cluster_all=False)

    bw_lbp_use = best_bw_lbp
    meanshift_lbp = SKLearnMeanShift(bandwidth=bw_lbp_use, bin_seeding=False, cluster_all=True)

    meanshift_resnet = SKLearnMeanShift(
        bandwidth=best_bw_resnet, bin_seeding=True, min_bin_freq=5, cluster_all=False
    ) if best_bw_resnet else SKLearnMeanShift(cluster_all=False)

    print("- calcul meanshift avec features HOG (PCA réduit)...")
    meanshift_hog.fit(descriptors_hog_pca)
    print(f"  HOG clusters trouvés : {len(np.unique(meanshift_hog.labels_))}")
    show_metric(labels_true, meanshift_hog.labels_, descriptors_hog_ms, bool_show=True, name_descriptor="HOG", bool_return=False, name_model="meanshift")
    print("- calcul meanshift avec features Histogram (PCA réduit)...")
    meanshift_hist.fit(descriptors_hist_pca)
    print(f"  HIST clusters trouvés : {len(np.unique(meanshift_hist.labels_))}")
    show_metric(labels_true, meanshift_hist.labels_, descriptors_hist_ms, bool_show=True, name_descriptor="HISTOGRAM", bool_return=False, name_model="meanshift")
    print("- calcul meanshift avec features HSV (PCA réduit)...")
    meanshift_hsv.fit(descriptors_hsv_pca)
    print(f"  HSV clusters trouvés : {len(np.unique(meanshift_hsv.labels_))}")
    show_metric(labels_true, meanshift_hsv.labels_, descriptors_hsv_ms, bool_show=True, name_descriptor="HSV", bool_return=False, name_model="meanshift")
    print("- calcul meanshift avec features LBP (PCA réduit)...")
    meanshift_lbp.fit(descriptors_lbp_pca)
    print(f"  LBP clusters trouvés : {len(np.unique(meanshift_lbp.labels_))}")
    show_metric(labels_true, meanshift_lbp.labels_, descriptors_lbp_ms, bool_show=True, name_descriptor="LBP", bool_return=False, name_model="meanshift")
    print("- calcul meanshift avec features ResNet50 (PCA réduit)...")
    meanshift_resnet.fit(descriptors_resnet_pca)
    print(f"  RESNET clusters trouvés : {len(np.unique(meanshift_resnet.labels_))}")
    show_metric(labels_true, meanshift_resnet.labels_, descriptors_resnet_ms, bool_show=True, name_descriptor="RESNET50", bool_return=False, name_model="meanshift")

    # ──────────────────────────────────────────────
    # 8. MÉTRIQUES
    # ──────────────────────────────────────────────
    print("\n\n ##### Résultats ######")

    # KMeans
    metric_hog    = show_metric(labels_true, kmeans_hog.labels_,    descriptors_hog_km,    bool_show=True, name_descriptor="HOG",       bool_return=True, name_model="kmeans")
    metric_hist   = show_metric(labels_true, kmeans_hist.labels_,   descriptors_hist_km,   bool_show=True, name_descriptor="HISTOGRAM",  bool_return=True, name_model="kmeans")
    metric_hsv    = show_metric(labels_true, kmeans_hsv.labels_,    descriptors_hsv_km,    bool_show=True, name_descriptor="HSV",        bool_return=True, name_model="kmeans")
    metric_lbp    = show_metric(labels_true, kmeans_lbp.labels_,    descriptors_lbp_km,    bool_show=True, name_descriptor="LBP",        bool_return=True, name_model="kmeans")
    metric_resnet = show_metric(labels_true, kmeans_resnet.labels_, descriptors_resnet_km, bool_show=True, name_descriptor="RESNET50",   bool_return=True, name_model="kmeans")

    SILHOUETTE_K_VALUES = (5, 10, 15, 20, 25)

    # Traces silhouette KMeans (k fixes pour dashboard)
    silhouette_kmeans = {
        "HOG": compute_silhouette_at_fixed_k(descriptors_hog_km, SILHOUETTE_K_VALUES),
        "HISTOGRAM": compute_silhouette_at_fixed_k(descriptors_hist_km, SILHOUETTE_K_VALUES),
        "HSV": compute_silhouette_at_fixed_k(descriptors_hsv_km, SILHOUETTE_K_VALUES),
        "LBP": compute_silhouette_at_fixed_k(descriptors_lbp_km, SILHOUETTE_K_VALUES),
        "RESNET50": compute_silhouette_at_fixed_k(descriptors_resnet_km, SILHOUETTE_K_VALUES),
    }

    # MeanShift
    metric_hog_ms    = show_metric(labels_true, meanshift_hog.labels_,    descriptors_hog_ms,    bool_show=True, name_descriptor="HOG",      bool_return=True, name_model="meanshift")
    metric_hist_ms   = show_metric(labels_true, meanshift_hist.labels_,   descriptors_hist_ms,   bool_show=True, name_descriptor="HISTOGRAM", bool_return=True, name_model="meanshift")
    metric_hsv_ms    = show_metric(labels_true, meanshift_hsv.labels_,    descriptors_hsv_ms,    bool_show=True, name_descriptor="HSV",       bool_return=True, name_model="meanshift")
    metric_lbp_ms    = show_metric(labels_true, meanshift_lbp.labels_,    descriptors_lbp_ms,    bool_show=True, name_descriptor="LBP",       bool_return=True, name_model="meanshift")
    metric_resnet_ms = show_metric(labels_true, meanshift_resnet.labels_, descriptors_resnet_ms, bool_show=True, name_descriptor="RESNET50",  bool_return=True, name_model="meanshift")

    silhouette_meanshift = {
        "HOG": compute_silhouette_at_fixed_k(descriptors_hog_ms, SILHOUETTE_K_VALUES),
        "HISTOGRAM": compute_silhouette_at_fixed_k(descriptors_hist_ms, SILHOUETTE_K_VALUES),
        "HSV": compute_silhouette_at_fixed_k(descriptors_hsv_ms, SILHOUETTE_K_VALUES),
        "LBP": compute_silhouette_at_fixed_k(descriptors_lbp_ms, SILHOUETTE_K_VALUES),
        "RESNET50": compute_silhouette_at_fixed_k(descriptors_resnet_ms, SILHOUETTE_K_VALUES),
    }

    # Spectral
    metric_hog_sc    = show_metric(labels_true, spectral_hog.labels_,    descriptors_hog_pca,    bool_show=True, name_descriptor="HOG",      bool_return=True, name_model="spectralclustering")
    metric_hist_sc   = show_metric(labels_true, spectral_hist.labels_,   descriptors_hist_pca,   bool_show=True, name_descriptor="HISTOGRAM", bool_return=True, name_model="spectralclustering")
    metric_hsv_sc    = show_metric(labels_true, spectral_hsv.labels_,    descriptors_hsv_pca,    bool_show=True, name_descriptor="HSV",       bool_return=True, name_model="spectralclustering")
    metric_lbp_sc    = show_metric(labels_true, spectral_lbp.labels_,    descriptors_lbp_pca,    bool_show=True, name_descriptor="LBP",       bool_return=True, name_model="spectralclustering")
    metric_resnet_sc = show_metric(labels_true, spectral_resnet.labels_, descriptors_resnet_sc, bool_show=True, name_descriptor="RESNET50",  bool_return=True, name_model="spectralclustering")

    # Traces silhouette Spectral (k fixes pour dashboard)
    silhouette_spectral = {
        "HOG": compute_silhouette_at_fixed_k(descriptors_hog_pca, SILHOUETTE_K_VALUES),
        "HISTOGRAM": compute_silhouette_at_fixed_k(descriptors_hist_pca, SILHOUETTE_K_VALUES),
        "HSV": compute_silhouette_at_fixed_k(descriptors_hsv_pca, SILHOUETTE_K_VALUES),
        "LBP": compute_silhouette_at_fixed_k(descriptors_lbp_pca, SILHOUETTE_K_VALUES),
        "RESNET50": compute_silhouette_at_fixed_k(descriptors_resnet_sc, SILHOUETTE_K_VALUES),
    }

    # ──────────────────────────────────────────────
    # 9. EXPORT DASHBOARD
    # ──────────────────────────────────────────────
    print("- export des données vers le dashboard")

    list_dict = [
        metric_hog, metric_hist, metric_hsv, metric_lbp, metric_resnet,
        metric_hog_ms, metric_hist_ms, metric_hsv_ms, metric_lbp_ms, metric_resnet_ms,
        metric_hog_sc, metric_hist_sc, metric_hsv_sc, metric_lbp_sc, metric_resnet_sc,
    ]
    df_metric = pd.DataFrame(list_dict)

    # Normalisation pour visualisation 3D
    scaler_viz = StandardScaler()
    descriptors_hog_norm    = scaler_viz.fit_transform(descriptors_hog)
    descriptors_hist_norm   = scaler_viz.fit_transform(descriptors_hist)
    descriptors_hsv_norm    = scaler_viz.fit_transform(descriptors_hsv)
    descriptors_lbp_norm    = scaler_viz.fit_transform(descriptors_lbp)
    descriptors_resnet_norm = scaler_viz.fit_transform(descriptors_resnet)

    x_3d_hog    = conversion_3d(descriptors_hog_norm)
    x_3d_hist   = conversion_3d(descriptors_hist_norm)
    x_3d_hsv    = conversion_3d(descriptors_hsv_norm)
    x_3d_lbp    = conversion_3d(descriptors_lbp_norm)
    x_3d_resnet = conversion_3d(descriptors_resnet_norm)

    # DataFrames KMeans
    df_hog    = create_df_to_export(x_3d_hog,    labels_true, kmeans_hog.labels_)
    df_hist   = create_df_to_export(x_3d_hist,   labels_true, kmeans_hist.labels_)
    df_hsv    = create_df_to_export(x_3d_hsv,    labels_true, kmeans_hsv.labels_)
    df_lbp    = create_df_to_export(x_3d_lbp,    labels_true, kmeans_lbp.labels_)
    df_resnet = create_df_to_export(x_3d_resnet, labels_true, kmeans_resnet.labels_)

    # DataFrames MeanShift
    df_hog_ms    = create_df_to_export(x_3d_hog,    labels_true, meanshift_hog.labels_)
    df_hist_ms   = create_df_to_export(x_3d_hist,   labels_true, meanshift_hist.labels_)
    df_hsv_ms    = create_df_to_export(x_3d_hsv,    labels_true, meanshift_hsv.labels_)
    df_lbp_ms    = create_df_to_export(x_3d_lbp,    labels_true, meanshift_lbp.labels_)
    df_resnet_ms = create_df_to_export(x_3d_resnet, labels_true, meanshift_resnet.labels_)

    # DataFrames Spectral
    df_hog_sc    = create_df_to_export(x_3d_hog,    labels_true, spectral_hog.labels_)
    df_hist_sc   = create_df_to_export(x_3d_hist,   labels_true, spectral_hist.labels_)
    df_hsv_sc    = create_df_to_export(x_3d_hsv,    labels_true, spectral_hsv.labels_)
    df_lbp_sc    = create_df_to_export(x_3d_lbp,    labels_true, spectral_lbp.labels_)
    df_resnet_sc = create_df_to_export(x_3d_resnet, labels_true, spectral_resnet.labels_)

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # Sauvegarde KMeans
    save_dataframe_multi_format(df_hog,    output_path, "save_clustering_hog_kmeans")
    save_dataframe_multi_format(df_hist,   output_path, "save_clustering_hist_kmeans")
    save_dataframe_multi_format(df_hsv,    output_path, "save_clustering_hsv_kmeans")
    save_dataframe_multi_format(df_lbp,    output_path, "save_clustering_lbp_kmeans")
    save_dataframe_multi_format(df_resnet, output_path, "save_clustering_resnet_kmeans")

    # Sauvegarde MeanShift
    save_dataframe_multi_format(df_hog_ms,    output_path, "save_clustering_hog_meanshift")
    save_dataframe_multi_format(df_hist_ms,   output_path, "save_clustering_hist_meanshift")
    save_dataframe_multi_format(df_hsv_ms,    output_path, "save_clustering_hsv_meanshift")
    save_dataframe_multi_format(df_lbp_ms,    output_path, "save_clustering_lbp_meanshift")
    save_dataframe_multi_format(df_resnet_ms, output_path, "save_clustering_resnet_meanshift")

    # Sauvegarde Spectral
    save_dataframe_multi_format(df_hog_sc,    output_path, "save_clustering_hog_spectralclustering")
    save_dataframe_multi_format(df_hist_sc,   output_path, "save_clustering_hist_spectralclustering")
    save_dataframe_multi_format(df_hsv_sc,    output_path, "save_clustering_hsv_spectralclustering")
    save_dataframe_multi_format(df_lbp_sc,    output_path, "save_clustering_lbp_spectralclustering")
    save_dataframe_multi_format(df_resnet_sc, output_path, "save_clustering_resnet_spectralclustering")

    # Sauvegarde traces silhouette 
    for descriptor_name, descriptor_key in descriptor_key_map.items():
        save_dataframe_multi_format(
            silhouette_kmeans[descriptor_name],
            output_path,
            f"save_silhouette_tracking_{descriptor_key}_kmeans",
        )
        save_dataframe_multi_format(
            silhouette_meanshift[descriptor_name],
            output_path,
            f"save_silhouette_tracking_{descriptor_key}_meanshift",
        )
        save_dataframe_multi_format(
            silhouette_spectral[descriptor_name],
            output_path,
            f"save_silhouette_tracking_{descriptor_key}_spectralclustering",
        )

    save_dataframe_multi_format(df_metric, output_path, "save_metric")

    print(f"Résultats exportés dans: {output_path}")
    print("Fin.\n\nPour avoir la visualisation dashboard, veuillez lancer la commande : python dashboard.py --path_data chemin_vers_les_analyse_ia")


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
