import argparse
from sklearn.preprocessing import StandardScaler
import numpy as np
import os
import pandas as pd
import cv2

try:
    from .features import compute_hog_descriptors, compute_gray_histograms, compute_resnet50_descriptors
    from .clustering import show_metric
    from .utils import conversion_3d, create_df_to_export
    from .constant import PATH_OUTPUT, PATH_DATA, REPO_ROOT, LEGACY_PATH_DATA
except ImportError:
    # Support direct execution from Projet/src.
    from features import compute_hog_descriptors, compute_gray_histograms, compute_resnet50_descriptors
    from clustering import show_metric
    from utils import conversion_3d, create_df_to_export
    from constant import PATH_OUTPUT, PATH_DATA, REPO_ROOT, LEGACY_PATH_DATA
from sklearn.cluster import KMeans as SKLearnKMeans
from sklearn.cluster import MeanShift as SKLearnMeanShift
from sklearn.cluster import estimate_bandwidth
from sklearn.decomposition import PCA



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
    images, labels_true, label_names = load_snack_images(data_path)
    print(f"Nombre d'images chargées : {len(images)}")
    print(f"Catégories : {label_names}")
   
    print("\n\n ##### Extraction de Features ######")
    print("- calcul features hog...")
    descriptors_hog = compute_hog_descriptors(images)
    print("- calcul features Histogram...")
    descriptors_hist = compute_gray_histograms(images)
    print("- calcul features ResNet50...")
    descriptors_resnet = compute_resnet50_descriptors(images)

    # Réduction de dimension pour MeanShift (PCA ~10 dims)
    def _safe_n_components(X, target=10):
        n_samples, n_features = X.shape[0], X.shape[1]
        return max(1, min(target, n_features, max(1, n_samples - 1)))

    # Normalisation uniquement pour MeanShift (pas pour KMeans)
    scaler_ms_hog = StandardScaler()
    scaler_ms_hist = StandardScaler()
    scaler_ms_resnet = StandardScaler()
    descriptors_hog_ms = scaler_ms_hog.fit_transform(np.array(descriptors_hog))
    descriptors_hist_ms = scaler_ms_hist.fit_transform(np.array(descriptors_hist))
    descriptors_resnet_ms = scaler_ms_resnet.fit_transform(np.array(descriptors_resnet))

    n_comp_hog = _safe_n_components(descriptors_hog_ms, target=10)
    n_comp_hist = _safe_n_components(descriptors_hist_ms, target=10)
    n_comp_resnet = _safe_n_components(descriptors_resnet_ms, target=10)
    pca_hog = PCA(n_components=n_comp_hog)
    pca_hist = PCA(n_components=n_comp_hist)
    pca_resnet = PCA(n_components=n_comp_resnet)
    descriptors_hog_pca = pca_hog.fit_transform(descriptors_hog_ms)
    descriptors_hist_pca = pca_hist.fit_transform(descriptors_hist_ms)
    descriptors_resnet_pca = pca_resnet.fit_transform(descriptors_resnet_ms)

    print(f"Applied PCA: HOG -> {n_comp_hog} dims, HIST -> {n_comp_hist} dims, RESNET50 -> {n_comp_resnet} dims")

    number_cluster = len(label_names)  # Nombre de catégories

    # Recherche automatique de bandwidth via estimate_bandwidth (grid de quantiles)
    print("Recherche automatique de bandwidth via estimate_bandwidth (grid de quantiles)...")
    quantiles = list(np.linspace(0.01, 0.5, 20))
    results_hog = []
    results_hist = []
    results_resnet = []
    target = number_cluster
    for q in quantiles:
        # HOG
        try:
            bw = estimate_bandwidth(descriptors_hog_pca, quantile=q, n_samples=min(500, len(descriptors_hog_pca)))
            if bw is None or bw <= 0:
                n_hog = None
            else:
                n_hog = len(np.unique(SKLearnMeanShift(bandwidth=bw, bin_seeding=True).fit(descriptors_hog_pca).labels_))
        except Exception:
            bw = None
            n_hog = None
        results_hog.append((q, bw, n_hog))

        # HIST
        try:
            bw2 = estimate_bandwidth(descriptors_hist_pca, quantile=q, n_samples=min(500, len(descriptors_hist_pca)))
            if bw2 is None or bw2 <= 0:
                n_hist = None
            else:
                n_hist = len(np.unique(SKLearnMeanShift(bandwidth=bw2, bin_seeding=True).fit(descriptors_hist_pca).labels_))
        except Exception:
            bw2 = None
            n_hist = None
        results_hist.append((q, bw2, n_hist))

        # RESNET50
        try:
            bw3 = estimate_bandwidth(descriptors_resnet_pca, quantile=q, n_samples=min(500, len(descriptors_resnet_pca)))
            if bw3 is None or bw3 <= 0:
                n_resnet = None
            else:
                n_resnet = len(np.unique(SKLearnMeanShift(bandwidth=bw3, bin_seeding=True).fit(descriptors_resnet_pca).labels_))
        except Exception:
            bw3 = None
            n_resnet = None
        results_resnet.append((q, bw3, n_resnet))

        print(f"quantile={q:.3f} -> HOG: bw={bw}, n_clusters={n_hog} | HIST: bw={bw2}, n_clusters={n_hist} | RESNET50: bw={bw3}, n_clusters={n_resnet}")

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

    if best_resnet[1] is None:
        print("Aucun bandwidth utile trouvé pour RESNET50 via estimate_bandwidth; utilisation du bandwidth par défaut.")
        best_bw_resnet = None
    else:
        best_bw_resnet = best_resnet[1]
        print(f"Choix RESNET50 -> quantile={best_resnet[0]:.3f}, bandwidth={best_bw_resnet}, clusters={best_resnet[2]}")

    print("\n\n ##### Clustering ######")
    kmeans_hog = SKLearnKMeans(n_clusters=number_cluster, random_state=0)
    kmeans_hist = SKLearnKMeans(n_clusters=number_cluster, random_state=0)
    kmeans_resnet = SKLearnKMeans(n_clusters=number_cluster, random_state=0)

    print("- calcul kmeans avec features HOG ...")
    kmeans_hog.fit(np.array(descriptors_hog))
    print("- calcul kmeans avec features Histogram...")
    kmeans_hist.fit(np.array(descriptors_hist))
    print("- calcul kmeans avec features ResNet50...")
    kmeans_resnet.fit(np.array(descriptors_resnet))

    # MeanShift clustering (sur données réduites par PCA) avec les bandwidth choisis
    if best_bw_hog is not None:
        meanshift_hog = SKLearnMeanShift(bandwidth=best_bw_hog, bin_seeding=True)
    else:
        meanshift_hog = SKLearnMeanShift()

    if best_bw_hist is not None:
        meanshift_hist = SKLearnMeanShift(bandwidth=best_bw_hist, bin_seeding=True)
    else:
        meanshift_hist = SKLearnMeanShift()

    if best_bw_resnet is not None:
        meanshift_resnet = SKLearnMeanShift(bandwidth=best_bw_resnet, bin_seeding=True)
    else:
        meanshift_resnet = SKLearnMeanShift()

    print("- calcul meanshift avec features HOG (PCA réduit)...")
    meanshift_hog.fit(descriptors_hog_pca)
    print("- calcul meanshift avec features Histogram (PCA réduit)...")
    meanshift_hist.fit(descriptors_hist_pca)
    print("- calcul meanshift avec features ResNet50 (PCA réduit)...")
    meanshift_resnet.fit(descriptors_resnet_pca)


    print("\n\n ##### Résultat ######")
    metric_hist = show_metric(labels_true, kmeans_hist.labels_, descriptors_hist, bool_show=True, name_descriptor="HISTOGRAM", bool_return=True, name_model="kmeans")
    metric_hog = show_metric(labels_true, kmeans_hog.labels_, descriptors_hog,bool_show=True, name_descriptor="HOG", bool_return=True, name_model="kmeans")
    metric_resnet = show_metric(labels_true, kmeans_resnet.labels_, descriptors_resnet, bool_show=True, name_descriptor="RESNET50", bool_return=True, name_model="kmeans")

    metric_hist_ms = show_metric(labels_true, meanshift_hist.labels_, descriptors_hist, bool_show=True, name_descriptor="HISTOGRAM", bool_return=True, name_model="meanshift")
    metric_hog_ms = show_metric(labels_true, meanshift_hog.labels_, descriptors_hog, bool_show=True, name_descriptor="HOG", bool_return=True, name_model="meanshift")
    metric_resnet_ms = show_metric(labels_true, meanshift_resnet.labels_, descriptors_resnet, bool_show=True, name_descriptor="RESNET50", bool_return=True, name_model="meanshift")


    print("- export des données vers le dashboard")
    # conversion des données vers le format du dashboard
    list_dict = [metric_hist, metric_hog, metric_resnet, metric_hist_ms, metric_hog_ms, metric_resnet_ms]
    df_metric = pd.DataFrame(list_dict)
    
    # Normalisation des données
    scaler = StandardScaler()
    descriptors_hist_norm = scaler.fit_transform(descriptors_hist)
    descriptors_hog_norm = scaler.fit_transform(descriptors_hog)
    descriptors_resnet_norm = scaler.fit_transform(descriptors_resnet)

    #conversion vers un format 3D pour la visualisation
    x_3d_hist = conversion_3d(descriptors_hist_norm)
    x_3d_hog = conversion_3d(descriptors_hog_norm)
    x_3d_resnet = conversion_3d(descriptors_resnet_norm)

    # création des dataframe pour la sauvegarde des données pour la visualisation
    df_hist = create_df_to_export(x_3d_hist, labels_true, kmeans_hist.labels_)
    df_hog = create_df_to_export(x_3d_hog, labels_true, kmeans_hog.labels_)
    df_resnet = create_df_to_export(x_3d_resnet, labels_true, kmeans_resnet.labels_)

    # Dataframes for MeanShift
    df_hist_meanshift = create_df_to_export(x_3d_hist, labels_true, meanshift_hist.labels_)
    df_hog_meanshift = create_df_to_export(x_3d_hog, labels_true, meanshift_hog.labels_)
    df_resnet_meanshift = create_df_to_export(x_3d_resnet, labels_true, meanshift_resnet.labels_)

    # Vérifie si le dossier existe déjà
    if not os.path.exists(output_path):
        # Crée le dossier
        os.makedirs(output_path)

    # sauvegarde des données
    save_dataframe_multi_format(df_hist, output_path, "save_clustering_hist_kmeans")
    save_dataframe_multi_format(df_hog, output_path, "save_clustering_hog_kmeans")
    save_dataframe_multi_format(df_resnet, output_path, "save_clustering_resnet_kmeans")
    save_dataframe_multi_format(df_hist_meanshift, output_path, "save_clustering_hist_meanshift")
    save_dataframe_multi_format(df_hog_meanshift, output_path, "save_clustering_hog_meanshift")
    save_dataframe_multi_format(df_resnet_meanshift, output_path, "save_clustering_resnet_meanshift")
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