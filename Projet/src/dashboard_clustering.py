import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.metrics import silhouette_score
import plotly.express as px
import cv2
import os

try:
    from .features import compute_hog_descriptors, compute_gray_histograms, compute_resnet50_descriptors
    from .constant import PATH_DATA, PATH_OUTPUT, REPO_ROOT, LEGACY_PATH_DATA
except ImportError:
    from features import compute_hog_descriptors, compute_gray_histograms, compute_resnet50_descriptors
    from constant import PATH_DATA, PATH_OUTPUT, REPO_ROOT, LEGACY_PATH_DATA


def resolve_data_path():
    env_path = os.getenv("PATH_DATA")
    if env_path:
        return os.path.abspath(env_path)
    if os.path.isdir(PATH_DATA):
        return PATH_DATA
    if os.path.isdir(LEGACY_PATH_DATA):
        return LEGACY_PATH_DATA
    return PATH_DATA


def resolve_analysis_path():
    env_analysis = os.getenv("PATH_ANALYSIS")
    if env_analysis:
        return os.path.abspath(env_analysis)
    env_output = os.getenv("PATH_OUTPUT")
    if env_output:
        return os.path.abspath(env_output)
    if os.path.isabs(PATH_OUTPUT):
        return PATH_OUTPUT
    return os.path.join(REPO_ROOT, PATH_OUTPUT)


def read_analysis_file(analysis_dir, base_name):
    """Read exported pipeline artifact preferring xlsx, then csv."""
    xlsx_path = os.path.join(analysis_dir, f"{base_name}.xlsx")
    csv_path = os.path.join(analysis_dir, f"{base_name}.csv")
    if os.path.exists(xlsx_path):
        return pd.read_excel(xlsx_path)
    if os.path.exists(csv_path):
        return pd.read_csv(csv_path)
    return None


@st.cache_data
def colorize_cluster(cluster_data, selected_cluster):
    fig = px.scatter_3d(cluster_data, x='x', y='y', z='z', color='cluster')
    filtered_data = cluster_data[cluster_data['cluster'] == selected_cluster]
    fig.add_scatter3d(x=filtered_data['x'], y=filtered_data['y'], z=filtered_data['z'],
                    mode='markers', marker=dict(color='red', size=10),
                    name=f'Cluster {selected_cluster}')
    return fig

@st.cache_data
def plot_metric(df_metric):
    fig = px.bar(df_metric, x="descriptor", y="ami", color="descriptor",
                 title="Score AMI par descripteur")
    st.plotly_chart(fig)


@st.cache_data
def load_snack_images_with_paths(data_path, img_size=(64, 64)):
    images = []
    labels = []
    label_names = []
    image_paths = []

    categories = sorted([
        d for d in os.listdir(data_path)
        if os.path.isdir(os.path.join(data_path, d))
    ])

    for label_idx, category in enumerate(categories):
        category_path = os.path.join(data_path, category)
        label_names.append(category)

        # Keep iteration order aligned with pipeline generation.
        for img_name in os.listdir(category_path):
            img_path = os.path.join(category_path, img_name)
            img = cv2.imread(img_path)
            if img is not None:
                img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                img_resized = cv2.resize(img_gray, img_size)
                images.append(img_resized)
                labels.append(label_idx)
                image_paths.append(img_path)

    return np.array(images), np.array(labels), label_names, image_paths


@st.cache_data
def compute_descriptor_matrix(images, descriptor):
    if descriptor == "HOG":
        return compute_hog_descriptors(images)
    if descriptor == "RESNET50":
        return compute_resnet50_descriptors(images)
    return compute_gray_histograms(images)


@st.cache_data
def compute_silhouette_tracking(descriptors, model_name, k_values):
    n_samples = len(descriptors)
    data = []

    for k in k_values:
        if k >= n_samples:
            data.append({"k": k, "silhouette": np.nan})
            continue

        try:
            if model_name == "kmeans":
                model = KMeans(n_clusters=k, random_state=0)
                labels = model.fit_predict(descriptors)
            else:
                quantile = min(0.5, max(0.01, k / max(1, n_samples)))
                bandwidth = estimate_bandwidth(
                    descriptors,
                    quantile=quantile,
                    n_samples=min(500, n_samples),
                )
                if bandwidth is None or bandwidth <= 0:
                    data.append({"k": k, "silhouette": np.nan})
                    continue
                model = MeanShift(bandwidth=bandwidth, bin_seeding=True)
                labels = model.fit_predict(descriptors)

            n_labels = len(np.unique(labels))
            if n_labels <= 1 or n_labels >= n_samples:
                score = np.nan
            else:
                score = float(silhouette_score(descriptors, labels))
            data.append({"k": k, "silhouette": score})
        except Exception:
            data.append({"k": k, "silhouette": np.nan})

    return pd.DataFrame(data)

        
analysis_dir = resolve_analysis_path()
data_dir = resolve_data_path()

df_metric = read_analysis_file(analysis_dir, "save_metric")
df_hist_kmeans = read_analysis_file(analysis_dir, "save_clustering_hist_kmeans")
df_hog_kmeans = read_analysis_file(analysis_dir, "save_clustering_hog_kmeans")
df_resnet_kmeans = read_analysis_file(analysis_dir, "save_clustering_resnet_kmeans")
df_hist_meanshift = read_analysis_file(analysis_dir, "save_clustering_hist_meanshift")
df_hog_meanshift = read_analysis_file(analysis_dir, "save_clustering_hog_meanshift")
df_resnet_meanshift = read_analysis_file(analysis_dir, "save_clustering_resnet_meanshift")

if df_metric is None:
    st.error(f"Aucun fichier métrique trouvé dans: {analysis_dir}")
    st.info("Exécutez d'abord le pipeline IA, puis relancez le dashboard.")
    st.stop()

if 'Unnamed: 0' in df_metric.columns:
    df_metric.drop(columns="Unnamed: 0", inplace=True)

# Ensure same naming convention as pipeline.
if "name_model" in df_metric.columns:
    df_metric["name_model"] = df_metric["name_model"].astype(str).str.lower()

# Création de deux onglets
tab1, tab2 = st.tabs(["Analyse par descripteur", "Analyse global" ])

# Onglet numéro 1
with tab1:

    st.write('## Résultat de Clustering des données SNACK')
    st.sidebar.write("####  Veuillez sélectionner les clusters à analyser" )
    # Sélection du modèle
    model = st.sidebar.selectbox('Sélectionner le modèle de clustering', ["kmeans", "meanshift"])
    # Sélection des descripteurs
    descriptor =  st.sidebar.selectbox('Sélectionner un descripteur', ["HISTOGRAM","HOG", "RESNET50"])
    # Récupérer le dataframe correspondant au modèle et au descripteur
    df = None
    if model == "kmeans":
        if descriptor == "HISTOGRAM":
            df = df_hist_kmeans
        elif descriptor == "HOG":
            df = df_hog_kmeans
        else:
            df = df_resnet_kmeans
    else:
        if descriptor == "HISTOGRAM":
            df = df_hist_meanshift
        elif descriptor == "HOG":
            df = df_hog_meanshift
        else:
            df = df_resnet_meanshift

    if df is None:
        st.warning("Données manquantes pour le modèle/descripteur sélectionné. Exécutez le pipeline d'abord.")
        st.stop()
    # Ajouter un sélecteur pour les clusters - valeurs réelles présentes dans les données
    cluster_values = sorted(df['cluster'].unique().tolist())
    selected_cluster =  st.sidebar.selectbox('Sélectionner un Cluster', cluster_values)
    # Filtrer les données en fonction du cluster sélectionné
    cluster_indices = df[df.cluster==selected_cluster].index    
    st.write(f"###  Analyse du descripteur {descriptor}" )
    st.write(f"#### Analyse du cluster : {selected_cluster}")
    st.write(f"####  Visualisation 3D du clustering avec descripteur {descriptor}" )
    # Sélection du cluster choisi
    filtered_data = df[df['cluster'] == selected_cluster]
    # Création d'un graph 3D des clusters
    fig = colorize_cluster(df, selected_cluster)
    st.plotly_chart(fig)

    st.write("#### Image exemple du cluster")
    images, labels_true, label_names, image_paths = load_snack_images_with_paths(data_dir)

    if len(cluster_indices) > 0:
        example_idx = int(cluster_indices[0])
        if example_idx < len(images):
            st.image(images[example_idx], caption=f"Exemple index {example_idx} | cluster {selected_cluster}", clamp=True)
            if "label" in df.columns:
                true_label_idx = int(df.iloc[example_idx]["label"])
                if 0 <= true_label_idx < len(label_names):
                    st.caption(f"Label réel: {label_names[true_label_idx]}")
        else:
            st.warning("Impossible de retrouver l'image correspondante (index hors limites).")
    else:
        st.info("Aucune image trouvée pour ce cluster.")

    st.write("#### Métriques du clustering sélectionné")
    metric_row = df_metric[
        (df_metric["descriptor"] == descriptor)
        & (df_metric["name_model"] == model)
    ]

    if metric_row.empty:
        st.warning("Aucune métrique trouvée pour ce modèle/descripteur.")
    else:
        m = metric_row.iloc[0]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("AMI", f"{m['ami']:.4f}")
        c2.metric("ARI", f"{m['ari']:.4f}")
        c3.metric("Silhouette", "N/A" if pd.isna(m['silhouette']) else f"{m['silhouette']:.4f}")
        c4.metric("Jaccard", f"{m['jaccard']:.4f}")

        c5, c6, c7 = st.columns(3)
        c5.metric("Homogeneity", f"{m['homogeneity']:.4f}")
        c6.metric("Completeness", f"{m['completeness']:.4f}")
        c7.metric("V-measure", f"{m['v_measure']:.4f}")

    st.write("#### Suivi du silhouette score (k = 5, 10, 15, 20, 25)")
    k_values = [5, 10, 15, 20, 25]
    descriptors = compute_descriptor_matrix(images, descriptor)
    df_sil = compute_silhouette_tracking(descriptors, model, k_values)

    fig_sil = px.line(
        df_sil,
        x="k",
        y="silhouette",
        markers=True,
        title=f"Évolution du silhouette score ({model} - {descriptor})",
    )
    fig_sil.update_xaxes(type="category")
    st.plotly_chart(fig_sil)
    st.dataframe(df_sil)

# Onglet numéro 2
with tab2:
    st.write('## Analyse Global des descripteurs' )

    # Complèter la fonction plot_metric() pour afficher les histogrammes du score AMI

    plot_metric(df_metric)
    st.write('## Métriques ' )
    st.dataframe(df_metric)

