import os

import cv2
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

try:
    from .constant import PATH_DATA, PATH_OUTPUT, REPO_ROOT, LEGACY_PATH_DATA
except ImportError:
    from constant import PATH_DATA, PATH_OUTPUT, REPO_ROOT, LEGACY_PATH_DATA


MODELS = ("kmeans", "meanshift", "spectralclustering")
DESCRIPTORS = ("HISTOGRAM", "HOG", "HSV", "LBP", "RESNET50")
DESCRIPTOR_TO_ARTIFACT = {
    "HISTOGRAM": "hist",
    "HOG": "hog",
    "HSV": "hsv",
    "LBP": "lbp",
    "RESNET50": "resnet",
}


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


@st.cache_data
def read_analysis_file(analysis_dir, base_name):
    """Read exported pipeline artifact preferring xlsx, then csv."""
    xlsx_path = os.path.join(analysis_dir, f"{base_name}.xlsx")
    csv_path = os.path.join(analysis_dir, f"{base_name}.csv")
    if os.path.exists(xlsx_path):
        return pd.read_excel(xlsx_path)
    if os.path.exists(csv_path):
        return pd.read_csv(csv_path)
    return None


def build_cluster_artifact_name(model, descriptor):
    descriptor_key = DESCRIPTOR_TO_ARTIFACT[descriptor]
    return f"save_clustering_{descriptor_key}_{model}"


@st.cache_data
def load_metrics(analysis_dir):
    df_metric = read_analysis_file(analysis_dir, "save_metric")
    if df_metric is None:
        return None

    df_metric = df_metric.copy()
    if "Unnamed: 0" in df_metric.columns:
        df_metric.drop(columns="Unnamed: 0", inplace=True)

    if "name_model" in df_metric.columns:
        df_metric["name_model"] = df_metric["name_model"].astype(str).str.lower()

    return df_metric


@st.cache_data
def get_data(analysis_dir, model, descriptor):
    """Load one clustering artifact on demand to limit memory usage."""
    base_name = build_cluster_artifact_name(model, descriptor)
    df = read_analysis_file(analysis_dir, base_name)
    if df is None:
        return None

    df = df.copy()
    if "Unnamed: 0" in df.columns:
        df.drop(columns="Unnamed: 0", inplace=True)
    return df


@st.cache_data
def load_silhouette_tracking(analysis_dir, model, descriptor):
    descriptor_key = DESCRIPTOR_TO_ARTIFACT[descriptor]
    candidates = [
        f"save_silhouette_tracking_{descriptor_key}_{model}",
        f"save_silhouette_{descriptor_key}_{model}",
    ]

    for base_name in candidates:
        df = read_analysis_file(analysis_dir, base_name)
        if df is not None:
            return df
    return None


@st.cache_data
def load_snack_images_with_paths(data_path, img_size=(128, 128)):
    images_gray = []
    images_bgr = []
    labels = []
    label_names = []
    image_paths = []

    if not os.path.isdir(data_path):
        return np.array([]), np.array([]), np.array([]), [], []

    categories = sorted(
        [
            d
            for d in os.listdir(data_path)
            if os.path.isdir(os.path.join(data_path, d))
        ]
    )

    for label_idx, category in enumerate(categories):
        category_path = os.path.join(data_path, category)
        label_names.append(category)

        # Keep iteration order aligned with pipeline generation.
        for img_name in os.listdir(category_path):
            img_path = os.path.join(category_path, img_name)
            img = cv2.imread(img_path)
            if img is None:
                continue

            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img_resized = cv2.resize(img_gray, img_size)
            img_bgr_resized = cv2.resize(img, img_size)
            images_gray.append(img_resized)
            images_bgr.append(img_bgr_resized)
            labels.append(label_idx)
            image_paths.append(img_path)

    return (
        np.array(images_gray),
        np.array(images_bgr),
        np.array(labels),
        label_names,
        image_paths,
    )


@st.cache_data
def build_cluster_figure(cluster_data, selected_cluster):
    fig = px.scatter_3d(cluster_data, x="x", y="y", z="z", color="cluster")
    filtered_data = cluster_data[cluster_data["cluster"] == selected_cluster]
    fig.add_scatter3d(
        x=filtered_data["x"],
        y=filtered_data["y"],
        z=filtered_data["z"],
        mode="markers",
        marker=dict(color="red", size=10),
        name=f"Cluster {selected_cluster}",
    )
    return fig


@st.cache_data
def build_metric_figure(df_metric):
    return px.bar(
        df_metric,
        x="descriptor",
        y="ami",
        color="descriptor",
        title="Score AMI par descripteur",
    )


def normalize_silhouette_columns(df_sil):
    if df_sil is None or df_sil.empty:
        return None

    df_norm = df_sil.copy()
    if "n_clusters" in df_norm.columns and "k" not in df_norm.columns:
        df_norm.rename(columns={"n_clusters": "k"}, inplace=True)

    required_columns = {"k", "silhouette"}
    if not required_columns.issubset(df_norm.columns):
        return None

    df_norm["k"] = pd.to_numeric(df_norm["k"], errors="coerce")
    df_norm["silhouette"] = pd.to_numeric(df_norm["silhouette"], errors="coerce")
    df_norm = df_norm.dropna(subset=["k"])
    df_norm = df_norm.sort_values(by="k", kind="mergesort").reset_index(drop=True)

    return df_norm[["k", "silhouette"]]


def render_sidebar():
    st.sidebar.write("#### Veuillez selectionner les clusters a analyser")
    selected_model = st.sidebar.selectbox(
        "Selectionner le modele de clustering", MODELS
    )
    selected_descriptor = st.sidebar.selectbox(
        "Selectionner un descripteur", DESCRIPTORS
    )
    return selected_model, selected_descriptor


def render_metrics_panel(df_metric, model, descriptor):
    st.write("#### Metriques du clustering selectionne")

    metric_row = df_metric[
        (df_metric["descriptor"] == descriptor) & (df_metric["name_model"] == model)
    ]
    if metric_row.empty:
        st.warning("Aucune metrique trouvee pour ce modele/descripteur.")
        return

    m = metric_row.iloc[0]
    db_metric = m.get("davies_bouldin", np.nan)
    if pd.isna(db_metric):
        st.info(
            "Davies-Bouldin absent dans save_metric: valeur non recalculee dans le dashboard "
            "pour conserver des performances stables."
        )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("AMI", f"{m['ami']:.4f}")
    c2.metric("ARI", f"{m['ari']:.4f}")
    c3.metric(
        "Silhouette",
        "N/A" if pd.isna(m["silhouette"]) else f"{m['silhouette']:.4f}",
    )
    c4.metric("Jaccard", f"{m['jaccard']:.4f}")

    c5, c6, c7, c8 = st.columns(4)
    c5.metric("Homogeneity", f"{m['homogeneity']:.4f}")
    c6.metric("Completeness", f"{m['completeness']:.4f}")
    c7.metric("V-measure", f"{m['v_measure']:.4f}")
    c8.metric("Davies-Bouldin", "N/A" if pd.isna(db_metric) else f"{db_metric:.4f}")


def render_silhouette_panel(analysis_dir, model, descriptor):
    st.write("#### Suivi du silhouette score (k = 5, 10, 15, 20, 25)")
    df_sil = normalize_silhouette_columns(
        load_silhouette_tracking(analysis_dir, model, descriptor)
    )

    if df_sil is None:
        st.info("Aucun suivi silhouette pre-calcule trouve.")
        return

    df_sil = df_sil.copy()
    df_sil["k"] = df_sil["k"].astype(int)
    df_sil = df_sil[df_sil["k"].isin([5, 10, 15, 20, 25])].copy()

    if df_sil.empty:
        st.info("Les donnees silhouette ne contiennent pas les valeurs k = 5, 10, 15, 20, 25.")
        return

    fig_sil = px.line(
        df_sil,
        x="k",
        y="silhouette",
        markers=True,
        title=f"Silhouette score pour k in {{5, 10, 15, 20, 25}} - {model} / {descriptor}",
        labels={"k": "Nombre de clusters k", "silhouette": "Score Silhouette"},
    )
    fig_sil.update_xaxes(
        tickvals=[5, 10, 15, 20, 25],
        ticktext=["5", "10", "15", "20", "25"],
        type="category",
    )
    fig_sil.update_yaxes(title="Score Silhouette")
    st.plotly_chart(fig_sil, use_container_width=True)
    st.dataframe(df_sil.reset_index(drop=True))


def render_cluster_images(cluster_rows, images_bgr, label_names):
    st.write("#### Extrait de 10 images du cluster")
    if len(images_bgr) == 0:
        st.warning("Aucune image chargee depuis le dossier de donnees.")
        return

    cluster_indices = cluster_rows.index.to_list()
    if not cluster_indices:
        st.info("Aucune image trouvee pour ce cluster.")
        return

    sample_indices = [int(i) for i in cluster_indices[:10]]
    valid_indices = [i for i in sample_indices if 0 <= i < len(images_bgr)]

    if not valid_indices:
        st.warning("Impossible de retrouver les images correspondantes (index hors limites).")
        return

    cols = st.columns(5)
    for pos, image_idx in enumerate(valid_indices):
        col = cols[pos % 5]
        with col:
            img_rgb = cv2.cvtColor(images_bgr[image_idx], cv2.COLOR_BGR2RGB)
            st.image(img_rgb, clamp=True)

            caption = f"Index {image_idx}"
            if "label" in cluster_rows.columns and image_idx in cluster_rows.index:
                true_label_idx = int(cluster_rows.loc[image_idx, "label"])
                if 0 <= true_label_idx < len(label_names):
                    caption += f" | {label_names[true_label_idx]}"
            st.caption(caption)

    if len(cluster_indices) < 10:
        st.info(f"Ce cluster contient seulement {len(cluster_indices)} image(s).")


def render_descriptor_tab(df_metric, analysis_dir, data_dir):
    st.write("## Resultat de Clustering des donnees SNACK")

    model, descriptor = render_sidebar()
    df = get_data(analysis_dir, model, descriptor)
    if df is None:
        expected_name = build_cluster_artifact_name(model, descriptor)
        st.error(
            f"Fichier manquant pour la selection courante: {expected_name}.csv/.xlsx "
            f"dans {analysis_dir}"
        )
        return

    required_columns = {"cluster", "x", "y", "z"}
    if not required_columns.issubset(df.columns):
        st.error(
            "Le fichier de clustering ne contient pas toutes les colonnes attendues: "
            "cluster, x, y, z."
        )
        return

    cluster_values = sorted(df["cluster"].dropna().unique().tolist())
    if not cluster_values:
        st.warning("Le fichier de clustering est vide ou ne contient aucun cluster valide.")
        return

    selected_cluster = st.sidebar.selectbox("Selectionner un Cluster", cluster_values)
    selected_data = df[df["cluster"] == selected_cluster]

    st.write(f"### Analyse du descripteur {descriptor}")
    st.write(f"#### Analyse du cluster: {selected_cluster}")
    st.write(f"#### Visualisation 3D du clustering avec descripteur {descriptor}")
    st.plotly_chart(build_cluster_figure(df, selected_cluster))

    _, images_bgr, _, label_names, _ = load_snack_images_with_paths(data_dir)
    render_cluster_images(selected_data, images_bgr, label_names)
    render_metrics_panel(df_metric, model, descriptor)
    render_silhouette_panel(analysis_dir, model, descriptor)


def render_global_tab(df_metric):
    st.write("## Analyse Globale des descripteurs")
    st.plotly_chart(build_metric_figure(df_metric))
    st.write("## Metriques")
    st.dataframe(df_metric)


def main():
    st.set_page_config(page_title="Dashboard Clustering", layout="wide")

    analysis_dir = resolve_analysis_path()
    data_dir = resolve_data_path()
    df_metric = load_metrics(analysis_dir)

    if df_metric is None:
        st.error(f"Aucun fichier metrique trouve dans: {analysis_dir}")
        st.info("Executez d'abord le pipeline IA, puis relancez le dashboard.")
        return

    if "davies_bouldin" not in df_metric.columns:
        st.info(
            "La colonne davies_bouldin est absente des metriques exportees. "
            "Le dashboard n'effectue pas de recalcul lourd."
        )

    tab1, tab2 = st.tabs(["Analyse par descripteur", "Analyse globale"])
    with tab1:
        render_descriptor_tab(df_metric, analysis_dir, data_dir)
    with tab2:
        render_global_tab(df_metric)


if __name__ == "__main__":
    main()
