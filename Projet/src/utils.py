import os
import pandas as pd
from sklearn.manifold import TSNE
import numpy as np

try:
        from .constant import PATH_OUTPUT, PATH_DATA, REPO_ROOT, LEGACY_PATH_DATA
except ImportError:
        from constant import PATH_OUTPUT, PATH_DATA, REPO_ROOT, LEGACY_PATH_DATA


def resolve_data_path(path_data=None):
        """
        Détermine le chemin des données d'entrée selon une hiérarchie de priorité.
        Priorité : Argument CLI > Variable d'environnement > Chemin par défaut > Legacy.

        Args:
                path_data (str, optional): Chemin passé explicitement en argument.

        Returns:
                str: Chemin absolu ou relatif valide vers le dossier de données.
        """
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
        """
        Détermine le dossier de destination pour les résultats et exports.
        Gère la conversion des chemins relatifs par rapport à la racine du projet.

        Args:
                path_output (str, optional): Dossier cible passé en argument.

        Returns:
                str: Chemin absolu vers le dossier de sortie.
        """
        if path_output:
                return os.path.abspath(path_output)

        env_path = os.getenv("PATH_OUTPUT")
        if env_path:
                return os.path.abspath(env_path)

        if os.path.isabs(PATH_OUTPUT):
                return PATH_OUTPUT

        return os.path.join(REPO_ROOT, PATH_OUTPUT)


def save_dataframe_multi_format(df, out_dir, base_name):
        """
        Exporte un DataFrame pandas simultanément en formats Excel et CSV.

        Args:
                df (pd.DataFrame): Les données (métriques ou clusters) à sauvegarder.
                out_dir (str): Dossier de destination (doit être résolu au préalable).
                base_name (str): Nom du fichier sans extension.
        """
        excel_path = os.path.join(out_dir, f"{base_name}.xlsx")
        csv_path = os.path.join(out_dir, f"{base_name}.csv")

        df.to_excel(excel_path, index=False)
        df.to_csv(csv_path, index=False)

def conversion_3d(X, n_components=3,perplexity=50,random_state=42, early_exaggeration=10,max_iter=3000):
    """
    Conversion des vecteurs de N dimensions vers une dimension précise (n_components) pour la visualisation
    Input : X (array-like) : données à convertir en 3D
            n_components (int) : nombre de dimensions cibles (par défaut : 3)
            perplexity (float) : valeur de perplexité pour t-SNE (par défaut : 50)
            random_state (int) : graine pour la génération de nombres aléatoires (par défaut : 42)
            early_exaggeration (float) : facteur d'exagération pour t-SNE (par défaut : 10)
            max_iter (int) : nombre d'itérations pour t-SNE (par défaut : 3000)
    Output : X_3d (array-like) : données converties en 3D
    """
    tsne = TSNE(n_components=n_components,
                random_state=random_state,
                perplexity=perplexity,
                early_exaggeration=early_exaggeration,
                max_iter=max_iter
               )
    X = np.array(X)
    X_3d = tsne.fit_transform(X)
    return X_3d


def create_df_to_export(data_3d, l_true_label,l_cluster):
    """
    Création d'un DataFrame pour stocker les données et les labels
    Input : data_3d (array-like) : données converties en 3D
            l_true_label (list) : liste des labels vrais
            l_cluster (list) : liste des labels de cluster
            l_path_img (list) : liste des chemins des images
    Output : df (DataFrame) : DataFrame contenant les données et les labels
    """
    df = pd.DataFrame(data_3d, columns=['x', 'y', 'z'])
    df['label'] = l_true_label
    df['cluster'] = l_cluster
    
    return df
