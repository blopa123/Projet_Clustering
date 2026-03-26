import os
import cv2
import numpy as np
from skimage.feature import hog
from skimage.feature import local_binary_pattern
from skimage import transform
import itertools


def compute_gray_histograms(images):
    """
    Calcule les histogrammes de niveau de gris pour les images MNIST.
    Input : images (list) : liste des images en niveaux de gris
    Output : descriptors (list) : liste des descripteurs d'histogrammes de niveau de gris
    """
    descriptors = []
    for image in images:
        # Convertir l'image en uint8 pour cv2.calcHist
        image_uint8 = image.astype(np.uint8)
        hist = cv2.calcHist([image_uint8], [0], None, [64], [0, 256])
        descriptors.append(hist.flatten())
    return np.array(descriptors)

def compute_hog_descriptors(images):
    """
    Calcule les descripteurs HOG pour les images en niveaux de gris.
    Input : images (array) : tableau numpy des images
    Output : descriptors (list) : liste des descripteurs HOG
    """
    descriptors = []
    for image in images:
        fd = hog(image, orientations=14, 
                 pixels_per_cell=(8, 8),
                 cells_per_block=(2, 2), 
                 visualize=False)
        descriptors.append(fd)
    return np.array(descriptors)


def compute_resnet50_descriptors(images, batch_size=32, layer_name="conv5_block3_out"):
    """
    Extrait des caractéristiques sémantiques profondes via un modèle ResNet50 pré-entraîné.
    
    Args:
        images (list/array): Liste d'images (niveaux de gris ou BGR).
        batch_size (int): Taille du lot pour l'inférence (optimise la mémoire GPU/CPU).
        layer_name (str): Nom de la couche de sortie. 
            - "avg_pool" : Sortie 1D (2048 dims).
            - "convX_blockY_out" : Sortie spatiale (nécessite un pooling supplémentaire).

    Returns:
        np.array: Descripteurs normalisés de taille (N, D).
    """
    # Désactivation des logs TensorFlow pour plus de clarté
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    
    from tensorflow.keras.applications import ResNet50
    from tensorflow.keras.applications.resnet50 import preprocess_input
    from tensorflow.keras.layers import Concatenate, GlobalAveragePooling2D, GlobalMaxPooling2D
    from tensorflow.keras.models import Model

    # Chargement du modèle sans la tête de classification (ImageNet weights)
    base_model = ResNet50(weights="imagenet", include_top=False, pooling=None)

    # 1. Gestion de la couche de sortie
    if layer_name == "avg_pool":
        output_layer = GlobalAveragePooling2D()(base_model.output)
        model = Model(inputs=base_model.input, outputs=output_layer)
    else:
        output_layer = base_model.get_layer(layer_name).output

        # Stratégie "Best of Both Worlds" : 
        # Si la sortie est spatiale (4D), on concatène l'AvgPooling (contexte) 
        # et le MaxPooling (détails saillants) pour un descripteur plus riche.
        if len(output_layer.shape) == 4:
            out_avg = GlobalAveragePooling2D()(output_layer)
            out_max = GlobalMaxPooling2D()(output_layer)
            out = Concatenate()([out_avg, out_max])
            model = Model(inputs=base_model.input, outputs=out)
        else:
            model = Model(inputs=base_model.input, outputs=output_layer)

    # 2. Prétraitement du lot d'images
    processed = []
    for image in images:
        # Conversion forcée en RGB (ResNet a été entraîné sur 3 canaux)
        if image.ndim == 2:
            image = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_GRAY2RGB)
        else:
            image = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_BGR2RGB)
        
        # Redimensionnement standard ResNet (224x224)
        image = cv2.resize(image, (224, 224)).astype(np.float32)
        processed.append(image)

    # Normalisation spécifique à ResNet (centrage et mise à l'échelle)
    processed = preprocess_input(np.array(processed, dtype=np.float32))
    
    # Inférence par lots pour la stabilité mémoire
    return model.predict(processed, batch_size=batch_size, verbose=0)


def compute_color_histograms_hsv(images_bgr):
    """
    Calcule des histogrammes multi-canaux dans l'espace HSV.
    Pondération : Teinte (48) > Saturation (32) > Valeur (16).
    
    Args:
        images_bgr (list/array): Images au format BGR original.
        
    Returns:
        np.array: Vecteur de caractéristiques concaténé et normalisé.
    """
    eps = 1e-7  # Évite la division par zéro
    descriptors = []
    
    for image in images_bgr:
        image_uint8 = image.astype(np.uint8)
        hsv = cv2.cvtColor(image_uint8, cv2.COLOR_BGR2HSV)

        # Calcul des histogrammes individuels. 
        # On utilise plus de 'bins' pour H car c'est l'info la plus stable pour identifier un produit.
        h_hist = cv2.calcHist([hsv], [0], None, [48], [0, 180]).flatten()
        s_hist = cv2.calcHist([hsv], [1], None, [32], [0, 256]).flatten()
        v_hist = cv2.calcHist([hsv], [2], None, [16], [0, 256]).flatten()
        
        # Normalisation par canal : rend le descripteur invariant à la taille de l'image
        h_hist = h_hist / (h_hist.sum() + eps)
        s_hist = s_hist / (s_hist.sum() + eps)
        v_hist = v_hist / (v_hist.sum() + eps)

        # Fusion des informations
        descriptor = np.concatenate([h_hist, s_hist, v_hist])

        # Normalisation L2 finale pour assurer une distance euclidienne cohérente en clustering
        descriptor = descriptor / (np.linalg.norm(descriptor) + eps)

        descriptors.append(descriptor)
        
    return np.array(descriptors)


def compute_lbp_descriptors(images, radius=2, n_points=16, method="uniform"):
    """
    Calcule la signature de texture via Local Binary Patterns (LBP).
    Utilise la méthode 'uniform' pour un descripteur compact et robuste au bruit.
    
    Args:
        images (list/array): Images en niveaux de gris.
        radius (int): Rayon du cercle de voisinage.
        n_points (int): Nombre de points échantillonnés sur le cercle.
        method (str): "uniform" garantit l'invariance par rotation simple.
        
    Returns:
        np.array: Histogrammes de texture normalisés.
    """
    descriptors = []
    # En méthode uniform, n_bins = n_points + 2 (les motifs uniformes + 1 pour les non-uniformes)
    n_bins = n_points + 2 if method == "uniform" else int(2 ** n_points)

    for image in images:
        image_uint8 = image.astype(np.uint8)
        
        # Calcul de la carte LBP
        lbp = local_binary_pattern(image_uint8, P=n_points, R=radius, method=method)
        
        # Construction de l'histogramme de fréquences des motifs de texture
        hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins))
        
        # Normalisation pour obtenir une distribution de probabilité (Somme = 1)
        hist = hist.astype(np.float32)
        hist /= (hist.sum() + 1e-7)
        
        descriptors.append(hist)

    return np.array(descriptors)
