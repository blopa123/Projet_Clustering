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
        hist = cv2.calcHist([image_uint8], [0], None, [256], [0, 256])
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
        fd = hog(image, orientations=12, 
                 pixels_per_cell=(8, 8),
                 cells_per_block=(2, 2), 
                 visualize=False)
        descriptors.append(fd)
    return np.array(descriptors)


def compute_resnet50_descriptors(images, batch_size=32, layer_name="avg_pool"):
    """
    layer_name options :
      - "avg_pool"      : sortie standard 2048 dims (pooling avg)
      - "conv5_block3_out" : features conv avant pooling (7×7×2048)
      - "conv4_block6_out" : features plus génériques (14×14×1024)
    """
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    from tensorflow.keras.applications import ResNet50
    from tensorflow.keras.applications.resnet50 import preprocess_input
    from tensorflow.keras.layers import Concatenate, GlobalAveragePooling2D, GlobalMaxPooling2D
    from tensorflow.keras.models import Model

    base_model = ResNet50(weights="imagenet", include_top=False, pooling=None)

    # Avec include_top=False et pooling=None, avg_pool n'existe pas comme couche nommée.
    if layer_name == "avg_pool":
        output_layer = GlobalAveragePooling2D()(base_model.output)
        model = Model(inputs=base_model.input, outputs=output_layer)
    else:
        output_layer = base_model.get_layer(layer_name).output

        # Si la sortie est 3D (H×W×C), appliquer un pooling global enrichi.
        if len(output_layer.shape) == 4:
            out_avg = GlobalAveragePooling2D()(output_layer)
            out_max = GlobalMaxPooling2D()(output_layer)
            out = Concatenate()([out_avg, out_max])
            model = Model(inputs=base_model.input, outputs=out)
        else:
            model = Model(inputs=base_model.input, outputs=output_layer)

    # Prétraitement et inférence (identique à votre code actuel)
    processed = []
    for image in images:
        if image.ndim == 2:
            image = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_GRAY2RGB)
        else:
            image = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (224, 224)).astype(np.float32)
        processed.append(image)

    processed = preprocess_input(np.array(processed, dtype=np.float32))
    return model.predict(processed, batch_size=batch_size, verbose=0)
    

def compute_color_histograms_hsv(images_bgr):
    """
    Calcule des histogrammes HSV pondérés pour des images BGR.
    H : 36 bins  (couleur pure — le plus discriminant)
    S : 32 bins  (saturation)
    V : 16 bins  (luminosité — le moins discriminant)
    """
    descriptors = []
    for image in images_bgr:
        image_uint8 = image.astype(np.uint8)
        hsv = cv2.cvtColor(image_uint8, cv2.COLOR_BGR2HSV)

        # Histogrammes par canal avec bins adaptés à l'importance
        h_hist = cv2.calcHist([hsv], [0], None, [36], [0, 180]).flatten()
        s_hist = cv2.calcHist([hsv], [1], None, [32], [0, 256]).flatten()
        v_hist = cv2.calcHist([hsv], [2], None, [16], [0, 256]).flatten()

        descriptor = np.concatenate([h_hist, s_hist, v_hist])

        # Normalisation L1 — invariante à la taille de l'image
        descriptor = descriptor / (descriptor.sum() + 1e-7)

        descriptors.append(descriptor)
    return np.array(descriptors)


def compute_lbp_descriptors(images, radius=2, n_points=16, method="uniform"):
    """
    Calcule un descripteur LBP global par image via histogramme normalise.
    Input : images (array) : images en niveaux de gris
            radius (int) : rayon du voisinage LBP
            n_points (int) : nombre de points echantillonnes
            method (str) : methode LBP (uniform recommande)
    Output : descriptors (array) : histogrammes LBP normalises
    """
    descriptors = []
    n_bins = n_points + 2 if method == "uniform" else int(2 ** n_points)

    for image in images:
        image_uint8 = image.astype(np.uint8)
        lbp = local_binary_pattern(image_uint8, P=n_points, R=radius, method=method)
        hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins))
        hist = hist.astype(np.float32)
        hist /= (hist.sum() + 1e-7)
        descriptors.append(hist)

    return np.array(descriptors)
