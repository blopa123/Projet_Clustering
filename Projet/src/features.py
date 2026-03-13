import os
import cv2
import numpy as np
from skimage.feature import hog
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
        fd = hog(image, orientations=9, 
                 pixels_per_cell=(4, 4),
                 cells_per_block=(1, 1), 
                 visualize=False)
        descriptors.append(fd)
    return np.array(descriptors)


def compute_resnet50_descriptors(images, batch_size=32):
    """
    Extrait des descripteurs profonds avec ResNet50 (sans couche de classification).
    Input : images (array) : images en niveaux de gris ou RGB
            batch_size (int) : taille de batch pour l'inférence
    Output : descriptors (array) : vecteurs de features convolutionnelles
    """
    try:
        from tensorflow.keras.applications import ResNet50
        from tensorflow.keras.applications.resnet50 import preprocess_input
    except Exception as exc:
        raise ImportError(
            "TensorFlow/Keras est requis pour le descripteur ResNet50."
        ) from exc

    model = ResNet50(weights="imagenet", include_top=False, pooling="avg")

    processed_images = []
    for image in images:
        if image.ndim == 2:
            image_rgb = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_GRAY2RGB)
        elif image.ndim == 3 and image.shape[2] == 1:
            image_rgb = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_GRAY2RGB)
        else:
            # cv2 charge en BGR; conversion vers RGB pour ResNet50.
            image_rgb = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_BGR2RGB)

        image_resized = cv2.resize(image_rgb, (224, 224))
        processed_images.append(image_resized.astype(np.float32))

    processed_images = np.array(processed_images, dtype=np.float32)
    processed_images = preprocess_input(processed_images)

    descriptors = model.predict(processed_images, batch_size=batch_size, verbose=0)
    return np.array(descriptors)
    
