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
    
