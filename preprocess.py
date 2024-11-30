# preprocess.py

import cv2
import numpy as np

def preprocess_custom_features(image):
    """Custom feature extraction: Edge detection and black spots analysis."""
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    if gray_image.dtype != np.uint8:
        gray_image = (gray_image * 255).astype(np.uint8)
    
    edges = cv2.Canny(gray_image, threshold1=100, threshold2=200)
    _, black_spots = cv2.threshold(gray_image, 50, 255, cv2.THRESH_BINARY_INV)
    
    custom_feature_image = np.stack([gray_image, edges, black_spots], axis=-1)
    return custom_feature_image
