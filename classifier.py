# classifier.py

import cv2
import numpy as np
import tensorflow as tf

# Load the fruit freshness classifier model
classifier_model = tf.keras.models.load_model('enhanced_fruit_freshness_classifier.h5')

def classify_freshness(roi):
    """Classify the freshness of the detected fruit (ROI) and return a freshness index."""
    # Convert the ROI to grayscale
    roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    
    # Resize the grayscale ROI to the required input size of the classifier
    roi_resized = cv2.resize(roi_gray, (224, 224))
    
    # Repeat the grayscale channel 3 times to simulate an RGB image
    roi_rgb = cv2.merge([roi_resized, roi_resized, roi_resized])
    
    # Normalize the image (same pre-processing as used during training)
    roi_normalized = roi_rgb / 255.0
    roi_normalized = roi_normalized.reshape(1, 224, 224, 3)  # Shape it as RGB (3 channels)

    # Make predictions with the classifier model
    predictions = classifier_model.predict(roi_normalized)
    freshness_probability = predictions[0][0]  # Probability of being 'Fresh'

    # Scale the probability to a freshness index (0 to 100)
    freshness_index = freshness_probability * 100  # 0% (Rotten) to 100% (Fresh)

    return freshness_index
