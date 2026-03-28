# ==========================================
# lane_predictor.py
# Lane Segmentation Prediction Module
# ==========================================

import tensorflow as tf
import numpy as np
import cv2

MODEL_PATH = r"D:\LaneDetection\src\lane_model_partialfreeze_epoch5.keras"
IMG_SIZE = (224, 224)

# Load model once
model = tf.keras.models.load_model(MODEL_PATH, compile=False)
print("Lane model loaded successfully.")


def preprocess_image(image):
    image = cv2.resize(image, IMG_SIZE)
    image = image / 255.0
    image = np.expand_dims(image, axis=0)
    return image


def predict_mask(image):
    processed = preprocess_image(image)
    pred = model.predict(processed, verbose=0)
    return pred[0, :, :, 0]