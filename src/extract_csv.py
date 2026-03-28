# ==========================================
# Lane Visibility & Continuity Extraction
# Train + Test Dataset → CSV (BI Ready)
# ==========================================

import os
import tensorflow as tf
import numpy as np
import cv2
import pandas as pd
from skimage.morphology import skeletonize

# ------------------------------------------
# PATHS
# ------------------------------------------
BASE_DATASET_PATH = os.path.join(os.getcwd(), "..", "dataset", "tusimple_processed")

TRAIN_IMG_FOLDER = os.path.join(BASE_DATASET_PATH, "train", "images")
TEST_IMG_FOLDER  = os.path.join(BASE_DATASET_PATH, "test", "images")

MODEL_PATH = os.path.join(os.getcwd(), "lane_model_partialfreeze_epoch5.keras")
OUTPUT_CSV = "lane_kpis_train_test.csv"

# ------------------------------------------
# LOAD MODEL
# ------------------------------------------
model = tf.keras.models.load_model(MODEL_PATH, compile=False)
print("Model loaded successfully")

# ------------------------------------------
# FUNCTION: PROCESS ONE DATASET
# ------------------------------------------
def process_dataset(image_folder, split_name, rows):

    dataset = tf.keras.utils.image_dataset_from_directory(
        image_folder,
        labels=None,
        image_size=(224, 224),
        batch_size=1,
        shuffle=False
    )

    THRESHOLD = 0.3
    MIN_PIXELS = 40
    MAX_Y_RATIO = 0.6

    for img in dataset:

        pred = model.predict(img, verbose=0)
        pred_mask = pred[0, :, :, 0]

        binary_mask = (pred_mask > THRESHOLD).astype(np.uint8)
        H, W = binary_mask.shape

        skeleton = skeletonize(binary_mask).astype(np.uint8)
        num_labels, labels = cv2.connectedComponents(skeleton)

        for lane_id in range(1, num_labels):

            ys, xs = np.where(labels == lane_id)

            if len(xs) < MIN_PIXELS:
                continue

            y_min, y_max = ys.min(), ys.max()

            # ignore upper-image clutter
            if (y_max / H) < MAX_Y_RATIO:
                continue

            # -------------------------
            # KPI 1: Lane Visibility
            # -------------------------
            visibility = (y_max - y_min) / H
            visibility = np.clip(visibility, 0, 1)

            # -------------------------
            # KPI 2: Lane Continuity
            # -------------------------
            y_sorted = np.sort(ys)
            gaps = np.diff(y_sorted)
            continuity = 1 - (np.sum(gaps > 3) / len(y_sorted))
            continuity = np.clip(continuity, 0, 1)

            rows.append({
                "dataset_split": split_name,        # train / test
                "lane_id": int(lane_id),
                "lane_visibility": round(float(visibility), 3),
                "lane_continuity": round(float(continuity), 3),
                "lane_pixel_count": int(len(xs))
            })

# ------------------------------------------
# MAIN PROCESSING
# ------------------------------------------
rows = []

print("Processing TRAIN dataset...")
process_dataset(TRAIN_IMG_FOLDER, "train", rows)

print("Processing TEST dataset...")
process_dataset(TEST_IMG_FOLDER, "test", rows)

# ------------------------------------------
# SAVE CSV
# ------------------------------------------
df = pd.DataFrame(
    rows,
    columns=[
        "dataset_split",
        "lane_id",
        "lane_visibility",
        "lane_continuity",
        "lane_pixel_count"
    ]
)

df.to_csv(OUTPUT_CSV, index=False)

print("====================================")
print("CSV saved:", OUTPUT_CSV)
print("Total lane records:", len(df))
print(df.head())
