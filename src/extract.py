# ==========================================
# PHASE 3 (REVISED): Lane Visibility & Continuity ONLY
# TensorFlow / Keras – Research-SAFE Script
# ==========================================

import os
import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage.morphology import skeletonize

# ------------------------------------------
# PATHS
# ------------------------------------------
BASE_DATASET_PATH = os.path.join(os.getcwd(), "..", "dataset", "tusimple_processed")
TEST_IMG_FOLDER = os.path.join(BASE_DATASET_PATH, "test", "images")
MODEL_PATH = os.path.join(os.getcwd(), "lane_model_partialfreeze_epoch5.keras")

print("Model path:", MODEL_PATH)
print("Test image folder:", TEST_IMG_FOLDER)

# ------------------------------------------
# LOAD MODEL
# ------------------------------------------
model = tf.keras.models.load_model(MODEL_PATH, compile=False)
print("Model loaded successfully")

# ------------------------------------------
# LOAD TEST DATASET (RANDOM IMAGE EACH RUN)
# ------------------------------------------
testDataset = tf.keras.utils.image_dataset_from_directory(
    TEST_IMG_FOLDER,
    labels=None,
    image_size=(224, 224),
    batch_size=1,
    shuffle=True
)

print("Test dataset loaded")

# ------------------------------------------
# STEP 0: RANDOM IMAGE → MODEL PREDICTION
# ------------------------------------------
sample = testDataset.take(1)

for img in sample:
    print("Random image mean:", tf.reduce_mean(img).numpy())
    pred = model.predict(img)

# ------------------------------------------
# STEP 1: Prediction → Binary Lane Mask
# ------------------------------------------
pred_mask = pred[0, :, :, 0]

print("Prediction range:", pred_mask.min(), pred_mask.max())

THRESHOLD = 0.3
binary_mask = (pred_mask > THRESHOLD).astype(np.uint8)

lane_pixel_count = np.sum(binary_mask)
print("Lane pixels detected:", lane_pixel_count)

H, W = binary_mask.shape

# ------------------------------------------
# STEP 2: Visualize Binary Mask
# ------------------------------------------
plt.imshow(binary_mask, cmap="gray")
plt.title("Binary Lane Mask")
plt.axis("off")
plt.show()

# ------------------------------------------
# STEP 3: Skeletonization
# ------------------------------------------
skeleton = skeletonize(binary_mask).astype(np.uint8)
print("Skeleton pixels:", np.sum(skeleton))

plt.imshow(skeleton, cmap="gray")
plt.title("Skeletonized Lanes")
plt.axis("off")
plt.show()

# ------------------------------------------
# STEP 4: Connected Components (Lane Segments)
# ------------------------------------------
num_labels, labels = cv2.connectedComponents(skeleton)
print("Connected components found:", num_labels - 1)

# ------------------------------------------
# STEP 5: Lane Visibility & Continuity
# ------------------------------------------
MIN_PIXELS = 40        # remove noise
MAX_Y_RATIO = 0.6      # ignore top-half clutter

lanes = []

for label in range(1, num_labels):

    ys, xs = np.where(labels == label)

    if len(xs) < MIN_PIXELS:
        continue

    y_min, y_max = ys.min(), ys.max()

    # Ignore lanes too high in the image
    if (y_max / H) < MAX_Y_RATIO:
        continue

    # -----------------------------
    # KPI 1: Lane Visibility
    # -----------------------------
    lane_length = y_max - y_min
    visibility = lane_length / H
    visibility = np.clip(visibility, 0, 1)

    # -----------------------------
    # KPI 2: Lane Continuity
    # -----------------------------
    y_sorted = np.sort(ys)
    gaps = np.diff(y_sorted)
    continuity = 1 - (np.sum(gaps > 3) / len(y_sorted))
    continuity = np.clip(continuity, 0, 1)

    lanes.append({
        "lane_id": label,
        "pixels": len(xs),
        "visibility": round(float(visibility), 3),
        "continuity": round(float(continuity), 3),
        "y_bounds": (int(y_min), int(y_max))
    })

# ------------------------------------------
# STEP 6: PRINT KPI RESULTS
# ------------------------------------------
print("\n========== LANE KPIs (FINAL) ==========")

if len(lanes) == 0:
    print("No valid lanes detected.")
else:
    for lane in lanes:
        print(f"\nLane ID: {lane['lane_id']}")
        print(f" Pixels     : {lane['pixels']}")
        print(f" Visibility : {lane['visibility']}")
        print(f" Continuity : {lane['continuity']}")
        print(f" Y bounds   : {lane['y_bounds']}")

# ------------------------------------------
# STEP 7: OPTIONAL SUMMARY (DASHBOARD READY)
# ------------------------------------------
if len(lanes) > 0:
    avg_visibility = np.mean([l["visibility"] for l in lanes])
    avg_continuity = np.mean([l["continuity"] for l in lanes])

    print("\n====== IMAGE-LEVEL SUMMARY ======")
    print("Average Visibility :", round(float(avg_visibility), 3))
    print("Average Continuity :", round(float(avg_continuity), 3))
