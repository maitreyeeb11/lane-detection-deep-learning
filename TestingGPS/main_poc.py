# ==========================================
# main_poc.py
# Lane Visibility + IDI + Severity + Overlay + CSV Extraction
# ==========================================

import os
import cv2
import pandas as pd
import numpy as np
import datetime

from lane_predictor import predict_mask
from lane_kpi import compute_visibility
from alert_engine import AlertEngine
from geo_sender import send_to_authority


# ------------------------------------------
# CONFIGURATION
# ------------------------------------------
IMAGE_FOLDER = r"D:\LaneDetection\dataset\laneImagesTest"
VISIBILITY_THRESHOLD = 0.50
OUTPUT_LOG = "lane_health_log.csv"


# ------------------------------------------
# LOAD GPS DATA
# ------------------------------------------
gps_path = os.path.join(IMAGE_FOLDER, "gps_data.csv")
gps_df = pd.read_csv(gps_path)

gps_dict = dict(
    zip(gps_df["image_name"], zip(gps_df["latitude"], gps_df["longitude"]))
)

# ------------------------------------------
# INITIALIZE ALERT ENGINE
# ------------------------------------------
engine = AlertEngine(threshold=VISIBILITY_THRESHOLD)

# ------------------------------------------
# SORT IMAGES NUMERICALLY
# ------------------------------------------
image_files = sorted(
    [f for f in os.listdir(IMAGE_FOLDER) if f.endswith(".jpg")],
    key=lambda x: int(x.replace("image", "").replace(".jpg", ""))
)

print(f"\nRunning lane visibility analysis on {len(image_files)} images\n")

# ------------------------------------------
# Prepare Data Storage for BI
# ------------------------------------------
rows = []

# ------------------------------------------
# MAIN LOOP
# ------------------------------------------
for img_name in image_files:

    img_path = os.path.join(IMAGE_FOLDER, img_name)
    image = cv2.imread(img_path)

    if image is None:
        continue

    original = image.copy()
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Predict mask
    pred_mask = predict_mask(image)

    # Compute visibility
    visibility = compute_visibility(pred_mask)

    # Compute IDI
    idi = 1 - visibility

    # Severity Classification
    if visibility >= 0.75:
        status = "GOOD"
        color = (0, 255, 0)        # Green
    elif 0.50 <= visibility < 0.75:
        status = "MODERATE"
        color = (0, 255, 255)      # Yellow
    elif 0.30 <= visibility < 0.50:
        status = "POOR"
        color = (0, 165, 255)      # Orange
    else:
        status = "CRITICAL"
        color = (0, 0, 255)        # Red

    print(f"{img_name} → Visibility: {visibility:.3f} | IDI: {idi:.3f} | Status: {status}")

    # Overlay Mask
    binary_mask = (pred_mask > 0.3).astype(np.uint8)
    binary_mask = cv2.resize(binary_mask, (original.shape[1], original.shape[0]))

    mask_color = np.zeros_like(original)
    mask_color[:, :, 1] = binary_mask * 255

    overlay = cv2.addWeighted(original, 1.0, mask_color, 0.6, 0)

    # Put Text Overlay
    cv2.putText(overlay,
                f"Visibility: {visibility:.2f}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                color,
                2)

    cv2.putText(overlay,
                f"IDI: {idi:.2f}",
                (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                color,
                2)

    cv2.putText(overlay,
                f"Status: {status}",
                (20, 120),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                color,
                2)

    cv2.imshow("Lane Health Monitoring", overlay)
    cv2.waitKey(800)

    # Get GPS
    latitude, longitude = gps_dict.get(img_name, (None, None))

    timestamp = datetime.datetime.now()

    # Store row for BI dashboard
    rows.append({
        "image_name": img_name,
        "latitude": latitude,
        "longitude": longitude,
        "visibility": round(float(visibility), 3),
        "IDI": round(float(idi), 3),
        "status": status,
        "timestamp": timestamp
    })

    # Alert if below threshold
    if engine.check(visibility):
        send_to_authority(latitude, longitude, visibility)

# ------------------------------------------
# SAVE CSV FOR BI
# ------------------------------------------
df = pd.DataFrame(rows)
df.to_csv(OUTPUT_LOG, index=False)

cv2.destroyAllWindows()

print("\nlane_health_log.csv generated successfully.")
print("Processing complete.")