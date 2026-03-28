# ==========================================
# SCRIPT 1 — DATASET PREPARATION
# Extract Frames + Generate Masks + Split
# ==========================================

import os
import cv2
import shutil
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split

# ------------------------------------------
# PATHS
# ------------------------------------------
BASE_PATH = os.path.join(os.getcwd(), "..", "dataset")
TUSIMPLE_PATH = os.path.join(BASE_PATH, "TUSimple", "train_set")
CLIPS_PATH = os.path.join(TUSIMPLE_PATH, "clips")

PROCESSED_PATH = os.path.join(BASE_PATH, "tusimple_processed")
IMAGE_FOLDER = os.path.join(PROCESSED_PATH, "images")
MASK_FOLDER = os.path.join(PROCESSED_PATH, "masks")

TRAIN_IMAGE = os.path.join(PROCESSED_PATH, "train", "images")
TRAIN_MASK = os.path.join(PROCESSED_PATH, "train", "masks")
TEST_IMAGE = os.path.join(PROCESSED_PATH, "test", "images")
TEST_MASK = os.path.join(PROCESSED_PATH, "test", "masks")

for path in [IMAGE_FOLDER, MASK_FOLDER, TRAIN_IMAGE, TRAIN_MASK, TEST_IMAGE, TEST_MASK]:
    os.makedirs(path, exist_ok=True)

# ------------------------------------------
# STEP 1 — Extract Frames
# ------------------------------------------
print("Extracting frames...")

for clip in os.listdir(CLIPS_PATH):
    clip_path = os.path.join(CLIPS_PATH, clip)

    for folder in os.listdir(clip_path):
        frame_path = os.path.join(clip_path, folder, "20.jpg")

        if not os.path.isfile(frame_path):
            continue

        temp = frame_path.replace("\\", "/")[:-7].split("/")[-2:]
        new_name = f"{temp[0]}_{temp[1]}.jpg"

        shutil.copy(frame_path, os.path.join(IMAGE_FOLDER, new_name))

print("Total Images:", len(os.listdir(IMAGE_FOLDER)))

# ------------------------------------------
# STEP 2 — Generate Masks
# ------------------------------------------
print("Generating masks...")

df1 = pd.read_json(os.path.join(TUSIMPLE_PATH, "label_data_0313.json"), lines=True)
df2 = pd.read_json(os.path.join(TUSIMPLE_PATH, "label_data_0531.json"), lines=True)
df3 = pd.read_json(os.path.join(TUSIMPLE_PATH, "label_data_0601.json"), lines=True)
df = pd.concat([df1, df2, df3])

for _, row in tqdm(df.iterrows(), total=len(df)):
    mask = np.zeros((720, 1280, 1), dtype=np.uint8)

    h_samples = row.h_samples
    lanes = row.lanes
    raw_file = row.raw_file

    for lane in lanes:
        y_filtered = [y for x, y in zip(lane, h_samples) if x != -2]
        x_filtered = [x for x in lane if x != -2]

        lane_points = np.array(list(zip(x_filtered, y_filtered)))

        if len(lane_points) > 1:
            cv2.polylines(mask, [lane_points], False, (255, 255, 255), 15)

    temp = raw_file.replace("\\", "/")[:-7].split("/")[-2:]
    mask_name = f"{temp[0]}_{temp[1]}.jpg"

    cv2.imwrite(os.path.join(MASK_FOLDER, mask_name), mask)

# ------------------------------------------
# STEP 3 — Train/Test Split
# ------------------------------------------
images = [f for f in os.listdir(IMAGE_FOLDER) if f.endswith(".jpg")]

train_images, test_images = train_test_split(images, test_size=0.2, random_state=42)

for file in train_images:
    shutil.move(os.path.join(IMAGE_FOLDER, file), os.path.join(TRAIN_IMAGE, file))
    shutil.move(os.path.join(MASK_FOLDER, file), os.path.join(TRAIN_MASK, file))

for file in test_images:
    shutil.move(os.path.join(IMAGE_FOLDER, file), os.path.join(TEST_IMAGE, file))
    shutil.move(os.path.join(MASK_FOLDER, file), os.path.join(TEST_MASK, file))

print("Dataset preparation completed.")