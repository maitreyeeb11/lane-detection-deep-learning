# ==========================================
# lane_kpi.py
# Visibility Calculation
# ==========================================

import numpy as np
import cv2
from skimage.morphology import skeletonize

BINARY_THRESHOLD = 0.3
MIN_PIXELS = 40
MAX_Y_RATIO = 0.6


def compute_visibility(pred_mask):

    binary_mask = (pred_mask > BINARY_THRESHOLD).astype(np.uint8)

    H, W = binary_mask.shape

    skeleton = skeletonize(binary_mask).astype(np.uint8)

    num_labels, labels = cv2.connectedComponents(skeleton)

    visibilities = []

    for label in range(1, num_labels):

        ys, xs = np.where(labels == label)

        if len(xs) < MIN_PIXELS:
            continue

        y_min, y_max = ys.min(), ys.max()

        if (y_max / H) < MAX_Y_RATIO:
            continue

        lane_length = y_max - y_min
        visibility = lane_length / H
        visibility = np.clip(visibility, 0, 1)

        visibilities.append(visibility)

    if len(visibilities) == 0:
        return 0

    return max(visibilities)