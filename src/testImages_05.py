import os
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

# ==============================
# MODEL
# ==============================

MODEL_PATH = "lane_model_partialfreeze_epoch5.keras"

# ==============================
# IMAGE FOLDER
# ==============================

IMAGE_FOLDER = r"D:\LaneDetection\dataset\laneImagesTest"

IMG_SIZE = (224,224)

# ==============================
# LOAD MODEL
# ==============================

model = tf.keras.models.load_model(MODEL_PATH, compile=False)

print("Model loaded successfully")

# ==============================
# LOAD IMAGE FUNCTION
# ==============================

def load_user_image(image_path):

    img = tf.io.read_file(image_path)
    img = tf.image.decode_image(img, channels=3)

    img = tf.image.resize(img, IMG_SIZE)

    img = img / 255.0

    return img


# ==============================
# DISPLAY RESULT
# ==============================

def show_prediction(image, mask):

    plt.figure(figsize=(10,5))

    plt.subplot(1,2,1)
    plt.title("Input Image")
    plt.imshow(image)
    plt.axis("off")

    plt.subplot(1,2,2)
    plt.title("Predicted Lane Mask")
    plt.imshow(mask, cmap="gray")
    plt.axis("off")

    plt.show()


# ==============================
# RUN PREDICTIONS
# ==============================

image_files = [f for f in os.listdir(IMAGE_FOLDER)
               if f.lower().endswith((".jpg",".jpeg",".png"))]

print("Running lane detection on", len(image_files), "images")

for img_name in image_files:

    img_path = os.path.join(IMAGE_FOLDER, img_name)

    image = load_user_image(img_path)

    pred = model.predict(tf.expand_dims(image, axis=0))[0]

    pred_mask = np.squeeze(pred)
    pred_mask = np.round(pred_mask)

    show_prediction(image, pred_mask)