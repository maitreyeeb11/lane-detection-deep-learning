import os
import tensorflow as tf
from tensorflow.keras import backend as K

# ==============================
# PATHS
# ==============================

BASE_DATASET_PATH = os.path.join(os.getcwd(), "..", "dataset", "tusimple_processed")

test_images_folder = os.path.join(BASE_DATASET_PATH, "test", "images")
test_masks_folder  = os.path.join(BASE_DATASET_PATH, "test", "masks")

MODEL_PATH = "lane_model_partialfreeze_epoch5.keras"

print("Dataset Path:", BASE_DATASET_PATH)
print("Test Images:", test_images_folder)
print("Test Masks:", test_masks_folder)

# ==============================
# METRICS (same as training)
# ==============================

def dice_coeff(y_true, y_pred):
    y_true = K.flatten(y_true)
    y_pred = K.flatten(y_pred)
    intersection = K.sum(y_true * y_pred)
    return (2. * intersection + 1e-7) / (K.sum(y_true) + K.sum(y_pred) + 1e-7)


def dice_loss(y_true, y_pred):
    return 1 - dice_coeff(y_true, y_pred)


def iou_metric(y_true, y_pred):
    y_true = K.flatten(y_true)
    y_pred = K.flatten(y_pred)

    intersection = K.sum(y_true * y_pred)
    union = K.sum(y_true) + K.sum(y_pred) - intersection

    return (intersection + 1e-7) / (union + 1e-7)


# ==============================
# LOAD IMAGE + MASK
# ==============================

def loadImage(imagePath, maskPath):

    image = tf.io.read_file(imagePath)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, (224, 224)) / 255.0

    mask = tf.io.read_file(maskPath)
    mask = tf.image.decode_jpeg(mask, channels=1)
    mask = tf.image.resize(mask, (224, 224)) / 255.0

    return image, mask


def datasetFromFolder(imageFolder, maskFolder):

    imageFiles = sorted(os.listdir(imageFolder))
    maskFiles  = sorted(os.listdir(maskFolder))

    imagePaths = [os.path.join(imageFolder, f) for f in imageFiles]
    maskPaths  = [os.path.join(maskFolder, f) for f in maskFiles]

    dataset = tf.data.Dataset.from_tensor_slices((imagePaths, maskPaths))
    dataset = dataset.map(loadImage)

    return dataset


# ==============================
# LOAD TEST DATASET
# ==============================

BATCH_SIZE = 8

testDataset = datasetFromFolder(test_images_folder, test_masks_folder)
testDataset = testDataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

print("Test dataset loaded")

# ==============================
# LOAD MODEL
# ==============================

model = tf.keras.models.load_model(
    MODEL_PATH,
    custom_objects={
        "dice_coeff": dice_coeff,
        "dice_loss": dice_loss,
        "iou_metric": iou_metric
    }
)

print("Model Loaded Successfully")

# ==============================
# EVALUATE
# ==============================

results = model.evaluate(testDataset)

print("\nValidation Results:")

for name, value in zip(model.metrics_names, results):
    print(f"{name}: {value:.4f}")