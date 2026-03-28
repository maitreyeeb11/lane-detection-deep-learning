# ==========================================
# TRAIN MODEL (5 EPOCHS - SAFE VERSION)
# PARTIAL FREEZE
# SAVE METRICS PER EPOCH (LIVE)
# ==========================================

import os
import pandas as pd
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Conv2D, UpSampling2D, Concatenate, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, CSVLogger

# ------------------------------------------
# PATHS
# ------------------------------------------
BASE_DATASET_PATH = os.path.join(os.getcwd(),"..","dataset","tusimple_processed")

train_images_folder = os.path.join(BASE_DATASET_PATH,"train","images")
train_mask_folder   = os.path.join(BASE_DATASET_PATH,"train","masks")

test_images_folder  = os.path.join(BASE_DATASET_PATH,"test","images")
test_mask_folder    = os.path.join(BASE_DATASET_PATH,"test","masks")

# ------------------------------------------
# DATA LOADER
# ------------------------------------------
def loadImage(imagePath, maskPath):
    image = tf.io.read_file(imagePath)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, (224,224)) / 255.0

    mask = tf.io.read_file(maskPath)
    mask = tf.image.decode_jpeg(mask, channels=1)
    mask = tf.image.resize(mask, (224,224)) / 255.0

    return image, mask

def datasetFromFolder(imageFolder, maskFolder):
    imageFiles = sorted(os.listdir(imageFolder))
    maskFiles  = sorted(os.listdir(maskFolder))

    imagePaths = [os.path.join(imageFolder,f) for f in imageFiles]
    maskPaths  = [os.path.join(maskFolder,f) for f in maskFiles]

    dataset = tf.data.Dataset.from_tensor_slices((imagePaths,maskPaths))
    dataset = dataset.map(loadImage)
    return dataset

# ------------------------------------------
# DATASET
# ------------------------------------------
BATCH_SIZE = 8

trainDataset = datasetFromFolder(train_images_folder,train_mask_folder)
testDataset  = datasetFromFolder(test_images_folder,test_mask_folder)

trainDataset = trainDataset.shuffle(1000).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
testDataset  = testDataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

train_count = len(os.listdir(train_images_folder))
test_count  = len(os.listdir(test_images_folder))

print("Training Images:", train_count)
print("Testing Images :", test_count)

# ------------------------------------------
# MODEL
# ------------------------------------------
def VGG_UNet(input_shape=(224,224,3)):

    def ConvBlock(filters,x):
        x = Conv2D(filters,(3,3),padding='same',activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.3)(x)
        return x

    vgg = VGG16(weights='imagenet',include_top=False,input_shape=input_shape)

    # -------- PARTIAL FREEZE --------
    for layer in vgg.layers:
        if "block1" in layer.name or "block2" in layer.name or "block3" in layer.name:
            layer.trainable = False
        else:
            layer.trainable = True

    s1 = vgg.get_layer("block1_conv2").output
    s2 = vgg.get_layer("block2_conv2").output
    s3 = vgg.get_layer("block3_conv3").output
    s4 = vgg.get_layer("block4_conv3").output
    b1 = vgg.get_layer("block5_conv3").output

    d1 = Concatenate()([UpSampling2D()(b1), s4])
    d1 = ConvBlock(512,d1)

    d2 = Concatenate()([UpSampling2D()(d1), s3])
    d2 = ConvBlock(256,d2)

    d3 = Concatenate()([UpSampling2D()(d2), s2])
    d3 = ConvBlock(128,d3)

    d4 = Concatenate()([UpSampling2D()(d3), s1])
    d4 = ConvBlock(64,d4)

    outputs = Conv2D(1,(1,1),activation='sigmoid')(d4)

    return Model(vgg.input,outputs)

# ------------------------------------------
# METRICS
# ------------------------------------------
def dice_coeff(y_true,y_pred):
    y_true = K.flatten(y_true)
    y_pred = K.flatten(y_pred)
    intersection = K.sum(y_true*y_pred)
    return (2.*intersection+1e-7)/(K.sum(y_true)+K.sum(y_pred)+1e-7)

def dice_loss(y_true,y_pred):
    return 1-dice_coeff(y_true,y_pred)

def iou_metric(y_true,y_pred):
    y_true = K.flatten(y_true)
    y_pred = K.flatten(y_pred)
    intersection = K.sum(y_true*y_pred)
    union = K.sum(y_true)+K.sum(y_pred)-intersection
    return (intersection+1e-7)/(union+1e-7)

# ------------------------------------------
# COMPILE
# ------------------------------------------
model = VGG_UNet()

model.compile(
    optimizer='adam',
    loss=dice_loss,
    metrics=[dice_coeff,iou_metric,'accuracy']
)

print("Model Parameters:", model.count_params())

# ------------------------------------------
# CALLBACKS
# ------------------------------------------
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=3,
    restore_best_weights=True
)

# NEW: CSV Logger (updates after each epoch)
csv_logger = CSVLogger("training_metrics_epoch5.csv", append=False)

# ------------------------------------------
# TRAIN (ONLY 5 EPOCHS)
# ------------------------------------------
EPOCHS = 5

history = model.fit(
    trainDataset,
    validation_data=testDataset,
    epochs=EPOCHS,
    callbacks=[early_stop, csv_logger]
)

# ------------------------------------------
# SAVE MODEL
# ------------------------------------------
model.save("lane_model_partialfreeze_epoch5.keras")
print("Model saved as lane_model_partialfreeze_epoch5.keras")

print("\nTraining Completed Successfully")