# ==========================================
# SCRIPT 2 — MODEL ARCHITECTURE (VGG-UNet)
# ==========================================

import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Conv2D, UpSampling2D, Concatenate, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K

# ------------------------------------------
# MODEL
# ------------------------------------------
def VGG_UNet(input_shape=(224, 224, 3)):

    def ConvBlock(filters, x):
        x = Conv2D(filters, (3,3), padding="same", activation="relu")(x)
        x = BatchNormalization()(x)
        x = Dropout(0.3)(x)
        return x

    vgg = VGG16(weights="imagenet", include_top=False, input_shape=input_shape)

    inputs = vgg.input
    s1 = vgg.get_layer("block1_conv2").output
    s2 = vgg.get_layer("block2_conv2").output
    s3 = vgg.get_layer("block3_conv3").output
    s4 = vgg.get_layer("block4_conv3").output
    b1 = vgg.get_layer("block5_conv3").output

    d1 = Concatenate()([UpSampling2D()(b1), s4])
    d1 = ConvBlock(512, d1)

    d2 = Concatenate()([UpSampling2D()(d1), s3])
    d2 = ConvBlock(256, d2)

    d3 = Concatenate()([UpSampling2D()(d2), s2])
    d3 = ConvBlock(128, d3)

    d4 = Concatenate()([UpSampling2D()(d3), s1])
    d4 = ConvBlock(64, d4)

    outputs = Conv2D(1, (1,1), activation="sigmoid")(d4)

    return Model(inputs, outputs)

# ------------------------------------------
# METRICS
# ------------------------------------------
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