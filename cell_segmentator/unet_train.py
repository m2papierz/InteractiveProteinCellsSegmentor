import os
import numpy as np
import tensorflow as tf
import unet_model

from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.callbacks import ModelCheckpoint

# Image parameters
IMAGE_HEIGHT = 512
IMAGE_WIDTH = 512
IMAGE_CHANNELS = 4

# Network parameters
LOSS = tf.losses.binary_crossentropy
OPTIMIZER = tf.optimizers.Adam()
METRIC = tf.metrics.MeanIoU(num_classes=2)


def load_segmentation_data():
    pass


def load_model(img_height, img_width, img_ch, loss, optimizer, metric):
    model = unet_model.Unet().build(img_width=img_height, img_height=img_width, img_channels=img_ch)
    model.compile(loss=loss, optimizer=optimizer, metrics=[metric])
    model.summary()

    return model


def make_callbacks():
    callbacks = [
        EarlyStopping(monitor="val_loss", patience=3, mode="min", verbose=1),
        # ModelCheckpoint()
    ]
    return callbacks


def train_model():
    pass


if __name__ == '__main__':
    Unet_model = load_model(IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS, LOSS, OPTIMIZER, METRIC)
