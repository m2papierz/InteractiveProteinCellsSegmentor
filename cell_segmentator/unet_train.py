import os
import numpy as np
import tensorflow as tf

from glob import glob
from unet_model import Unet
from datetime import datetime
from utils import process_train_images, config_data_pipeline_performance, DisplayCallback
from utils import iou, dice, combined_iou_dice_loss, show_predictions
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.callbacks import ModelCheckpoint

# Paths
DATA_PATH = "D:/DataScience/THESIS/Data/HPA_segmentation/prepared/"
BEST_MODEL_PATH = "D:/DataScience/THESIS/models/best_segmentation_model.hdf5"
TENSORBOARD_LOGS_PATH = ""

# General
AUTOTUNE = tf.data.experimental.AUTOTUNE
BATCH_SIZE = 8
BUFFER_SIZE = 1024
SEED = 42
SEGMENTATION_IMAGE_CHANNELS = 3
EPOCHS = 200
EARLY_STOP_PATIENCE = 200
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15

# Image parameters
IMAGE_HEIGHT = 512
IMAGE_WIDTH = 512
IMAGE_CHANNELS = 3

# Train parameters
LOSS = combined_iou_dice_loss
OPTIMIZER = tf.keras.optimizers.Adam()
METRICS = [iou, dice]


def print_device_info():
    """
    Function printing tensorflow device info.

    :return: None
    """
    print("Tensorflow version: ", tf.__version__)

    if tf.test.is_built_with_cuda():
        physical_device = tf.config.list_physical_devices('GPU')
        tf.config.experimental.set_memory_growth(physical_device[0], True)
        print("Num GPUs Available: ", len(physical_device))
        print("Tensorflow built with CUDA (GPU) support.")
    else:
        print("Tensorflow is NOT built with CUDA (GPU) support.")


def parse_image(image_path: str) -> dict:
    """
    Load an image and its mask.

    :param image_path: path to the image
    :return: Dictionary mapping an image and its mask.
    """
    image = tf.io.read_file(image_path)
    image = tf.image.decode_png(image, channels=3)
    image = tf.image.convert_image_dtype(image, tf.uint8)

    mask_path = tf.strings.regex_replace(image_path, "image", "mask")
    mask = tf.io.read_file(mask_path)
    mask = tf.image.decode_png(mask, channels=1)
    mask = tf.where(mask == 255, np.dtype('uint8').type(1), np.dtype('uint8').type(0))

    return {'image': image, 'segmentation_mask': mask}


def create_dataset(data_path: str) -> tuple:
    """
    Creates dataset for image segmentation.

    :param data_path: path to data dictionary
    :return: Tuple with dataset, train dataset size and validation dataset size
    """
    dataset_size = len(glob(data_path + "image/*.png"))
    train_dataset_size = int(TRAIN_RATIO * dataset_size)
    val_dataset_size = int(VAL_RATIO * dataset_size)
    test_dataset_size = int(VAL_RATIO * dataset_size)

    full_dataset = tf.data.Dataset.list_files(data_path + "image/*.png", seed=SEED)
    train_dataset = full_dataset.take(train_dataset_size)
    remaining = full_dataset.skip(train_dataset_size)
    val_dataset = remaining.take(val_dataset_size)
    test_dataset = remaining.skip(test_dataset_size)

    train_dataset = train_dataset.map(parse_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    val_dataset = val_dataset.map(parse_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    dataset = {"train": train_dataset, "val": val_dataset, "test": test_dataset}

    dataset['train'] = dataset['train'].map(process_train_images, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset['val'] = dataset['val'].map(process_train_images, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    dataset['train'] = config_data_pipeline_performance(dataset['train'], True, BUFFER_SIZE, BATCH_SIZE, SEED, AUTOTUNE)
    dataset['val'] = config_data_pipeline_performance(dataset['val'], False, BUFFER_SIZE, BATCH_SIZE, SEED, AUTOTUNE)

    return dataset, train_dataset_size, val_dataset_size


def build_model(img_height: int, img_width: int, img_channels: int, loss: tf.keras.losses.Loss,
                optimizer: tf.keras.optimizers.Optimizer, metrics: list) -> Unet:
    model = Unet(img_height, img_width, img_channels)
    model.compile(loss_function=loss, optimizer=optimizer, metrics=metrics)
    return model


def make_callbacks(sample_images: list, early_stop_patience: int, save_model_path: str):
    callbacks = [
        DisplayCallback(sample_images, displaying_freq=10, enable_displaying=False),
        EarlyStopping(monitor="val_loss", patience=early_stop_patience, mode="min", verbose=1),
        ModelCheckpoint(filepath=save_model_path, monitor="val_loss", verbose=1, save_best_only=True)
    ]
    return callbacks


def main():
    print_device_info()
    segmentation_dataset, train_size, val_size = create_dataset(DATA_PATH)
    samples = segmentation_dataset['train'].take(1)

    callbacks_list = make_callbacks(sample_images=samples,
                                    early_stop_patience=EARLY_STOP_PATIENCE,
                                    save_model_path=BEST_MODEL_PATH)

    unet = build_model(img_width=IMAGE_WIDTH, img_height=IMAGE_HEIGHT, img_channels=IMAGE_CHANNELS, loss=LOSS,
                       optimizer=OPTIMIZER, metrics=METRICS)

    with tf.device("device:GPU:0"):
        unet.train(dataset=segmentation_dataset, train_size=train_size, val_size=val_size, batch_size=BATCH_SIZE,
                   epochs=EPOCHS, callbacks=callbacks_list)




if __name__ == '__main__':
    main()
