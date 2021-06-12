import unet_model
import numpy as np
import tensorflow as tf

from glob import glob
from utils import process_train_images
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.callbacks import ModelCheckpoint

# Paths
DATA_PATH = "D:/DataScience/THESIS/Data/HPA_segmentation/prepared/"
BEST_MODEL_PATH = ""
TENSORBOARD_LOGS_PATH = ""

# General
AUTOTUNE = tf.data.experimental.AUTOTUNE
BATCH_SIZE = 32
BUFFER_SIZE = 1000
SEED = 42
SEGMENTATION_IMAGE_CHANNELS = 3

# Image parameters
IMAGE_HEIGHT = 512
IMAGE_WIDTH = 512
IMAGE_CHANNELS = 4

# Network parameters
LOSS = tf.losses.binary_crossentropy


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

    return {'image': image, 'segmentation_mask': mask}


def create_dataset(data_path: str) -> dict:
    """
    Creates dataset for image segmentation.

    :param data_path: path to data dictionary
    :return: Dictionary with full data splitted into train, test and validation datasets.
    """
    dataset_size = len(glob(data_path + "**/*.png"))

    train_size = int(0.7 * dataset_size)
    val_size = int(0.15 * dataset_size)
    test_size = int(0.15 * dataset_size)

    full_dataset = tf.data.Dataset.list_files(data_path + "**/*.png", seed=SEED)
    train_dataset = full_dataset.take(train_size)
    remaining = full_dataset.skip(train_size)
    val_dataset = remaining.take(val_size)
    test_dataset = remaining.skip(test_size)

    train_dataset.map(parse_image)
    val_dataset.map(parse_image)

    dataset = {"train": train_dataset, "val": val_dataset, "test": test_dataset}

    # Train dataset
    dataset['train'] = dataset['train'].map(process_train_images, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset['train'] = dataset['train'].shuffle(buffer_size=BUFFER_SIZE, seed=SEED)
    dataset['train'] = dataset['train'].repeat()
    dataset['train'] = dataset['train'].batch(BATCH_SIZE)
    dataset['train'] = dataset['train'].prefetch(buffer_size=AUTOTUNE)

    # Validation dataset
    dataset['val'] = dataset['val'].map(process_train_images)
    dataset['val'] = dataset['val'].repeat()
    dataset['val'] = dataset['val'].batch(BATCH_SIZE)
    dataset['val'] = dataset['val'].prefetch(buffer_size=AUTOTUNE)

    # Test dataset
    # TODO: Investigate if test dataset has to be also processed
    # dataset['test'] = dataset['test'].map(process_train_images)
    dataset['test'] = dataset['test'].repeat()
    dataset['test'] = dataset['test'].batch(BATCH_SIZE)
    dataset['test'] = dataset['test'].prefetch(buffer_size=AUTOTUNE)

    return dataset


def load_model(img_height, img_width, img_ch):
    model = unet_model.Unet(img_height, img_width, img_ch)
    model.model.summary()
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
    print_device_info()
    Unet_model = load_model(IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS)
    # seg_dataset = create_dataset(DATA_PATH)
