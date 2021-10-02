import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from unet_architectures import unet_shallow, unet_pp, unet_mobilenet
from datetime import datetime
from utils.image_processing import process_train_images
from utils.callback import DisplayCallback
from utils.loss_functions import combined_dice_iou_loss, iou, dice, jaccard_distance_loss
from utils.configuaration import config_data_pipeline_performance
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.callbacks import ModelCheckpoint

# Paths
DATA_PATH = "D:/DataScience/THESIS/Data/HPA_segmentation/FINAL_DATA/train/"
BEST_MODEL_PATH_UNET = "D:/DataScience/THESIS/models/unet_best_model.hdf5"
BEST_MODEL_PATH_UNETPP = "D:/DataScience/THESIS/models/unetpp_best_model.hdf5"
BEST_MODEL_PATH_UNETFT = "D:/DataScience/THESIS/models/unetft_best_model.hdf5"
TENSORBOARD_LOGS_PATH = 'D:\\DataScience\\THESIS\\models\\logs\\'

# General
AUTOTUNE = tf.data.experimental.AUTOTUNE
BATCH_SIZE = 1
BUFFER_SIZE = 512
SEED = 42
EPOCHS = 500
EARLY_STOP_PATIENCE = 20
TRAIN_RATIO = 0.90
VAL_RATIO = 0.1
TEST_DATASET = False

# Image parameters
IMAGE_HEIGHT = 512
IMAGE_WIDTH = 512

IMAGE_CHANNELS = 3

# Model parameters
MODEL_ARCHITECTURES = ["STANDARD_UNET", "UNETPP", "FT_UNET"]
UNET_ARCHITECTURE = MODEL_ARCHITECTURES[2]
LOSSES = [combined_dice_iou_loss, jaccard_distance_loss]
LOSS = LOSSES[1]
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
    mask = tf.where(mask == 41, np.dtype('uint8').type(1), np.dtype('uint8').type(0))

    return {'image': image, 'segmentation_mask': mask}


def create_dataset(data_path: str) -> tuple:
    """
    Creates dataset for image segmentation.

    :param data_path: path to utils dictionary
    :return: Tuple with dataset, train dataset size and validation dataset size
    """

    full_dataset = tf.data.Dataset.list_files(data_path + "image/*.png", seed=SEED)
    full_dataset = full_dataset.map(parse_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    full_dataset = full_dataset.map(process_train_images, num_parallel_calls=tf.data.experimental.AUTOTUNE).repeat(3)

    dataset_size = tf.data.experimental.cardinality(full_dataset).numpy()
    train_dataset_size = int(TRAIN_RATIO * dataset_size)
    val_dataset_size = int(VAL_RATIO * dataset_size)

    train_dataset = full_dataset.take(train_dataset_size)
    remaining = full_dataset.skip(train_dataset_size)
    val_dataset = remaining.take(val_dataset_size)

    dataset = {"train": train_dataset, "val": val_dataset}

    dataset['train'] = config_data_pipeline_performance(dataset['train'], True, BUFFER_SIZE, BATCH_SIZE, SEED, AUTOTUNE)
    dataset['val'] = config_data_pipeline_performance(dataset['val'], False, BUFFER_SIZE, BATCH_SIZE, SEED, AUTOTUNE)

    return dataset, train_dataset_size, val_dataset_size


def build_model(img_height: int, img_width: int, img_channels: int, loss: tf.keras.losses.Loss,
                optimizer: tf.keras.optimizers.Optimizer, metrics: list):
    """
    Build and compile model.

    :param img_height: height of the image
    :param img_width: width of the image
    :param img_channels: number of image channels
    :param loss: loss function
    :param optimizer: optimizer
    :param metrics: metrics for training
    :return: Build and compiled model.
    """
    if UNET_ARCHITECTURE == MODEL_ARCHITECTURES[0]:
        model = unet_shallow.Unet(img_height=img_height, img_width=img_width, img_channels=img_channels)
    elif UNET_ARCHITECTURE == MODEL_ARCHITECTURES[1]:
        model = unet_pp.UnetPP(img_height=img_height, img_width=img_width, img_channels=img_channels)
    else:
        model = unet_mobilenet.UnetMobilenet(img_height=img_height, img_width=img_width, img_channels=img_channels)

    model.compile(loss_function=loss, optimizer=optimizer, metrics=metrics)

    return model


def make_callbacks(sample_images: list, early_stop_patience: int, save_model_path: str):
    """
    Make list of callbacks used during training.

    :param sample_images: sample images used for DisplayCallback
    :param early_stop_patience: number of epochs with no improvement after which training will be stopped
    :param save_model_path: path for saving the best model from ModelCheckpoint callback
    :return: List of callbacks.
    """
    log_dir = TENSORBOARD_LOGS_PATH + datetime.now().strftime("%Y%m%d-%H%M%S")
    callbacks = [
        DisplayCallback(sample_images, displaying_freq=10, enable_displaying=False),
        EarlyStopping(monitor="val_loss", patience=early_stop_patience, mode="min", verbose=1),
        ModelCheckpoint(filepath=save_model_path, monitor="val_loss", verbose=1, save_best_only=True, mode="min"),
        TensorBoard(log_dir=log_dir, histogram_freq=1, profile_batch=0)
    ]
    return callbacks


def plot_history(model_history: tf.keras.callbacks.History) -> None:
    """

    :param model_history: dictionary caring history records of model training
    :return: None
    """
    loss = model_history.history['loss']
    val_loss = model_history.history['val_loss']

    plt.figure()
    plt.plot(model_history.epoch, loss, 'r', label='Training loss')
    plt.plot(model_history.epoch, val_loss, 'b', label='Validation loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss Value')
    plt.ylim([0, 1])
    plt.legend()
    plt.show()


if __name__ == '__main__':
    print_device_info()
    segmentation_dataset, train_size, val_size = create_dataset(DATA_PATH)
    samples = segmentation_dataset['train'].take(1)

    if UNET_ARCHITECTURE == MODEL_ARCHITECTURES[0]:
        model_path = BEST_MODEL_PATH_UNET
    elif UNET_ARCHITECTURE == MODEL_ARCHITECTURES[1]:
        model_path = BEST_MODEL_PATH_UNETPP
    else:
        model_path = BEST_MODEL_PATH_UNETFT

    callbacks_list = make_callbacks(sample_images=samples,
                                    early_stop_patience=EARLY_STOP_PATIENCE,
                                    save_model_path=model_path)

    unet = build_model(img_width=IMAGE_WIDTH, img_height=IMAGE_HEIGHT, img_channels=IMAGE_CHANNELS, loss=LOSS,
                       optimizer=OPTIMIZER, metrics=METRICS)

    with tf.device("device:GPU:0"):
        history = unet.train(dataset=segmentation_dataset, train_size=train_size, val_size=val_size,
                             batch_size=BATCH_SIZE, epochs=EPOCHS, callbacks=callbacks_list)

    plot_history(history)
