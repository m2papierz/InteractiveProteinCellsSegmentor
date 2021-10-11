import tensorflow as tf
import matplotlib.pyplot as plt

from datetime import datetime
from utils.image_processing import process_train_images
from utils.callback import DisplayCallback
from utils.loss_functions import combined_dice_iou_loss, iou, dice, jaccard_distance_loss
from utils.configuaration import config_data_pipeline_performance, read_yaml_file
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.callbacks import ModelCheckpoint
from unet_architectures import unet_shallow, unet_pp, unet_mobilenet


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
    image = tf.image.decode_png(image)
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)

    mask_path = tf.strings.regex_replace(image_path, "image", "mask")
    mask = tf.io.read_file(mask_path)
    mask = tf.image.decode_png(mask)
    mask = tf.image.convert_image_dtype(mask, dtype=tf.float32)
    mask = tf.image.resize(mask, (512, 512))

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


def build_model(model_arch: str, img_height: int, img_width: int, img_channels: int, loss: tf.keras.losses.Loss,
                optimizer: tf.keras.optimizers.Optimizer, metrics: list):
    """
    Build and compile model.

    :param model_arch: name of the u-net model to be used
    :param img_height: height of the image
    :param img_width: width of the image
    :param img_channels: number of image channels
    :param loss: loss function
    :param optimizer: optimizer
    :param metrics: metrics for training
    :return: Build and compiled model.
    """
    if model_arch == "UNET_SHALLOW":
        model = unet_shallow.Unet(img_height=img_height, img_width=img_width, img_channels=img_channels)
    elif model_arch == "UNET_PP":
        model = unet_pp.UnetPP(img_height=img_height, img_width=img_width, img_channels=img_channels)
    elif model_arch == "UNET_MOBILENET":
        model = unet_mobilenet.UnetMobilenet(img_height=img_height, img_width=img_width, img_channels=img_channels)
    else:
        raise NotImplementedError()

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

    config = read_yaml_file("./config.yaml")

    # Paths
    DATA_PATH = config["DATA_PATH_TRAIN"]
    MODELS_PATH = config["MODELS_PATH"]
    UNET_MODEL_PATH = config["UNET_MODEL_PATH"]
    UNETPP_MODEL_PATH = config["UNETPP_MODEL_PATH"]
    UNET_MOBILENET_MODEL_PATH = config["UNET_MOBILENET_MODEL_PATH"]
    TENSORBOARD_LOGS_PATH = config["TENSORBOARD_LOGS_PATH"]

    # Train parameters
    BATCH_SIZE = config["BATCH_SIZE"]
    BUFFER_SIZE = config["BUFFER_SIZE"]
    SEED = config["SEED"]
    EPOCHS = config["EPOCHS"]
    EARLY_STOP_PATIENCE = config["EARLY_STOP_PATIENCE"]
    TRAIN_RATIO = config["TRAIN_RATIO"]
    VAL_RATIO = config["VAL_RATIO"]
    AUTOTUNE = tf.data.experimental.AUTOTUNE

    # Image parameters
    IMAGE_HEIGHT = config["IMAGE_HEIGHT"]
    IMAGE_WIDTH = config["IMAGE_WIDTH"]
    IMAGE_CHANNELS = config["IMAGE_CHANNELS"]

    # Model parameters
    MODELS = config["MODELS"]
    UNET_SHALLOW = config["UNET_SHALLOW"]
    UNET_PP = config["UNET_PP"]
    UNET_MOBILENET = config["UNET_MOBILENET"]
    OPTIMIZER = config["OPTIMIZER"]
    METRICS = [iou, dice]

    # Loss functions
    LOSSES = [combined_dice_iou_loss, jaccard_distance_loss]
    COMBINED_LOSS = config["COMBINED_LOSS"]
    JACCARD_LOSS = config["JACCARD_LOSS"]

    if UNET_SHALLOW:
        model_path = MODELS_PATH + UNET_MODEL_PATH
        model_name = MODELS[0]
    elif UNET_PP:
        model_path = MODELS_PATH + UNETPP_MODEL_PATH
        model_name = MODELS[1]
    elif UNET_MOBILENET:
        model_path = MODELS_PATH + UNET_MOBILENET_MODEL_PATH
        model_name = MODELS[2]
    else:
        raise NotImplementedError()

    if COMBINED_LOSS:
        LOSS = LOSSES[0]
    elif JACCARD_LOSS:
        LOSS = LOSSES[1]
    else:
        raise NotImplementedError()

    segmentation_dataset, train_size, val_size = create_dataset(DATA_PATH)
    samples = segmentation_dataset['train'].take(1)

    callbacks_list = make_callbacks(sample_images=samples, early_stop_patience=EARLY_STOP_PATIENCE,
                                    save_model_path=model_path)

    unet = build_model(model_arch=model_name, img_width=IMAGE_WIDTH, img_height=IMAGE_HEIGHT,
                       img_channels=IMAGE_CHANNELS, loss=LOSS,  optimizer=OPTIMIZER, metrics=METRICS)

    with tf.device("device:GPU:0"):
        history = unet.train(dataset=segmentation_dataset,
                             train_size=train_size,
                             val_size=val_size,
                             batch_size=BATCH_SIZE,
                             epochs=EPOCHS,
                             callbacks=callbacks_list)

    plot_history(history)
