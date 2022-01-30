import os
import tensorflow as tf
import matplotlib.pyplot as plt

from datetime import datetime
from utils.loss_and_metrics import iou, dice, JaccardLoss
from utils.configuaration import config_data_pipeline_performance, read_yaml_file
from utils.image_processing import parse_images
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.callbacks import ModelCheckpoint
from unet_architectures.ShallowUnet import ShallowUnet
from unet_architectures.DualPathUnet import DualPathUnet


def create_dataset(data_path: str) -> tuple:
    """
    Creates train and validation datasets.

    :param data_path: path to the data
    :return: tuple with dataset, train dataset size and validation dataset size
    """

    full_dataset = tf.data.Dataset.list_files(data_path + "image/*.png", seed=SEED)
    full_dataset = full_dataset.map(parse_images, num_parallel_calls=tf.data.experimental.AUTOTUNE)

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


def build_model(img_height: int, img_width: int, in_channels: int, loss: tf.keras.losses.Loss,
                optimizer: tf.keras.optimizers.Optimizer, metrics: list):
    """
    Builds and compiles model.

    :param img_height: height of the image
    :param img_width: width of the image
    :param in_channels: number of input channels
    :param loss: loss function
    :param optimizer: optimizer
    :param metrics: metrics for training
    :return: Build and compiled model.
    """
    if UNET_SHALLOW:
        model = ShallowUnet(img_height=img_height, img_width=img_width, img_channels=in_channels, attention=ATTENTION)
    elif UNET_DUAL_PATH:
        model = DualPathUnet(img_height=img_height, img_width=img_width, img_channels=in_channels, attention=ATTENTION)
    else:
        raise NotImplementedError()

    model.compile(loss_function=loss, optimizer=optimizer, metrics=metrics)
    model.model.summary()

    return model


def make_callbacks(early_stop_patience: int, model: str) -> list:
    """
    Makes list of callbacks for model training.

    :param early_stop_patience: number of epochs with no improvement after which training will be stopped
    :param model: path for saving the best model from ModelCheckpoint callback
    :return: list of callbacks
    """
    save_path = os.path.join(MODELS_PATH, model + '.hdf5')
    log_dir = TENSORBOARD_LOGS_PATH + datetime.now().strftime("%Y%m%d-%H%M%S")

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=early_stop_patience, mode="min", verbose=1),
        ModelCheckpoint(filepath=save_path, monitor="val_loss", verbose=1, save_best_only=True, mode="min"),
        TensorBoard(log_dir=log_dir, histogram_freq=1, profile_batch=0)
    ]
    return callbacks


def plot_history(model_history: tf.keras.callbacks.History) -> None:
    """
    Plots training history.

    :param model_history: dictionary caring history records of the model training
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
    config = read_yaml_file("./config.yaml")

    # Paths
    PROJECT_PATH = config["PROJECT_PATH"]
    DATA_PATH_TRAIN = PROJECT_PATH + config["DATA_PATH_TRAIN"]
    MODELS_PATH = config["MODELS_PATH"]
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
    INPUT_CHANNELS = config["INPUT_CHANNELS"]

    # Model parameters
    UNET_SHALLOW = config["UNET_SHALLOW"]
    UNET_DUAL_PATH = config["UNET_DUAL_PATH"]
    ATTENTION = config['ATTENTION']
    OPTIMIZER = config["OPTIMIZER"]
    METRICS = [iou, dice]

    segmentation_dataset, train_size, val_size = create_dataset(DATA_PATH_TRAIN)

    unet = build_model(img_width=IMAGE_WIDTH,
                       img_height=IMAGE_HEIGHT,
                       in_channels=INPUT_CHANNELS,
                       loss=JaccardLoss(),
                       optimizer=OPTIMIZER,
                       metrics=METRICS)

    model_name = None
    if ATTENTION:
        model_name = unet.__class__.__name__ + '_attention'
    else:
        model_name = unet.__class__.__name__

    callbacks_list = make_callbacks(early_stop_patience=EARLY_STOP_PATIENCE,
                                    model=model_name)

    with tf.device("device:GPU:0"):
        history = unet.train(dataset=segmentation_dataset,
                             train_size=train_size,
                             val_size=val_size,
                             batch_size=BATCH_SIZE,
                             epochs=EPOCHS,
                             callbacks=callbacks_list)

    plot_history(history)
