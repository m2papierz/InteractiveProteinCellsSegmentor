import tensorflow as tf
import matplotlib.pyplot as plt

from datetime import datetime
from utils.loss_and_metrics import iou, dice, JaccardLoss
from utils.configuaration import config_data_pipeline_performance, read_yaml_file
from utils.image_processing import parse_images
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.callbacks import ModelCheckpoint
from unet_architectures import unet_shallow, unet_dc, unet_dpn


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


def build_model(model_arch: str, img_height: int, img_width: int, in_channels: int, loss: tf.keras.losses.Loss,
                optimizer: tf.keras.optimizers.Optimizer, metrics: list):
    """
    Builds and compiles model.

    :param model_arch: name of the u-net model to be used
    :param img_height: height of the image
    :param img_width: width of the image
    :param in_channels: number of input channels
    :param loss: loss function
    :param optimizer: optimizer
    :param metrics: metrics for training
    :return: Build and compiled model.
    """
    if model_arch == "UNET_SHALLOW":
        model = unet_shallow.Unet(img_height=img_height, img_width=img_width, img_channels=in_channels)
    elif model_arch == "UNET_DC":
        model = unet_dc.UnetDC(img_height=img_height, img_width=img_width, img_channels=in_channels)
    elif model_arch == "UNET_DPN":
        model = unet_dpn.UnetDPN(img_height=img_height, img_width=img_width, img_channels=in_channels)
    else:
        raise NotImplementedError()

    model.compile(loss_function=loss, optimizer=optimizer, metrics=metrics)

    return model


def make_callbacks(early_stop_patience: int, save_model_path: str) -> list:
    """
    Makes list of callbacks for model training.

    :param early_stop_patience: number of epochs with no improvement after which training will be stopped
    :param save_model_path: path for saving the best model from ModelCheckpoint callback
    :return: list of callbacks
    """
    log_dir = TENSORBOARD_LOGS_PATH + datetime.now().strftime("%Y%m%d-%H%M%S")
    callbacks = [
        EarlyStopping(monitor="val_loss", patience=early_stop_patience, mode="min", verbose=1),
        ModelCheckpoint(filepath=save_model_path, monitor="val_loss", verbose=1, save_best_only=True, mode="min"),
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
    MODELS_PATH = PROJECT_PATH + config["MODELS_PATH"]
    UNET_MODEL_PATH = config["UNET_MODEL_PATH"]
    UNET_DC_MODEL_PATH = config["UNET_DC_MODEL_PATH"]
    UNET_DPN_MODEL_PATH = config["UNET_DPN_MODEL_PATH"]
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
    MODELS = config["MODELS"]
    UNET_SHALLOW = config["UNET_SHALLOW"]
    UNET_DC = config["UNET_DC"]
    UNET_DPN = config["UNET_DPN"]
    OPTIMIZER = config["OPTIMIZER"]
    METRICS = [iou, dice]

    if UNET_SHALLOW:
        model_path = MODELS_PATH + UNET_MODEL_PATH
        model_name = MODELS[0]
    elif UNET_DC:
        model_path = MODELS_PATH + UNET_DC_MODEL_PATH
        model_name = MODELS[1]
    elif UNET_DPN:
        model_path = MODELS_PATH + UNET_DPN_MODEL_PATH
        model_name = MODELS[2]
    else:
        raise NotImplementedError()

    segmentation_dataset, train_size, val_size = create_dataset(DATA_PATH_TRAIN)
    samples = segmentation_dataset['train'].take(1)

    callbacks_list = make_callbacks(early_stop_patience=EARLY_STOP_PATIENCE, save_model_path=model_path)

    unet = build_model(model_arch=model_name,
                       img_width=IMAGE_WIDTH,
                       img_height=IMAGE_HEIGHT,
                       in_channels=INPUT_CHANNELS,
                       loss=JaccardLoss(),
                       optimizer=OPTIMIZER,
                       metrics=METRICS)

    with tf.device("device:GPU:0"):
        history = unet.train(dataset=segmentation_dataset,
                             train_size=train_size,
                             val_size=val_size,
                             batch_size=BATCH_SIZE,
                             epochs=EPOCHS,
                             callbacks=callbacks_list)

    plot_history(history)
