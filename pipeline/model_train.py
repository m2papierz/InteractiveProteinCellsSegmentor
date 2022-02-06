import os.path
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
from unet_architectures.AttentionDualPath import AttentionDualPathUnet


def create_dataset(data_path: str) -> tuple:
    """
    Creates train and validation datasets.

    :param data_path: path to the data
    :return: tuple with dataset, train dataset size and validation dataset size
    """
    autotune = tf.data.experimental.AUTOTUNE

    full_dataset = tf.data.Dataset.list_files(data_path + "image/*.png", seed=seed)
    full_dataset = full_dataset.map(parse_images, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    dataset_size = tf.data.experimental.cardinality(full_dataset).numpy()
    train_dataset_size = int(train_val_ratio * dataset_size)
    val_dataset_size = int((1 - train_val_ratio) * dataset_size)

    train_dataset = full_dataset.take(train_dataset_size)
    remaining = full_dataset.skip(train_dataset_size)
    val_dataset = remaining.take(val_dataset_size)

    dataset = {"train": train_dataset, "val": val_dataset}

    dataset['train'] = config_data_pipeline_performance(
        dataset['train'], True, buffer_size, batch_size, seed, autotune)
    dataset['val'] = config_data_pipeline_performance(
        dataset['val'], False, buffer_size, batch_size, seed, autotune)

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
    if shallow:
        model = ShallowUnet(img_height=img_height, img_width=img_width, img_channels=in_channels)
    elif dual_path:
        model = DualPathUnet(img_height=img_height, img_width=img_width, img_channels=in_channels)
    elif attention_dual_path:
        model = AttentionDualPathUnet(img_height=img_height, img_width=img_width, img_channels=in_channels)
    else:
        raise NotImplementedError()

    model.compile(loss_function=loss, optimizer=optimizer, metrics=metrics)
    model.model.summary()

    return model


def make_callbacks(model_name: str) -> list:
    """
    Makes list of callbacks for model training.

    :param model_name: name of the model architecture
    :return: list of callbacks
    """
    save_path = os.path.join(models_dir, model_name + '.hdf5')
    log_dir = tensorboard_logs_dir + datetime.now().strftime("%Y%m%d-%H%M%S")

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
    project_dir = config["project_dir"]
    train_data_dir = project_dir + config["train_data_dir"]
    models_dir = config["models_dir"]
    tensorboard_logs_dir = config["tensorboard_logs_dir"]

    # Train parameters
    batch_size = config["batch_size"]
    buffer_size = config["buffer_size"]
    seed = config["seed"]
    epochs = config["epochs"]
    early_stop_patience = config["early_stop_patience"]
    train_val_ratio = config["train_val_ratio"]

    # Image parameters
    image_height = config["image_height"]
    image_width = config["image_width"]
    input_channels = config["input_channels"]

    # Model parameters
    shallow = config["shallow"]
    dual_path = config["dual_path"]
    attention_dual_path = config["attention_dual_path"]

    segmentation_dataset, train_size, val_size = create_dataset(train_data_dir)

    unet = build_model(img_width=image_height,
                       img_height=image_width,
                       in_channels=input_channels,
                       loss=JaccardLoss(),
                       optimizer=tf.optimizers.Adam(),
                       metrics=[iou, dice])

    callbacks_list = make_callbacks(model_name=unet.__class__.__name__)

    with tf.device("device:GPU:0"):
        history = unet.train(dataset=segmentation_dataset,
                             train_size=train_size,
                             val_size=val_size,
                             batch_size=batch_size,
                             epochs=epochs,
                             callbacks=callbacks_list)
