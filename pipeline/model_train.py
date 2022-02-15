import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from datetime import datetime
from pipeline.loss_and_metrics import iou, dice, JaccardLoss
from utils.configuaration import config_data_pipeline_performance, read_yaml_file
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.callbacks import ModelCheckpoint

from model_architectures.ShallowUnet import ShallowUnet
from model_architectures.FCN import FCN
from model_architectures.AttentionDualPath import AttentionDualPathUnet


def parse_image(path: str, channels: int, mask=False) -> tf.Tensor:
    """
    Reads, decodes and converts image into tf.Tensor

    :param path: path to the image
    :param channels: number of image channels
    :param mask: flag indicating if segmentation mask is parsed
    :return: parsed image
    """
    image = tf.io.read_file(path)
    image = tf.image.decode_png(image, channels=channels)
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    if mask:
        image = tf.where(image > 0.0, np.dtype('float32').type(1),
                         np.dtype('float32').type(0))
    return image


def process_images(image_path: str) -> tuple:
    """
    Processes images fetched  by the tf.Dataset instance.

    :param image_path: path to the image (loaded automatically by tf.Dataset)
    :return: tuple with input tensor and segmentation mask tensor
    """
    pos_click_map_path = tf.strings.regex_replace(image_path, "image", "pos_click")
    neg_click_map_path = tf.strings.regex_replace(image_path, "image", "neg_click")
    mask_path = tf.strings.regex_replace(image_path, "image", "mask")

    # Parse images
    hpa_image = parse_image(path=image_path, channels=3) / 255.0
    pos_click_map = parse_image(path=pos_click_map_path, channels=1)
    neg_click_map = parse_image(path=neg_click_map_path, channels=1)
    seg_mask = parse_image(path=mask_path, channels=1, mask=True)

    input_tensor = tf.concat([hpa_image, pos_click_map, neg_click_map], axis=2)

    return input_tensor, seg_mask


def create_train_and_val_dataset(data_path: str) -> tuple:
    """
    Creates train and validation datasets.

    :param data_path: path to the images
    :return: tuple with dataset dictionary, train dataset size and validation dataset size
    """
    autotune = tf.data.experimental.AUTOTUNE

    full_dataset = tf.data.Dataset.list_files(data_path + "image/*.png", seed=seed)
    full_dataset = full_dataset.map(process_images, num_parallel_calls=autotune)

    dataset_size = tf.data.experimental.cardinality(full_dataset).numpy()
    train_dataset_size = int(train_val_ratio * dataset_size)
    val_dataset_size = dataset_size - train_dataset_size

    train_dataset = full_dataset.take(train_dataset_size)
    remaining = full_dataset.skip(train_dataset_size)
    val_dataset = remaining.take(val_dataset_size)

    dataset = {"train": train_dataset, "val": val_dataset}

    dataset['train'] = config_data_pipeline_performance(
        dataset=dataset['train'], shuffle=True, buffer_size=buffer_size,
        batch_size=batch_size, seed=seed)
    dataset['val'] = config_data_pipeline_performance(
        dataset=dataset['val'], shuffle=False, buffer_size=buffer_size,
        batch_size=batch_size, seed=seed)

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
    :param metrics: training and validation metrics
    :return: compiled
    """
    if shallow_unet:
        seg_model = ShallowUnet(img_height=img_height, img_width=img_width, img_channels=in_channels)
    elif fcn:
        seg_model = FCN(img_height=img_height, img_width=img_width, img_channels=in_channels)
    elif attention_dual_path:
        seg_model = AttentionDualPathUnet(img_height=img_height, img_width=img_width, img_channels=in_channels)
    else:
        raise NotImplementedError()

    seg_model.compile(loss_function=loss, optimizer=optimizer, metrics=metrics)
    seg_model.model.summary()

    return seg_model


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
    shallow_unet = config["shallow_unet"]
    fcn = config["fcn"]
    attention_dual_path = config["attention_dual_path"]

    # Create training and validation datasets
    segmentation_dataset, train_size, val_size = create_train_and_val_dataset(train_data_dir)

    # Build the segmentation model
    model = build_model(img_width=image_height,
                        img_height=image_width,
                        in_channels=input_channels,
                        loss=JaccardLoss(),
                        optimizer=tf.optimizers.Adam(),
                        metrics=[iou, dice])

    # Create list of callbacks
    callbacks_list = make_callbacks(model_name=model.__class__.__name__)

    # Train the model on GPU
    with tf.device("device:GPU:0"):
        history = model.train(dataset=segmentation_dataset,
                              train_size=train_size,
                              val_size=val_size,
                              batch_size=batch_size,
                              epochs=epochs,
                              callbacks=callbacks_list)
