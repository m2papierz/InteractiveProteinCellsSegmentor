import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import IPython.display as display
import tensorflow.keras.backend as K

IMAGE_HEIGHT = 512
IMAGE_WIDTH = 512


@tf.function
def normalize_images(input_image: tf.Tensor, input_mask: tf.Tensor) -> tuple:
    """
    Normalize input tensors. Rescales images pixels values into 0.0 and 1.0 compared to [0,255].

    :param input_image: input image tensor
    :param input_mask: input mask tensor
    :return: Normalized tensors.
    """
    input_image = tf.cast(input_image, tf.float32) / 255.0
    return input_image, input_mask


@tf.function
def process_train_images(data: dict) -> tuple:
    """
    Resize and randomly augment architectures_unet.

    :param data: dict containing an image and its mask
    :return: Processed imaged and its mask.
    """
    input_image = tf.image.resize(data['image'], (IMAGE_HEIGHT, IMAGE_WIDTH))
    input_mask = tf.image.resize(data['segmentation_mask'], (IMAGE_HEIGHT, IMAGE_WIDTH))

    if tf.random.uniform(()) > 0.55:
        input_image = tf.image.central_crop(input_image, 0.7)
        input_mask = tf.image.central_crop(input_mask, 0.7)
        input_image = tf.image.resize(input_image, (512, 512))
        input_mask = tf.image.resize(input_mask, (512, 512))

    if tf.random.uniform(()) > 0.5:
        input_image = tf.image.random_brightness(input_image, 0.3)
        input_image = tf.image.random_contrast(input_image, 0.15, 0.4)

    if tf.random.uniform(()) > 0.5:
        input_image = tf.image.flip_left_right(input_image)
        input_mask = tf.image.flip_left_right(input_mask)

    if tf.random.uniform(()) > 0.5:
        input_image = tf.image.flip_up_down(input_image)
        input_mask = tf.image.flip_up_down(input_mask)

    input_image, input_mask = normalize_images(input_image, input_mask)

    return input_image, input_mask


@tf.function
def process_test_images(data: dict) -> tuple:
    """
    Normalize and resize test image and mask tensors.

    :param data: dict containing an image and its mask
    :return: Processed image and its mask.
    """
    input_image = tf.image.resize(data['image'], (IMAGE_HEIGHT, IMAGE_WIDTH))
    input_mask = tf.image.resize(data['segmentation_mask'], (IMAGE_HEIGHT, IMAGE_WIDTH))

    input_image, input_mask = normalize_images(input_image, input_mask)

    return input_image, input_mask


def config_data_pipeline_performance(dataset: tf.data.Dataset, training: bool, buffer_size: int, batch_size: int,
                                     seed: int, autotune: int) -> tf.data.Dataset:
    """
    Configure the architectures_unet pipeline for its performance enhancement.

    :param dataset: dataset to be configured
    :param training: a boolean which if true indicates that the dataset set is the training one
    :param buffer_size: size of the buffer
    :param batch_size: size of the batch
    :param seed: random seed for creation of the distribution
    :param autotune: maximum number of elements that will be used when prefetching
    :return: Configured dataset
    """
    if training:
        dataset = dataset.shuffle(buffer_size=buffer_size, seed=seed)
    dataset = dataset.repeat()
    dataset = dataset.batch(batch_size=batch_size)
    dataset = dataset.prefetch(buffer_size=autotune)

    return dataset


def display_sample_images(images_list: list) -> None:
    """
    Display sample image and its mask from the dataset.

    :param images_list: list of images to be displayed
    :return: None
    """
    plt.figure(figsize=(18, 18))
    title = ['Input image', 'True mask', 'Prediction mask', 'Processed prediction mask']

    for i in range(len(images_list)):
        plt.subplot(1, len(images_list), i + 1)
        plt.title(title[i])
        plt.imshow(tf.keras.preprocessing.image.array_to_img(images_list[i]))
        plt.axis('off')
    plt.show()


def process_prediction(prediction: float) -> tf.Tensor:
    """Temporary function for model evaluation test."""
    processed = tf.where(prediction >= 0.95, np.dtype('uint8').type(1), np.dtype('uint8').type(0))
    return processed


def show_predictions(model: tf.keras.Model, sample_images: tuple) -> None:
    """
    Show prediction from a sample image.

    :param model: model for predictions
    :param sample_images: dataset with sample images
    :return: None
    """
    for image, mask in sample_images:
        pred_mask = model.predict(image)
        display_sample_images([image[0], mask[0], pred_mask[0]])


@tf.function
def iou(y_true, y_pred, smooth=1) -> float:
    """
    Intersection over union loss function.

    :param y_true: truth tensor
    :param y_pred: prediction tensor
    :param smooth: parameter for numerical stability to avoid divide by zero errors
    :return: Intersection over union loss
    """
    intersection = K.sum(K.abs(y_true * y_pred), axis=[1, 2, 3])
    union = K.sum(y_true, [1, 2, 3]) + K.sum(y_pred, [1, 2, 3]) - intersection
    return K.mean((intersection + smooth) / (union + smooth), axis=0)


@tf.function
def dice(y_true, y_pred, smooth=1) -> float:
    """
    Dice loss function.

    :param y_true: truth tensor
    :param y_pred: prediction tensor
    :param smooth: parameter for numerical stability to avoid divide by zero errors
    :return: Dice loss
    """
    intersection = K.sum(y_true * y_pred, axis=[1, 2, 3])
    union = K.sum(y_true, axis=[1, 2, 3]) + K.sum(y_pred, axis=[1, 2, 3])
    return K.mean((2. * intersection + smooth) / (union + smooth), axis=0)


@tf.function
def combined_dice_iou_loss(y_true, y_pred, iou_weight=1, dice_weight=1):
    """
    Loss function combining binary crossentropy loss, dice loss and intersection over union loss.

    :param y_true: truth tensor
    :param y_pred: prediction tensor
    :param iou_weight: weight of the intersection over union loss
    :param dice_weight: weight of the dice loss
    :return: Combined loss
    """
    log_dice = -K.log(dice(y_true, y_pred))
    log_iou = - K.log(iou(y_true, y_pred))

    return iou_weight * log_iou + dice_weight * log_dice


@tf.function
def jaccard_distance_loss(y_true, y_pred, smooth=100):
    """
    Jaccard distance for semantic segmentation also known as the intersection-over-union loss.
    This implementation is adapted for semantic segmentation.

    :param y_true: truth tensor
    :param y_pred: prediction tensor
    :param smooth: parameter for numerical stability to avoid divide by zero error
    :return: Jaccard loss
    """
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    sum_ = K.sum(K.abs(y_true) + K.abs(y_pred), axis=-1)
    jac = (intersection + smooth) / (sum_ - intersection + smooth)
    return (1 - jac) * smooth


class DisplayCallback(tf.keras.callbacks.Callback):
    """
    Callbacks displaying sample images and predictions for them each 5th epoch.
    """

    def __init__(self, sample_images, displaying_freq=5, enable_displaying=False):
        super().__init__()
        self.sample_images = sample_images
        self.displaying_freq = displaying_freq
        self.enable_displaying = enable_displaying

    def on_epoch_end(self, epoch, logs=None):
        display.clear_output(wait=True)
        if self.enable_displaying:
            if ((epoch + 1) % self.displaying_freq) == 0:
                show_predictions(self.model, self.sample_images)
