import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import IPython.display as display


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


def create_mask(prediction: float) -> tf.Tensor:
    """Temporary function for model evaluation test."""
    processed = tf.where(prediction >= 0.5, np.dtype('uint8').type(1), np.dtype('uint8').type(0))
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
        display_sample_images([image[1], mask[1], pred_mask[1]])


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