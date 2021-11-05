import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


def display_sample_images(images_list: list) -> None:
    """
    Display sample image and its mask from the dataset.

    :param images_list: list of images to be displayed
    :return: None
    """
    plt.figure(figsize=(18, 18))
    title = ['Input image', 'True mask', 'Prediction mask']

    for i in range(len(images_list)):
        plt.subplot(1, len(images_list), i + 1)
        plt.title(title[i])
        if i == 0:
            plt.imshow(tf.keras.preprocessing.image.array_to_img(images_list[i]))
        else:
            plt.imshow(images_list[i])
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
        display_sample_images([image[0][:, :, :3], mask[0], pred_mask[0]])
