import tensorflow as tf
import matplotlib.pyplot as plt

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
    Augment images by flip rotations and normalize them.

    :param data: dict containing an image and its mask
    :return: Processed imaged and its mask.
    """
    input_image = tf.keras.preprocessing.image.load_img(data['image'])
    input_mask = tf.keras.preprocessing.image.load_img(data['segmentation_mask'])

    if tf.random.uniform(()) > 0.5:
        input_image = tf.image.flip_left_right(input_image)
        input_mask = tf.image.flip_left_right(input_mask)

    input_image, input_mask = normalize_images(input_image, input_mask)

    return input_image, input_mask


@tf.function
def process_test_images(data: dict, image_size: int) -> tuple:
    """
    Normalize and resize test image and mask tensors.

    :param data: dict containing an image and its mask
    :param image_size: size of the image tensor
    :return: Processed image and its mask.
    """
    input_image = tf.image.resize(data['image'], (image_size, image_size))
    input_mask = tf.image.resize(data['segmentation_mask'], (image_size, image_size))

    input_image, input_mask = normalize_images(input_image, input_mask)

    return input_image, input_mask


def display_sample(images_list: list) -> None:
    """
    Displays image and its mask.

    :param images_list:
    :return: None
    """
    plt.figure(figsize=(18, 18))

    title = ['Input Image', 'True Mask', 'Predicted Mask']

    for i in range(len(images_list)):
        plt.subplot(1, len(images_list), i+1)
        plt.title(title[i])
        plt.imshow(tf.keras.preprocessing.image.array_to_img(images_list[i]))
        plt.axis('off')
    plt.show()

