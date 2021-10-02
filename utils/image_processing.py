import tensorflow as tf

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
    Resize and randomly augment utils.

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
