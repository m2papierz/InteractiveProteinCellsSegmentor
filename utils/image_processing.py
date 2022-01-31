import numpy as np
import tensorflow as tf


def parse_images(image_path: str) -> tuple:
    """
    Loads and processes input images.

    :param image_path: path to the image
    :return: tuple with an input and mask
    """
    image = tf.io.read_file(image_path)
    image = tf.image.decode_png(image, channels=3)
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    image = image / 255.0

    pos_click_map = tf.strings.regex_replace(image_path, "image", "pos_click")
    pos_click_map = tf.io.read_file(pos_click_map)
    pos_click_map = tf.image.decode_png(pos_click_map, channels=1)
    pos_click_map = tf.image.convert_image_dtype(pos_click_map, dtype=tf.float32)

    neg_click_map = tf.strings.regex_replace(image_path, "image", "neg_click")
    neg_click_map = tf.io.read_file(neg_click_map)
    neg_click_map = tf.image.decode_png(neg_click_map, channels=1)
    neg_click_map = tf.image.convert_image_dtype(neg_click_map, dtype=tf.float32)

    input_ = tf.concat([image, pos_click_map, neg_click_map], axis=2)

    mask_path = tf.strings.regex_replace(image_path, "image", "mask")
    mask = tf.io.read_file(mask_path)
    mask = tf.image.decode_png(mask, channels=1)
    mask = tf.where(mask > 0, np.dtype('float32').type(1), np.dtype('float32').type(0))

    return input_, mask
