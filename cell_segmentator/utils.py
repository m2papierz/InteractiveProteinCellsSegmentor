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
    Augment resize images and augment by flip rotations.

    :param data: dict containing an image and its mask
    :return: Processed imaged and its mask.
    """
    input_image = tf.image.resize(data['image'], (512, 512))
    input_mask = tf.image.resize(data['segmentation_mask'], (512, 512))

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


def config_data_pipeline_performance(dataset: tf.data.Dataset, training: bool, buffer_size: int, batch_size: int,
                                     seed: int, autotune: int) -> tf.data.Dataset:
    """
    Configure the data pipeline for its performance enhancement.

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


def display_sample(dataset: tf.data.Dataset) -> None:
    """
    Displays sample image and its mask from the dataset.

    :param dataset from which sample images are to be displayed
    :return: None
    """
    sample_image = sample_mask = []

    for image, mask in dataset['train'].take(1):
        sample_image, sample_mask = image, mask

    images_list = [sample_image[0], sample_mask[0]]

    plt.figure(figsize=(18, 18))
    title = ['Input image', 'True mask']

    for i in range(len(images_list)):
        plt.subplot(1, len(images_list), i+1)
        plt.title(title[i])
        plt.imshow(tf.keras.preprocessing.image.array_to_img(images_list[i]))
        plt.axis('off')
    plt.show()

