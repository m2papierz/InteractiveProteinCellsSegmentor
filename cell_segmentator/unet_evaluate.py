import numpy as np
import tensorflow as tf

from glob import glob
from utils import show_predictions
from utils import combined_loss, iou_loss, dice_loss
from utils import process_test_images, config_data_pipeline_performance

# Paths
DATA_PATH = "D:/DataScience/THESIS/Data/HPA_segmentation/test/"
BEST_MODEL_PATH_UNET = "D:/DataScience/THESIS/models/unet_best_model.hdf5"
BEST_MODEL_PATH_UNETPP = "D:/DataScience/THESIS/models/unetpp_best_model.hdf5"

# General
BATCH_SIZE = 12
BUFFER_SIZE = 1024
SEED = 42
STANDARD_UNET = False
AUTOTUNE = tf.data.experimental.AUTOTUNE


def parse_image(image_path: str) -> dict:
    """
    Load an image and its mask.

    :param image_path: path to the image
    :return: Dictionary mapping an image and its mask.
    """
    image = tf.io.read_file(image_path)
    image = tf.image.decode_png(image, channels=3)
    image = tf.image.convert_image_dtype(image, tf.uint8)

    mask_path = tf.strings.regex_replace(image_path, "image", "mask")
    mask = tf.io.read_file(mask_path)
    mask = tf.image.decode_png(mask, channels=1)
    mask = tf.where(mask == 255, np.dtype('uint8').type(1), np.dtype('uint8').type(0))

    return {'image': image, 'segmentation_mask': mask}


def create_test_dataset(data_path: str) -> tuple:
    """
    Creates dataset for image segmentation.

    :param data_path: path to data dictionary
    :return: Tuple with dataset, train dataset size and validation dataset size
    """
    dataset_size = len(glob(data_path + "image/*.png"))

    full_dataset = tf.data.Dataset.list_files(data_path + "image/*.png", seed=SEED)
    test_dataset = full_dataset.take(dataset_size)
    test_dataset = test_dataset.map(parse_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    dataset = {"test": test_dataset}

    dataset['test'] = dataset['test'].map(process_test_images, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset['test'] = config_data_pipeline_performance(dataset['test'], False, BUFFER_SIZE, BATCH_SIZE, SEED, AUTOTUNE)

    return dataset, dataset_size


def evaluate_model(model: tf.keras.models.Model, dataset: dict, test_size: int, batch_size: int) -> None:
    """
    Calculate and print results of model evaluation.

    :param model: model to evaluate
    :param dataset: dataset used for evaluation
    :param test_size: size of test dataset
    :param batch_size: batch size
    :return: None
    """
    test_steps = tf.floor(test_size / batch_size)
    combined, iou, dice = model.evaluate(dataset['test'], steps=test_steps)

    print(f"Combined_loss: {combined}")
    print(f"IOU loss: {iou}")
    print(f"Dice loss: {dice}")


def main():
    segmentation_dataset, dataset_size = create_test_dataset(DATA_PATH)
    samples = segmentation_dataset['test'].take(1)
    custom_objects = {combined_loss.__name__: combined_loss,
                      iou_loss.__name__: iou_loss,
                      dice_loss.__name__: dice_loss}

    if STANDARD_UNET:
        best_model = tf.keras.models.load_model(BEST_MODEL_PATH_UNET, custom_objects=custom_objects)
    else:
        best_model = tf.keras.models.load_model(BEST_MODEL_PATH_UNETPP, custom_objects=custom_objects)

    evaluate_model(model=best_model,
                   dataset=segmentation_dataset,
                   test_size=dataset_size,
                   batch_size=BATCH_SIZE)

    show_predictions(model=best_model, sample_images=samples)


if __name__ == '__main__':
    main()
