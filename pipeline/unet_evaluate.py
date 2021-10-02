import numpy as np
import tensorflow as tf

from glob import glob
from utils.callback import show_predictions
from utils.loss_functions import combined_dice_iou_loss, iou, dice, jaccard_distance_loss
from utils.image_processing import process_test_images
from utils.configuaration import config_data_pipeline_performance

# Paths
DATA_PATH = "D:/DataScience/THESIS/Data/HPA_segmentation/FINAL_DATA/test_evaluate/"
BEST_MODEL_PATH_UNET = "D:/DataScience/THESIS/models/unet_best_model.hdf5"
BEST_MODEL_PATH_UNETPP = "D:/DataScience/THESIS/models/unetpp_best_model.hdf5"
BEST_MODEL_PATH_UNETFT = "D:/DataScience/THESIS/models/unetft_best_model.hdf5"

# General
BATCH_SIZE = 3
BUFFER_SIZE = 512
SEED = 42
MODEL_ARCHITECTURES = ["STANDARD_UNET", "UNETPP", "FT_UNET"]
UNET_ARCHITECTURE = MODEL_ARCHITECTURES[2]
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
    mask = tf.where(mask == 41, np.dtype('uint8').type(1), np.dtype('uint8').type(0))

    return {'image': image, 'segmentation_mask': mask}


def create_test_dataset(data_path: str) -> tuple:
    """
    Creates dataset for image segmentation.

    :param data_path: path to utils dictionary
    :return: Tuple with dataset, train dataset size and validation dataset size
    """
    dataset_size = len(glob(data_path + "image/*.png"))

    test_dataset = tf.data.Dataset.list_files(data_path + "image/*.png", seed=SEED)
    test_dataset = test_dataset.map(parse_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    test_dataset = test_dataset.map(process_test_images, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    dataset = {"test": test_dataset}
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
    loss, iou_score, dice_score = model.evaluate(dataset['test'], steps=test_steps)

    print(f"Loss: {loss}")
    print(f"IOU: {iou_score}")
    print(f"Dice: {dice_score}")


def main():
    segmentation_dataset, dataset_size = create_test_dataset(DATA_PATH)
    samples = segmentation_dataset['test'].take(1)
    custom_objects = {jaccard_distance_loss.__name__: jaccard_distance_loss,
                      combined_dice_iou_loss.__name__: combined_dice_iou_loss,
                      iou.__name__: iou,
                      dice.__name__: dice}

    if UNET_ARCHITECTURE == MODEL_ARCHITECTURES[0]:
        best_model = tf.keras.models.load_model(BEST_MODEL_PATH_UNET, custom_objects=custom_objects)
    elif UNET_ARCHITECTURE == MODEL_ARCHITECTURES[1]:
        best_model = tf.keras.models.load_model(BEST_MODEL_PATH_UNETPP, custom_objects=custom_objects)
    else:
        best_model = tf.keras.models.load_model(BEST_MODEL_PATH_UNETFT, custom_objects=custom_objects)

    evaluate_model(model=best_model,
                   dataset=segmentation_dataset,
                   test_size=dataset_size,
                   batch_size=BATCH_SIZE)

    show_predictions(model=best_model, sample_images=samples)


if __name__ == '__main__':
    main()
