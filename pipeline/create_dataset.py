import os
import sys
import tensorflow as tf

from tqdm import tqdm
from utils.configuaration import read_yaml_file


def merge_channels_to_rgb(data_dir_path: str, save_path: str, file_name: str) -> None:
    """
    Read and stack individual images into RGB image.

    :param data_dir_path: path to images to be stacked
    :param save_path: path to save image
    :param file_name: name of the image
    """
    red_img = tf.io.read_file(data_dir_path + '/microtubules.png')
    red_img = tf.image.decode_png(red_img)
    red_img = tf.image.convert_image_dtype(red_img, tf.float32)

    yellow_img = tf.io.read_file(data_dir_path + '/er.png')
    yellow_img = tf.image.decode_png(yellow_img)
    yellow_img = tf.image.convert_image_dtype(yellow_img, tf.float32)

    blue_img = tf.io.read_file(data_dir_path + '/nuclei.png')
    blue_img = tf.image.decode_png(blue_img)
    blue_img = tf.image.convert_image_dtype(blue_img, tf.float32)

    rgb_image = tf.concat([red_img, yellow_img, blue_img], axis=2)
    rgb_image = tf.image.resize(rgb_image, (512, 512))

    tf.keras.preprocessing.image.save_img(save_path + f"/{file_name}.png", rgb_image)


def preprocess_mask(mask_path: str, save_path: str, file_name: str) -> None:
    """
    Processes mask image.

    :param mask_path: path to mask
    :param save_path: path to save mask
    :param file_name: name of the image
    """

    mask_img = tf.io.read_file(mask_path)
    mask_img = tf.image.decode_png(mask_img, channels=1)
    mask_img = tf.image.resize(mask_img, (512, 512))
    tf.keras.preprocessing.image.save_img(save_path + f"/{file_name}.png", mask_img, data_format="channels_last")


def save_merged_images(images_path: str, save_path_train: str, save_path_test: str) -> None:
    """
    Save merged images.

    :param images_path: images directory
    :param save_path_train: directory to save merged train images
    :param save_path_test: directory to save merged test images
    """
    print("\n----- PROCESSING CELL IMAGES TRAIN -----")
    for subdir, dirs, files in tqdm(list(os.walk(images_path + "train/")), file=sys.stdout):
        for _dir in dirs:
            merge_channels_to_rgb(data_dir_path=subdir + _dir,
                                  save_path=save_path_train,
                                  file_name=_dir)

    print("\n----- PROCESSING CELL IMAGES TEST -----")
    for subdir, dirs, files in tqdm(list(os.walk(images_path + "test/")), file=sys.stdout):
        for _dir in dirs:
            merge_channels_to_rgb(data_dir_path=subdir + _dir,
                                  save_path=save_path_test,
                                  file_name=_dir)


def save_processed_masks(masks_path: str, save_path_train: str) -> None:
    """
    Save processed and renamed mask image into specific directory.

    :param masks_path: masks directory
    :param save_path_train: directory to save merged train images
    """
    print("\n----- PROCESSING CELL MASKS -----")
    for subdir, dirs, files in tqdm(list(os.walk(masks_path + "train/")), file=sys.stdout):
        for _dir in dirs:
            preprocess_mask(mask_path=subdir + _dir + "/cell_border_mask.png",
                            save_path=save_path_train,
                            file_name=_dir)


if __name__ == '__main__':
    physical_devices = tf.config.list_physical_devices('GPU')
    for device in physical_devices:
        tf.config.experimental.set_memory_growth(device, True)

    config = read_yaml_file("./config.yaml")

    DATA_PATH = config["DATA_PATH"]
    SAVE_PATH_IMAGE_TRAIN = config["IMAGES_PATH_TRAIN"]
    SAVE_PATH_MASK_TRAIN = config["MASKS_PATH_TRAIN"]
    SAVE_PATH_IMAGE_TEST = config["IMAGES_PATH_TEST"]

    save_merged_images(images_path=DATA_PATH,
                       save_path_train=SAVE_PATH_IMAGE_TRAIN,
                       save_path_test=SAVE_PATH_IMAGE_TEST)

    save_processed_masks(masks_path=DATA_PATH,
                         save_path_train=SAVE_PATH_MASK_TRAIN)
