import os
import cv2
import shutil
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Paths
ROOT_PATH = "D:/DataScience/THESIS/Data/"
DATA_PATH = ROOT_PATH + "HPA_segmentation/all/"
SAVE_PATH = ROOT_PATH + "HPA_segmentation/prepared/"
IMAGES_PATH = SAVE_PATH + "image/"
MASKS_PATH = SAVE_PATH + "mask/"
CSV_PATH = SAVE_PATH + "data_csv.txt"

# General
NUM_SEGMENTATION_CHANNELS = 3
NUM_VISUALIZATION_CHANNELS = 4


def merge_channels(data_dir, n_channels):
    """
    Read and stack individual images of given number of channels into RGB image.

    :param data_dir: data directory
    :param n_channels: number of channels to be stacked
    :return: Stacked image
    """
    stacked_images = []

    red_img = cv2.imread(data_dir + '/microtubules.png', cv2.IMREAD_UNCHANGED)
    yellow_img = cv2.imread(data_dir + '/er.png', cv2.IMREAD_UNCHANGED)
    green_img = cv2.imread(data_dir + '/protein.png', cv2.IMREAD_UNCHANGED)
    blue_img = cv2.imread(data_dir + '/nuclei.png', cv2.IMREAD_UNCHANGED)

    if n_channels == 3:
        stacked_images = np.stack((red_img, yellow_img, blue_img), axis=2)
    elif n_channels == 4:
        stacked_images = np.transpose(np.array([red_img, blue_img, green_img, yellow_img]), (1, 2, 0))
    else:
        RuntimeError("Wrong number of channels")

    return stacked_images


def save_merged_images(images_path, save_path, num_channels):
    """
    Save merged images into specific path.

    :param images_path: images directory
    :param save_path: directory to save merged images
    :param num_channels: number of channels to be merged
    :return: None
    """
    for subdir, dirs, files in os.walk(images_path):
        for _dir in dirs:
            merged_image = merge_channels(subdir + _dir, num_channels)
            plt.imsave(save_path + f"/{_dir}.png", merged_image, format="png")


def save_renamed_masks(masks_path, save_mask_path):
    """
    Save renamed mask image into specific path.

    :param masks_path: masks directory
    :param save_mask_path: directory to save renamed masks
    :return: None
    """
    for subdir, dirs, files in os.walk(masks_path):
        for _dir in dirs:
            shutil.copyfile(subdir + _dir + "/cell_border_mask.png", save_mask_path + f"/{_dir}.png")


def create_dataset_csv(data_path, image_path, mask_path, csv_save_path):
    """
    Create *.txt file for easy accessing segmentation data. Data is organised
    in three columns: image_id, image_path, mask_path.

    :param data_path: data directory
    :param image_path: images directory
    :param mask_path: masks directory
    :param csv_save_path: directory to save csv file
    :return: None
    """
    ids = []
    images_paths = []
    masks_paths = []

    for subdir, dirs, files in os.walk(data_path):
        for _dir in dirs:
            ids.append(str(_dir))
            images_paths.append(ROOT_PATH + image_path + f"{_dir}.png")
            masks_paths.append(ROOT_PATH + mask_path + f"{_dir}.png")

    data_csv = pd.DataFrame({"id": ids, "image_path": images_paths, "mask_path": masks_paths})
    data_csv.to_csv(csv_save_path, header=True, index=False)


if __name__ == '__main__':
    save_merged_images(DATA_PATH, IMAGES_PATH, NUM_SEGMENTATION_CHANNELS)
    save_renamed_masks(DATA_PATH, MASKS_PATH)
    create_dataset_csv(DATA_PATH, IMAGES_PATH, MASKS_PATH, CSV_PATH)
