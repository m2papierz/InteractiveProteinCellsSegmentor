import os
import sys
import numpy as np
from PIL import Image

from tqdm import tqdm
from bs4 import BeautifulSoup
from utils.configuaration import read_yaml_file


def euclidean_distance(p1: tuple, p2: tuple, scale=1.0) -> float:
    """
    Calculates euclidean distance between two points.

    :param p1: first point coordinates
    :param p2: second point coordinates
    :param scale: scale factor
    :return: euclidean distance
    """
    distance = 0
    for dim in range(len(p1)):
        distance += (p1[dim] - p2[dim]) ** 2
    return np.sqrt(distance) * scale


def create_guidance_map(shape: tuple, points: list, scale=1.0, image=False, omega=255.0) -> np.array:
    """
    Creates guidance map for given points being scaled reverted Euclidean distance map.

    :param shape: shape of the output map
    :param points: list of points coordinates
    :param scale: scale factor
    :param image: flag indicating if created map will be save as image
    :param omega: max map value
    :return: gaussian map array
    """
    dm = np.full(shape, omega)

    for p in points:
        x_min = 0
        x_max = shape[0]
        y_min = 0
        y_max = shape[1]

        for x in range(p[0], 0, -1):
            if euclidean_distance((x, p[1]), (p[0], p[1]), scale) > omega:
                x_min = x
                break
        for x in range(p[0], shape[0]):
            if euclidean_distance((x, p[1]), (p[0], p[1]), scale) > omega:
                x_max = x
                break

        for y in range(p[1], 0, -1):
            if euclidean_distance((p[0], y), (p[0], p[1]), scale) > omega:
                y_min = y
                break
        for y in range(p[1], shape[1]):
            if euclidean_distance((p[0], y), (p[0], p[1]), scale) > omega:
                y_max = y
                break

        for x in range(x_min, x_max):
            for y in range(y_min, y_max):
                if image:
                    dm[x, y] = min(dm[x, y], euclidean_distance((x, y), (p[0], p[1]), scale))
                else:
                    dm[y, x] = min(dm[y, x], euclidean_distance((y, x), (p[1], p[0]), scale))

    return np.abs(np.array(dm) - 255.0)


def get_annotations_dict(data_xml: BeautifulSoup) -> dict:
    """
    Creates dictionaries with point annotations.

    :param data_xml: BeautifulSoup instance of xml file with annotations
    :return: tuple of dictionaries with train and test point annotations
    """
    images = data_xml.find_all("image")

    ann_dict = {}
    for image in images:
        image_dict = {}
        for i, point in enumerate(image.find_all("points")):
            coordinates = list(map(lambda x: int(float(x)), point["points"].split(",")))[::-1]
            image_dict.update({point["label"] + str(i): coordinates})
        ann_dict.update({image["name"]: image_dict})

    return ann_dict


def create_and_save_guidance_maps(xml_files_path: str, pos_save: str, neg_save: str) -> None:
    """
    Creates and saves guidance maps images of positive and negative clicks for given annotations.

    :param xml_files_path: dictionary with points annotations
    :param pos_save: path to save positive clicks maps
    :param neg_save: path to save negative clicks maps
    """
    for i, xml_file in enumerate(os.listdir(xml_files_path)):
        with open(xml_files_path + xml_file, 'r') as xml:
            data = xml.read()
        ann_dict = get_annotations_dict(BeautifulSoup(data, "xml"))

        for filename, points_dict in tqdm(ann_dict.items(), total=len(ann_dict), file=sys.stdout):
            pos_coordinates = []
            neg_coordinates = []

            for label, coordinates in points_dict.items():
                if "pos_click" in label:
                    pos_coordinates.append(coordinates)
                if "neg_click" in label:
                    neg_coordinates.append(coordinates)

            pos_map = create_guidance_map(shape=(IMAGE_HEIGHT, IMAGE_WIDTH), points=pos_coordinates,
                                          image=True, scale=POS_CLICK_MAP_SCALE)
            neg_map = create_guidance_map(shape=(IMAGE_HEIGHT, IMAGE_WIDTH), points=neg_coordinates,
                                          image=True, scale=NEG_CLICK_MAP_SCALE)

            Image.fromarray(pos_map.astype(np.uint8)).save(pos_save + filename, "PNG")
            Image.fromarray(neg_map.astype(np.uint8)).save(neg_save + filename, "PNG")


if __name__ == '__main__':
    config = read_yaml_file("./config.yaml")

    PROJECT_PATH = config["PROJECT_PATH"]
    DATA_PATH_TRAIN = config["DATA_PATH_TRAIN"]
    DATA_PATH_TEST = config["DATA_PATH_TEST"]

    ANNOTATIONS_XML_TRAIN_PATH = PROJECT_PATH + config["ANNOTATIONS_XML_TRAIN_PATH"]
    ANNOTATIONS_XML_TEST_PATH = PROJECT_PATH + config["ANNOTATIONS_XML_TEST_PATH"]
    POS_CLICK_MAPS_TRAIN_PATH = PROJECT_PATH + DATA_PATH_TRAIN + config["POS_CLICK_MAPS_PATH"]
    NEG_CLICK_MAPS_TRAIN_PATH = PROJECT_PATH + DATA_PATH_TRAIN + config["NEG_CLICK_MAPS_PATH"]
    POS_CLICK_MAPS_TEST_PATH = PROJECT_PATH + DATA_PATH_TEST + config["POS_CLICK_MAPS_PATH"]
    NEG_CLICK_MAPS_TEST_PATH = PROJECT_PATH + DATA_PATH_TEST + config["NEG_CLICK_MAPS_PATH"]

    IMAGE_HEIGHT = config["IMAGE_HEIGHT"]
    IMAGE_WIDTH = config["IMAGE_WIDTH"]
    POS_CLICK_MAP_SCALE = config["POS_CLICK_MAP_SCALE"]
    NEG_CLICK_MAP_SCALE = config["NEG_CLICK_MAP_SCALE"]

    print("\n----- GENERATING TRAIN GUIDANCE MAPS -----")
    create_and_save_guidance_maps(xml_files_path=ANNOTATIONS_XML_TRAIN_PATH,
                                  pos_save=POS_CLICK_MAPS_TRAIN_PATH,
                                  neg_save=NEG_CLICK_MAPS_TRAIN_PATH)

    print("\n----- GENERATING TEST GUIDANCE MAPS -----")
    create_and_save_guidance_maps(xml_files_path=ANNOTATIONS_XML_TEST_PATH,
                                  pos_save=POS_CLICK_MAPS_TEST_PATH,
                                  neg_save=NEG_CLICK_MAPS_TEST_PATH)
