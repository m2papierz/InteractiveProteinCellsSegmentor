import sys
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from bs4 import BeautifulSoup
from utils.configuaration import read_yaml_file


def euclidean_distance(p1, p2, scale=2.5):
    distance = 0
    for dim in range(len(p1)):
        distance += (p1[dim] - p2[dim]) ** 2
    return np.sqrt(distance) * scale


def create_gaussian_distance_map(shape, points, omega=255.0):
    dm = np.full(shape, omega)

    for p in points:
        x_min = 0
        x_max = shape[0]
        y_min = 0
        y_max = shape[1]

        for x in range(p[0], 0, -1):
            if euclidean_distance((x, p[1]), (p[0], p[1])) > omega:
                x_min = x
                break
        for x in range(p[0], shape[0]):
            if euclidean_distance((x, p[1]), (p[0], p[1])) > omega:
                x_max = x
                break

        for y in range(p[1], 0, -1):
            if euclidean_distance((p[0], y), (p[0], p[1])) > omega:
                y_min = y
                break
        for y in range(p[1], shape[1]):
            if euclidean_distance((p[0], y), (p[0], p[1])) > omega:
                y_max = y
                break

        for x in range(x_min, x_max):
            for y in range(y_min, y_max):
                dm[x, y] = min(dm[x, y], euclidean_distance((x, y), (p[0], p[1])))

    return np.abs(np.array(dm) - 255.0)


def get_annotations_dict(data_xml) -> dict:
    images = data_xml.find_all("image")
    ann_dict = {}
    for image in images:
        image_dict = {}
        for i, point in enumerate(image.find_all("points")):
            coordinates = list(map(lambda x: int(float(x)), point["points"].split(",")))[::-1]
            image_dict.update({point["label"] + str(i): coordinates})
        ann_dict.update({image["name"]: image_dict})
    return ann_dict


def create_distance_maps(img_height: int, img_width: int, ann_dict: dict, pos_save: str, neg_save: str) -> None:
    for filename, points_dict in tqdm(ann_dict.items(), total=len(ann_dict), file=sys.stdout):
        pos_coordinates = []
        neg_coordinates = []

        for label, coordinates in points_dict.items():
            if "pos_click" in label:
                pos_coordinates.append(coordinates)
            if "neg_click" in label:
                neg_coordinates.append(coordinates)

        pos_map = create_gaussian_distance_map((img_height, img_width), pos_coordinates)
        neg_map = create_gaussian_distance_map((img_height, img_width), neg_coordinates)

        plt.imsave(pos_save + filename, pos_map)
        plt.imsave(neg_save + filename, neg_map)


if __name__ == '__main__':
    config = read_yaml_file("./config.yaml")

    IMAGE_HEIGHT = config["IMAGE_HEIGHT"]
    IMAGE_WIDTH = config["IMAGE_WIDTH"]
    ANNOTATIONS_XML_PATH = config["ANNOTATIONS_XML_PATH"]
    POS_CLICK_MAPS_PATH = config["POS_CLICK_MAPS_PATH"]
    NEG_CLICK_MAPS_PATH = config["NEG_CLICK_MAPS_PATH"]

    with open(ANNOTATIONS_XML_PATH, 'r') as f:
        data = f.read()
    annotations_data = BeautifulSoup(data, "xml")

    annotations_dict = get_annotations_dict(data_xml=annotations_data)
    create_distance_maps(img_height=IMAGE_HEIGHT,
                         img_width=IMAGE_WIDTH,
                         ann_dict=annotations_dict,
                         pos_save=POS_CLICK_MAPS_PATH,
                         neg_save=NEG_CLICK_MAPS_PATH)
