import os
import sys
import argparse
import numpy as np
from PIL import Image

from tqdm import tqdm
from pathlib import Path
from typing import List, Tuple
from bs4 import BeautifulSoup

try:
    sys.path.insert(1, str(Path(__file__).parent.parent))
except Exception:
    raise EnvironmentError

from utils.paths_manager import PathsManager
from utils.constants import IMG_WIDTH, IMG_HEIGHT


def euclidean_distance(
        p1: Tuple[int, int],
        p2: Tuple[int, int],
        scale: float = 1.0
) -> float:
    """
    Calculates Euclidean distance between two points.
    """
    distance = 0
    for dim in range(len(p1)):
        distance += (p1[dim] - p2[dim]) ** 2
    return np.sqrt(distance) * scale


def create_guidance_map(
        shape: Tuple[int, int],
        points: List[list],
        scale: float = 1.0,
        image: bool = False,
        omega: float = 255.0
) -> np.ndarray:
    """
    Creates guidance map for given points being scaled reverted Euclidean distance map.
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


def get_annotations_dict(
        data_xml: BeautifulSoup
) -> dict:
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
            coordinates = list(map(
                lambda x: int(float(x)), point["points"].split(","))
            )
            image_dict.update({point["label"] + str(i): coordinates[::-1]})
        ann_dict.update({image["name"]: image_dict})

    return ann_dict


def create_and_save_guidance_maps(
        xml_files_path: Path,
        pos_click_map_scale: float,
        neg_click_map_scale: float,
        pos_save: Path,
        neg_save: Path
) -> None:
    """
    Creates and saves guidance maps images of positive and negative clicks for given annotations.
    """
    assert os.listdir(xml_files_path), \
        "No .xml file in the given directory"

    for i, xml_file in enumerate(os.listdir(xml_files_path)):
        with open(xml_files_path / xml_file, 'r') as xml:
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

            pos_map = create_guidance_map(
                shape=(IMG_HEIGHT, IMG_WIDTH),
                points=pos_coordinates,
                image=True,
                scale=pos_click_map_scale
            )

            neg_map = create_guidance_map(
                shape=(IMG_HEIGHT, IMG_WIDTH),
                points=neg_coordinates,
                image=True,
                scale=neg_click_map_scale
            )

            Image.fromarray(pos_map.astype(np.uint8)).save(pos_save / filename, "PNG")
            Image.fromarray(neg_map.astype(np.uint8)).save(neg_save / filename, "PNG")


if __name__ == '__main__':
    paths_manager = PathsManager()
    config = paths_manager.config
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--pos_click_map_scale', type=int, default=config.pos_click_map_scale,
        help='')
    parser.add_argument(
        '--neg_click_map_scale', type=int, default=config.neg_click_map_scale,
        help='')
    args = parser.parse_args()

    print("\n----- GENERATING TRAIN GUIDANCE MAPS -----")
    create_and_save_guidance_maps(
        xml_files_path=paths_manager.train_annotations_dir(),
        pos_click_map_scale=args.pos_click_map_scale,
        neg_click_map_scale=args.neg_click_map_scale,
        pos_save=paths_manager.train_pos_click_maps_dir(),
        neg_save=paths_manager.train_neg_click_maps_dir()
    )

    print("\n----- GENERATING TEST GUIDANCE MAPS -----")
    create_and_save_guidance_maps(
        xml_files_path=paths_manager.test_annotations_dir(),
        pos_click_map_scale=args.pos_click_map_scale,
        neg_click_map_scale=args.neg_click_map_scale,
        pos_save=paths_manager.test_pos_click_maps_dir(),
        neg_save=paths_manager.test_neg_click_maps_dir()
    )
