import os
import sys

from PyQt5 import QtWidgets
from utils.configuaration import read_yaml_file
from interactive_cell_segmentator import CellSegmentator, InteractiveCellSegmentator

from unet_architectures.ShallowUnet import ShallowUnet
from unet_architectures.DualPathUnet import DualPathUnet

if __name__ == '__main__':
    config = read_yaml_file("../pipeline/config.yaml")

    project_dir = config["project_dir"]
    models_dir = config["models_dir"]
    test_data_dir = config["test_data_dir"]
    test_image_path = config['test_image_path']

    shallow = config['shallow']
    dual_path = config['dual_path']

    image_height = config["image_height"]
    image_width = config["image_width"]
    image_channels = config["image_channels"]
    pos_click_map_scale = config["pos_click_map_scale"]
    neg_click_map_scale = config["neg_click_map_scale"]

    window_x = config["window_x"]
    window_y = config["window_y"]
    window_size = config["window_size"]
    img_dpi = config["img_dpi"]

    if shallow:
        model_path = os.path.join(models_dir, ShallowUnet.__name__ + '.hdf5')
    elif dual_path:
        model_path = os.path.join(models_dir, DualPathUnet.__name__ + '.hdf5')
    else:
        raise NotImplementedError()

    cell_segmentator = CellSegmentator(
        model_path=model_path,
        pos_clicks=None,
        neg_clicks=None,
        pos_clicks_scale=pos_click_map_scale,
        neg_clicks_scale=neg_click_map_scale,
        img_height=image_height,
        img_width=image_width)

    ics = InteractiveCellSegmentator(
        img_height=image_height,
        img_width=image_width,
        img_dpi=img_dpi,
        img_path=test_image_path,
        segmentation_model=cell_segmentator,
        img_channels=image_channels)

    app = QtWidgets.QApplication(sys.argv)
    ics.setGeometry(window_x, window_y, window_size, window_size)
    app.exec_()
