import os
import sys
import cgitb

from PyQt5 import QtWidgets
from utils.configuaration import read_yaml_file
from InteractiveCellSegmentator import CellSegmentator, InteractiveCellSegmentator

from model_architectures.ShallowUnet import ShallowUnet
from model_architectures.FCN import FCN
from model_architectures.AttentionDualPath import AttentionDualPathUnet

cgitb.enable(format='text')

if __name__ == '__main__':
    config = read_yaml_file("../pipeline/config.yaml")

    models_dir = config["models_dir"]
    test_image_path = config['test_image_path']

    shallow_unet = config["shallow_unet"]
    fcn = config["fcn"]
    attention_dual_path = config["attention_dual_path"]

    image_height = config["image_height"]
    image_width = config["image_width"]
    image_channels = config["image_channels"]
    pos_click_map_scale = config["pos_click_map_scale"]
    neg_click_map_scale = config["neg_click_map_scale"]

    window_x = config["window_x"]
    window_y = config["window_y"]
    window_size = config["window_size"]
    img_dpi = config["img_dpi"]

    if shallow_unet:
        model_path = os.path.join(models_dir, ShallowUnet.__name__ + '.hdf5')
    elif fcn:
        model_path = os.path.join(models_dir, FCN.__name__ + '.hdf5')
    elif attention_dual_path:
        model_path = os.path.join(models_dir, AttentionDualPathUnet.__name__ + '.hdf5')
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

    app = QtWidgets.QApplication(sys.argv)

    ics = InteractiveCellSegmentator(
        img_height=image_height,
        img_width=image_width,
        img_dpi=img_dpi,
        img_path=test_image_path,
        segmentation_model=cell_segmentator,
        img_channels=image_channels)

    ics.setGeometry(window_x, window_y, window_size, window_size)
    app.exec_()
