import os
import sys

from PyQt5 import QtWidgets
from utils.configuaration import read_yaml_file
from interactive_cell_segmentator import CellSegmentator, InteractiveCellSegmentator

from unet_architectures.ShallowUnet import ShallowUnet
from unet_architectures.DualPathUnet import DualPathUnet


if __name__ == '__main__':
    config = read_yaml_file("../pipeline/config.yaml")

    PROJECT_PATH = config["PROJECT_PATH"]
    MODELS_PATH = config["MODELS_PATH"]
    DATA_PATH_TEST = config["DATA_PATH_TEST"]
    TEST_IMAGE_PATH = config['TEST_IMAGE_PATH']

    UNET_SHALLOW = config['UNET_SHALLOW']
    UNET_DUAL_PATH = config['UNET_DUAL_PATH']
    ATTENTION = config['ATTENTION']

    IMAGE_HEIGHT = config["IMAGE_HEIGHT"]
    IMAGE_WIDTH = config["IMAGE_WIDTH"]
    IMAGE_CHANNELS = config["IMAGE_CHANNELS"]
    POS_CLICK_MAP_SCALE = config["POS_CLICK_MAP_SCALE"]
    NEG_CLICK_MAP_SCALE = config["NEG_CLICK_MAP_SCALE"]

    WINDOW_X = config["WINDOW_X"]
    WINDOW_Y = config["WINDOW_Y"]
    WINDOW_SIZE = config["WINDOW_SIZE"]
    IMG_DPI = config["IMG_DPI"]

    if UNET_SHALLOW:
        if ATTENTION:
            model_name = ShallowUnet.__name__ + '_attention'
        else:
            model_name = ShallowUnet.__name__
        model_path = os.path.join(MODELS_PATH, model_name + '.hdf5')
    elif UNET_DUAL_PATH:
        if ATTENTION:
            model_name = DualPathUnet.__name__ + '_attention'
        else:
            model_name = DualPathUnet.__name__
        model_path = os.path.join(MODELS_PATH, model_name + '.hdf5')
    else:
        raise NotImplementedError()

    cell_segmentator = CellSegmentator(model_path=model_path,
                                       pos_clicks=None,
                                       neg_clicks=None,
                                       pos_clicks_scale=POS_CLICK_MAP_SCALE,
                                       neg_clicks_scale=NEG_CLICK_MAP_SCALE,
                                       img_height=IMAGE_HEIGHT,
                                       img_width=IMAGE_WIDTH)

    app = QtWidgets.QApplication(sys.argv)
    ics = InteractiveCellSegmentator(img_width=IMAGE_WIDTH,
                                     img_height=IMAGE_HEIGHT,
                                     img_dpi=IMG_DPI,
                                     img_path=TEST_IMAGE_PATH,
                                     segmentation_model=cell_segmentator,
                                     img_channels=IMAGE_CHANNELS)
    ics.setGeometry(WINDOW_X, WINDOW_Y, WINDOW_SIZE, WINDOW_SIZE)
    app.exec_()
