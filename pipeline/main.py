import sys

from PyQt5 import QtWidgets
from utils.configuaration import read_yaml_file
from interactive_cell_segmentator import CellSegmentator, InteractiveCellSegmentator


if __name__ == '__main__':
    config = read_yaml_file("../pipeline/config.yaml")

    PROJECT_PATH = config["PROJECT_PATH"]
    MODELS_PATH = config["MODELS_PATH"]
    UNET_DP_MODEL_PATH = config["UNET_DP_MODEL_PATH"]
    DATA_PATH_TEST = config["DATA_PATH_TEST"]
    TEST_IMAGE_PATH = config['TEST_IMAGE_PATH']
    MODEL_PATH = PROJECT_PATH + MODELS_PATH + UNET_DP_MODEL_PATH

    IMAGE_HEIGHT = config["IMAGE_HEIGHT"]
    IMAGE_WIDTH = config["IMAGE_WIDTH"]
    IMAGE_CHANNELS = config["IMAGE_CHANNELS"]
    POS_CLICK_MAP_SCALE = config["POS_CLICK_MAP_SCALE"]
    NEG_CLICK_MAP_SCALE = config["NEG_CLICK_MAP_SCALE"]

    WINDOW_X = config["WINDOW_X"]
    WINDOW_Y = config["WINDOW_Y"]
    WINDOW_SIZE = config["WINDOW_SIZE"]
    IMG_DPI = config["IMG_DPI"]

    cell_segmentator = CellSegmentator(model_path=MODEL_PATH,
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
