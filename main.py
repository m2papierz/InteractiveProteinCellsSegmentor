import sys
import cgitb
import argparse

from PyQt5 import QtWidgets
from utils.paths_manager import PathsManager
from cells_segmentation.segmentor.cell_segmentor_ui import CellSegmentor
from cells_segmentation.segmentor.cell_segmentor_ui import InteractiveCellSegmentorUI

from cells_segmentation.models.shallow_unet import ShallowUnet
from cells_segmentation.models.fully_convolutional_network import FCN
from cells_segmentation.models.attention_dual_path import AttentionDualPathUnet
from utils.constants import WINDOW_SIZE, WINDOW_X_POS, WINDOW_Y_POS

cgitb.enable(format='text')


if __name__ == '__main__':
    paths_manager = PathsManager()
    config = paths_manager.config
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--model', type=str, default=config.model, choices=['FCN', 'UNET', 'DP-UNET'],
        help='')
    parser.add_argument(
        '--pos_click_map_scale', type=float, default=config.pos_click_map_scale,
        help='')
    parser.add_argument(
        '--neg_click_map_scale', type=float, default=config.neg_click_map_scale,
        help='')
    args = parser.parse_args()

    if args.model == 'UNET':
        model_path = paths_manager.models_dir() / f'{ShallowUnet.__name__}.hdf5'
    elif args.model == 'FCN':
        model_path = paths_manager.models_dir() / f'{FCN.__name__}.hdf5'
    elif args.model == 'DP-UNET':
        model_path = paths_manager.models_dir() / f'{AttentionDualPathUnet.__name__}.hdf5'
    else:
        raise NotImplementedError()

    cell_segmentor = CellSegmentor(
        model_path=model_path,
        pos_clicks=[],
        neg_clicks=[],
        pos_clicks_scale=args.pos_click_map_scale,
        neg_clicks_scale=args.neg_click_map_scale)

    app = QtWidgets.QApplication(sys.argv)
    ics = InteractiveCellSegmentorUI(
        img_path=paths_manager.test_img_path(),
        segmentation_model=cell_segmentor
    )

    ics.setGeometry(WINDOW_X_POS, WINDOW_Y_POS, WINDOW_SIZE, WINDOW_SIZE)
    app.exec_()
