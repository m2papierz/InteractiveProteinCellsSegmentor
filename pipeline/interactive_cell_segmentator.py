import os
import sys
import matplotlib
import tensorflow as tf
import matplotlib.pyplot as plt

from PyQt5 import QtWidgets, QtGui
from PyQt5.QtWidgets import *
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from utils.configuaration import read_yaml_file
from pipeline.generate_click_distance_maps import create_gaussian_distance_map

matplotlib.use('Qt5Agg')


def load_images(path):
    return [plt.imread(path + img) for img in os.listdir(path)]


def generate_click_maps(pos_clicks, neg_clicks, pos_click_map_scale, neg_click_map_scale):
    pos_map = create_gaussian_distance_map(shape=(IMAGE_HEIGHT, IMAGE_WIDTH),
                                           points=pos_clicks,
                                           scale=pos_click_map_scale)
    
    neg_map = create_gaussian_distance_map(shape=(IMAGE_HEIGHT, IMAGE_WIDTH),
                                           points=neg_clicks,
                                           scale=neg_click_map_scale)
    return pos_map, neg_map


class MplCanvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=5, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
        super(MplCanvas, self).__init__(fig)


class CellSegmentator:
    def __init__(self, model_path, image, pos_clicks, neg_click):
        self.model_path = model_path
        self.image = image
        self.pos_clicks, self.neg_click = generate_click_maps(pos_clicks, neg_click,
                                                              POS_CLICK_MAP_SCALE,
                                                              NEG_CLICK_MAP_SCALE)

    def pred_cells(self):
        pass


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, img_width, img_height, imgs_path, img_dpi, *args, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)

        self.images = load_images(path=imgs_path)
        self.pos_clicks = []
        self.neg_clicks = []
        self.pos_click = True

        # Widgets
        self.sc = MplCanvas(self, width=img_width, height=img_height, dpi=img_dpi)

        self.click_typ_button = QRadioButton("Positive click")
        self.click_typ_button.setChecked(True)

        self.__setup_scene()

        # Interactions
        self.sc.mpl_connect("button_press_event", self.__on_click)
        self.click_typ_button.toggled.connect(lambda: self.__on_click_type_change())

    def start(self):
        self.sc.axes.imshow(self.images[0])
        self.show()

    def __setup_scene(self):
        layout = QVBoxLayout()
        layout.addWidget(self.sc)
        layout.addWidget(self.click_typ_button)
        widget = QWidget()
        widget.setLayout(layout)
        self.setCentralWidget(widget)

        self.sc.axes.get_xaxis().set_visible(False)
        self.sc.axes.get_yaxis().set_visible(False)

        if self.pos_click:
            self.sc.setCursor(QtGui.QCursor(QtGui.QPixmap("utils/pyqt_cursor_imgs/green.png")))
        else:
            self.sc.setCursor(QtGui.QCursor(QtGui.QPixmap("utils/pyqt_cursor_imgs/red.png")))

    def __on_click(self, event):
        if event.xdata is None or event.ydata is None:
            return

        if self.pos_click:
            self.pos_clicks.append([event.xdata, event.ydata])
        else:
            self.neg_clicks.append([event.xdata, event.ydata])
        print(f"x: {event.xdata}, y: {event.ydata}")

    def __on_click_type_change(self):
        if self.click_typ_button.isChecked():
            self.pos_click = True
            self.sc.setCursor(QtGui.QCursor(QtGui.QPixmap("utils/pyqt_cursor_imgs/green.png")))
        else:
            self.pos_click = False
            self.sc.setCursor(QtGui.QCursor(QtGui.QPixmap("utils/pyqt_cursor_imgs/red.png")))


if __name__ == '__main__':
    config = read_yaml_file("pipeline/config.yaml")

    MODELS_PATH = config["MODELS_PATH"]
    UNET_DP_MODEL_PATH = config["UNET_DP_MODEL_PATH"]
    PROJECT_PATH = config["PROJECT_PATH"]
    DATA_PATH_TEST = config["DATA_PATH_TEST"]

    MODEL_PATH = MODELS_PATH + UNET_DP_MODEL_PATH
    IMAGES_PATH = PROJECT_PATH + DATA_PATH_TEST + "image/"

    IMAGE_HEIGHT = config["IMAGE_HEIGHT"]
    IMAGE_WIDTH = config["IMAGE_WIDTH"]
    POS_CLICK_MAP_SCALE = config["POS_CLICK_MAP_SCALE"]
    NEG_CLICK_MAP_SCALE = config["NEG_CLICK_MAP_SCALE"]

    # cell_segmentator = CellSegmentator(model_path=MODEL_PATH)

    app = QtWidgets.QApplication(sys.argv)
    w = MainWindow(img_width=10,
                   img_height=10,
                   img_dpi=300,
                   imgs_path=IMAGES_PATH)
    w.start()
    app.exec_()
