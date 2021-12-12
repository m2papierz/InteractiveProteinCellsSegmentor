import os
import sys
import matplotlib
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import cgitb

from PyQt5 import QtWidgets, QtGui
from PyQt5.QtWidgets import *
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar

from utils.configuaration import read_yaml_file
from pipeline.generate_click_distance_maps import create_gaussian_distance_map
from utils.loss_and_metrics import JaccardLoss, iou, dice

matplotlib.use('Qt5Agg')
cgitb.enable(format='text')


def load_images(path):
    imgs = []
    for img in os.listdir(path):
        image = tf.io.read_file(path + img)
        image = tf.image.decode_png(image, channels=3)
        image = tf.image.convert_image_dtype(image, dtype=tf.float32)
        imgs.append(image)
    return imgs


def generate_click_maps(pos_clicks, neg_clicks, pos_click_map_scale, neg_click_map_scale):
    pos_map = create_gaussian_distance_map(shape=(IMAGE_HEIGHT, IMAGE_WIDTH),
                                           points=pos_clicks,
                                           scale=pos_click_map_scale)

    neg_map = create_gaussian_distance_map(shape=(IMAGE_HEIGHT, IMAGE_WIDTH),
                                           points=neg_clicks,
                                           scale=neg_click_map_scale)
    return pos_map, neg_map


def cancat_input_tensors(image, pos_map, neg_map):
    image_normalized = image / 255.0

    pos_map = tf.convert_to_tensor(pos_map.reshape((IMAGE_HEIGHT, IMAGE_WIDTH, 1)))
    pos_map = tf.cast(pos_map, dtype=tf.float32)
    pos_map = pos_map / 255.0

    neg_map = tf.convert_to_tensor(neg_map.reshape((IMAGE_HEIGHT, IMAGE_WIDTH, 1)))
    neg_map = tf.cast(neg_map, dtype=tf.float32)
    neg_map = neg_map / 255.0

    input_tensor = tf.concat([image_normalized, pos_map, neg_map], axis=2)
    return tf.convert_to_tensor([input_tensor])


class MplCanvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=5, dpi=100):
        self.fig = plt.figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111)
        super(MplCanvas, self).__init__(self.fig)


class CellSegmentator:
    def __init__(self, model_path, pos_clicks, neg_clicks):
        self.pos_clicks = pos_clicks
        self.neg_clicks = neg_clicks

        custom_objects = {JaccardLoss.__name__: JaccardLoss(),
                          iou.__name__: iou,
                          dice.__name__: dice}
        self.model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)
        self.model.compile(loss=JaccardLoss(), optimizer="Adam", metrics=[iou, dice])

    def __generate_click_maps(self):
        pos_map, neg_map = generate_click_maps(pos_clicks=self.pos_clicks,
                                               neg_clicks=self.neg_clicks,
                                               pos_click_map_scale=POS_CLICK_MAP_SCALE,
                                               neg_click_map_scale=NEG_CLICK_MAP_SCALE)
        return np.array(pos_map), np.array(neg_map)

    def segment(self, image):
        pos_click_map, neg_click_map = self.__generate_click_maps()
        input_ = cancat_input_tensors(image=image,
                                      pos_map=pos_click_map,
                                      neg_map=neg_click_map)
        pred_mask = self.model.predict(input_)
        cut_image = (np.array(image, dtype=np.float32) * np.array(pred_mask, dtype=np.float32))
        return cut_image[0]


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, img_width, img_height, imgs_path, img_dpi, *args, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)

        self.images = load_images(path=imgs_path)
        self.pos_clicks = []
        self.neg_clicks = []
        self.pos_click = True

        # Widgets
        self.sc = MplCanvas(self, width=img_width, height=img_height, dpi=img_dpi)
        self.toolbar = NavigationToolbar(self.sc, self)
        self.click_typ_button = QRadioButton("Positive click")
        self.click_typ_button.setChecked(True)

        self.pred_click = QPushButton("Segment cell")

        self.__setup_scene()

        # Interactions
        self.sc.mpl_connect("button_press_event", self.__on_click)
        self.click_typ_button.toggled.connect(lambda: self.__on_click_type_change())
        self.pred_click.clicked.connect(lambda: self.__on_click_segment())

        self.sc.axes.imshow(self.images[5])
        self.show()

    def __setup_scene(self):
        layout = QVBoxLayout()
        layout.addWidget(self.toolbar)
        layout.addWidget(self.sc)
        layout.addWidget(self.click_typ_button)
        layout.addWidget(self.pred_click)
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
            self.pos_clicks.append([int(event.xdata), int(event.ydata)])
            print(f'x: {event.xdata} y: {event.ydata}')
        else:
            self.neg_clicks.append([int(event.xdata), int(event.ydata)])
            print(f'x: {event.xdata} y: {event.ydata}')

    def __on_click_type_change(self):
        if self.click_typ_button.isChecked():
            self.pos_click = True
            self.sc.setCursor(QtGui.QCursor(QtGui.QPixmap("utils/pyqt_cursor_imgs/green.png")))
        else:
            self.pos_click = False
            self.sc.setCursor(QtGui.QCursor(QtGui.QPixmap("utils/pyqt_cursor_imgs/red.png")))

    def __on_click_segment(self):
        cell_segmentator.pos_clicks = self.pos_clicks
        cell_segmentator.neg_clicks = self.neg_clicks
        segmented_cell = cell_segmentator.segment(np.array(self.images[5]))

        self.sc.axes.imshow(segmented_cell)
        self.show()


if __name__ == '__main__':
    config = read_yaml_file("pipeline/config.yaml")

    PROJECT_PATH = config["PROJECT_PATH"]
    MODELS_PATH = config["MODELS_PATH"]
    UNET_DP_MODEL_PATH = config["UNET_DP_MODEL_PATH"]
    DATA_PATH_TEST = config["DATA_PATH_TEST"]

    MODEL_PATH = PROJECT_PATH + MODELS_PATH + UNET_DP_MODEL_PATH
    IMAGES_PATH = PROJECT_PATH + DATA_PATH_TEST + "image/"

    IMAGE_HEIGHT = config["IMAGE_HEIGHT"]
    IMAGE_WIDTH = config["IMAGE_WIDTH"]
    POS_CLICK_MAP_SCALE = config["POS_CLICK_MAP_SCALE"]
    NEG_CLICK_MAP_SCALE = config["NEG_CLICK_MAP_SCALE"]

    cell_segmentator = CellSegmentator(model_path=MODEL_PATH, pos_clicks=None, neg_clicks=None)
    app = QtWidgets.QApplication(sys.argv)
    w = MainWindow(img_width=10,
                   img_height=10,
                   img_dpi=300,
                   imgs_path=IMAGES_PATH)
    w.setGeometry(700, 200, 1200, 1200)
    app.exec_()
