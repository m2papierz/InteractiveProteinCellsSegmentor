import matplotlib
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import cgitb

from PyQt5 import QtWidgets, QtGui, QtCore
from PyQt5.QtWidgets import *
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar

from pipeline.generate_click_guidance_maps import create_guidance_map
from pipeline.loss_and_metrics import JaccardLoss, iou, dice

matplotlib.use('Qt5Agg')
cgitb.enable(format='text')


def load_image(path, img_channels):
    """
    Loads image which will be used for segmentation.

    :param path: path to the image
    :param img_channels: number of channels of the loaded image
    :return: loaded image
    """
    image = tf.io.read_file(path)
    image = tf.image.decode_png(image, channels=img_channels)
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    return image


def cancat_input_tensors(image, pos_map, neg_map, img_height, img_width):
    """
    Converts and concatenates image with guidance maps for an input of the segmentation model.

    :param image: input image
    :param pos_map: input positive clicks map
    :param neg_map: input negative clicks map
    :param img_height: height of an input image
    :param img_width: width of an input image
    :return: input tensor for segmentation model
    """
    image_normalized = image / 255.0

    pos_map = tf.convert_to_tensor(pos_map.reshape((img_height, img_width, 1)))
    pos_map = tf.cast(pos_map, dtype=tf.float32)
    pos_map = pos_map / 255.0

    neg_map = tf.convert_to_tensor(neg_map.reshape((img_height, img_width, 1)))
    neg_map = tf.cast(neg_map, dtype=tf.float32)
    neg_map = neg_map / 255.0

    input_tensor = tf.concat([image_normalized, pos_map, neg_map], axis=2)
    return tf.convert_to_tensor([input_tensor])


class MplCanvas(FigureCanvas):
    def __init__(self, img_width, img_height, dpi):
        self.fig = plt.figure(figsize=(img_width * dpi, img_height * dpi), dpi=dpi)
        self.axes = self.fig.add_subplot(111)
        self.fig.subplots_adjust(0, 0, 1, 1)
        super(MplCanvas, self).__init__(self.fig)


class CellSegmentator:
    def __init__(self, model_path, pos_clicks, neg_clicks, pos_clicks_scale, neg_clicks_scale, img_height, img_width):
        """
        Class using pre-trained model for semi-automatic image segmentation.

        :param model_path: path to pre-trained model
        :param pos_clicks: list of positive clicks coordinates
        :param neg_clicks: list of negative clicks coordinates
        :param pos_clicks_scale: scale factor of positive clicks map
        :param neg_clicks_scale: scale factor of negative clicks map
        :param img_height: height of the image
        :param img_width: width of the image
        """
        self.pos_clicks = pos_clicks
        self.neg_clicks = neg_clicks
        self.pos_clicks_scale = pos_clicks_scale
        self.neg_clicks_scale = neg_clicks_scale
        self.img_height = img_height
        self.img_width = img_width

        custom_objects = {JaccardLoss.__name__: JaccardLoss(),
                          iou.__name__: iou,
                          dice.__name__: dice}

        self.model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)
        self.model.compile(loss=JaccardLoss(), optimizer="Adam", metrics=[iou, dice])

    def __generate_click_maps(self):
        """
        Generates guidance maps from lists of positive and negative clicks coordinates.

        :return: positive and negative clicks maps.
        """
        pos_map = create_guidance_map(shape=(self.img_height, self.img_width),
                                      points=self.pos_clicks,
                                      scale=self.pos_clicks_scale)

        neg_map = create_guidance_map(shape=(self.img_height, self.img_width),
                                      points=self.neg_clicks,
                                      scale=self.neg_clicks_scale)
        return np.array(pos_map), np.array(neg_map)

    def segment(self, image: np.array):
        """
        Cuts target segmentation object from the image based on user interaction.

        :param image: image on which segmentation is conducted
        :return: image of the segmented object
        """
        pos_click_map, neg_click_map = self.__generate_click_maps()
        input_tensor = cancat_input_tensors(
            image=image,
            pos_map=pos_click_map,
            neg_map=neg_click_map,
            img_height=self.img_height,
            img_width=self.img_width)
        prediction = self.model.predict(input_tensor)

        image, pred_mask = np.array(image, dtype=np.float32), \
                           np.array(prediction, dtype=np.float32)
        pred_mask = np.where(pred_mask > 0.5, pred_mask, 0.0)

        return (image * pred_mask)[0]


class InteractiveCellSegmentator(QtWidgets.QMainWindow):
    def __init__(self, img_width: int, img_height: int, img_channels: int, img_path: str,
                 segmentation_model: CellSegmentator, img_dpi: int, *args, **kwargs):
        """
        Interactive cell segmentator GUI.

        :param img_width: height of the image
        :param img_height: weight of the image
        :param img_channels: number of image channels
        :param img_path: path to the image
        :param segmentation_model: CellSegmentator instance
        :param img_dpi: resolution of an image
        """
        super(InteractiveCellSegmentator, self).__init__(*args, **kwargs)
        self.setWindowTitle("Interactive Cell Segmentator")

        self.image = load_image(path=img_path, img_channels=img_channels)
        self.pos_clicks = []
        self.neg_clicks = []
        self.pos_click = True
        self.segmentation_model = segmentation_model
        self.img_channels = img_channels

        # Widgets
        self.sc = MplCanvas(img_width=img_width, img_height=img_height, dpi=img_dpi)
        self.toolbar = NavigationToolbar(self.sc, self)
        self.positive_click_btn = QRadioButton("Positive click")
        self.negative_click_btn = QRadioButton("Negative click")
        self.segment_click_btn = QPushButton("Perform segmentation")
        self.reset_click_btn = QPushButton("Reset application state")
        self.file_dialog_btn = QPushButton("Select image")

        self.__setup_scene()

        # Interactions
        self.sc.mpl_connect("button_press_event", self.__on_image_click)
        self.positive_click_btn.toggled.connect(lambda: self.__on_click_type_change_click())
        self.negative_click_btn.toggled.connect(lambda: self.__on_click_type_change_click())
        self.segment_click_btn.clicked.connect(lambda: self.__on_segment_click())
        self.reset_click_btn.clicked.connect(lambda: self.__on_reset_click())
        self.file_dialog_btn.clicked.connect(lambda: self.__on_file_dialog_click())

    def __setup_scene(self):
        layout = QVBoxLayout()
        layout.addWidget(self.toolbar)
        layout.addWidget(self.sc)
        layout.addWidget(self.positive_click_btn)
        layout.addWidget(self.negative_click_btn)
        layout.addWidget(self.segment_click_btn)
        layout.addWidget(self.reset_click_btn)
        layout.addWidget(self.file_dialog_btn)
        widget = QWidget(flags=QtCore.Qt.WindowCloseButtonHint | QtCore.Qt.WindowMinimizeButtonHint)
        widget.setLayout(layout)
        self.setCentralWidget(widget)

        self.positive_click_btn.setFont(QtGui.QFont('Arial', 12))
        self.negative_click_btn.setFont(QtGui.QFont('Arial', 12))
        self.segment_click_btn.setFont(QtGui.QFont('Arial', 12))
        self.reset_click_btn.setFont(QtGui.QFont('Arial', 12))
        self.file_dialog_btn.setFont(QtGui.QFont('Arial', 12))

        self.positive_click_btn.setChecked(False)
        self.negative_click_btn.setChecked(False)

        self.sc.axes.get_xaxis().set_visible(False)
        self.sc.axes.get_yaxis().set_visible(False)

        self.sc.axes.imshow(self.image)
        self.show()

    def __refresh_image(self):
        self.sc.axes.cla()
        self.sc.axes.imshow(self.image)
        self.sc.fig.canvas.draw_idle()

    def __on_image_click(self, event):
        if not self.positive_click_btn.isChecked() and not self.negative_click_btn.isChecked():
            return

        if event.xdata is None or event.ydata is None:
            return

        if self.pos_click:
            self.pos_clicks.append([int(event.xdata), int(event.ydata)])
            print(f'Positive click coordinates x: {event.xdata} y: {event.ydata}')
        else:
            self.neg_clicks.append([int(event.xdata), int(event.ydata)])
            print(f'Negative click coordinates x: {event.xdata} y: {event.ydata}')

    def __on_click_type_change_click(self):
        if self.positive_click_btn.isChecked():
            self.pos_click = True
            self.sc.setCursor(QtGui.QCursor(QtGui.QPixmap("../utils/pyqt_cursor_imgs/green.png")))
            self.negative_click_btn.setChecked(False)
        else:
            self.pos_click = False
            self.sc.setCursor(QtGui.QCursor(QtGui.QPixmap("../utils/pyqt_cursor_imgs/red.png")))
            self.positive_click_btn.setChecked(False)

    def __on_segment_click(self):
        self.segmentation_model.pos_clicks = self.pos_clicks
        self.segmentation_model.neg_clicks = self.neg_clicks
        segmented_cell = self.segmentation_model.segment(np.array(self.image))

        self.sc.axes.cla()
        self.sc.axes.imshow(segmented_cell)
        self.sc.fig.canvas.draw_idle()

    def __on_reset_click(self):
        self.pos_clicks = []
        self.neg_clicks = []
        self.__refresh_image()

    def __on_file_dialog_click(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        filename, _ = QFileDialog.getOpenFileNames(self, "QFileDialog.getOpenFileName()", "",
                                                   "All Files (*);;Python Files (*.py)", options=options)

        self.image = load_image(path=filename[0], img_channels=self.img_channels)
        self.__on_reset_click()
