import cgitb
import matplotlib
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from pathlib import Path
from PyQt5.QtWidgets import *
from utils.paths_manager import PathsManager
from cells_segmentation.segmentor.cnn_cell_segmentor import CellSegmentor
from PyQt5 import QtWidgets, QtGui, QtCore
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar

matplotlib.use('Qt5Agg')
cgitb.enable(format='text')


def load_image(
        path: Path
) -> tf.Tensor:
    """
    Loads image which will be used for segmentation.
    """
    image = tf.io.read_file(str(path))
    image = tf.image.decode_png(image, channels=3)
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    return image


class MplCanvas(FigureCanvas):
    def __init__(
            self,
            img_width: int = 512,
            img_height: int = 512,
            dpi: int = 300
    ):
        self.fig = plt.figure(figsize=(img_width * dpi, img_height * dpi), dpi=dpi)
        self.axes = self.fig.add_subplot(111)
        self.fig.subplots_adjust(0, 0, 1, 1)
        super(MplCanvas, self).__init__(self.fig)


class InteractiveCellSegmentorUI(QtWidgets.QMainWindow):
    def __init__(
            self,
            img_path: Path,
            segmentation_model: CellSegmentor,
            *args, **kwargs):
        """
        Interactive cell segmentor GUI.
        """
        super(InteractiveCellSegmentorUI, self).__init__(*args, **kwargs)
        self.setWindowTitle("Interactive Cell Segmentor")

        self.image = load_image(path=img_path)
        self.pos_clicks = []
        self.neg_clicks = []
        self.pos_click = True
        self.segmentation_model = segmentation_model
        self.paths_manager = PathsManager()

        # Widgets
        self.sc = MplCanvas()
        self.toolbar = NavigationToolbar(self.sc, self)
        self.positive_click_btn = QRadioButton("Positive click")
        self.negative_click_btn = QRadioButton("Negative click")
        self.segment_click_btn = QPushButton("Perform segmentation")
        self.reset_click_btn = QPushButton("Reset application state")
        self.file_dialog_btn = QPushButton("Select image")

        self._setup_scene()

        # Interactions
        self.sc.mpl_connect("button_press_event", self._on_image_click)
        self.positive_click_btn.toggled.connect(lambda: self._on_click_type_change_click())
        self.negative_click_btn.toggled.connect(lambda: self._on_click_type_change_click())
        self.segment_click_btn.clicked.connect(lambda: self._on_segment_click())
        self.reset_click_btn.clicked.connect(lambda: self._on_reset_click())
        self.file_dialog_btn.clicked.connect(lambda: self._on_file_dialog_click())

    def _setup_scene(self):
        layout = QVBoxLayout()
        layout.addWidget(self.toolbar)
        layout.addWidget(self.sc)
        layout.addWidget(self.positive_click_btn)
        layout.addWidget(self.negative_click_btn)
        layout.addWidget(self.segment_click_btn)
        layout.addWidget(self.reset_click_btn)
        layout.addWidget(self.file_dialog_btn)
        widget = (QWidget(flags=QtCore.Qt.WindowCloseButtonHint | QtCore.Qt.WindowMinimizeButtonHint))
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

    def _refresh_image(self):
        self.sc.axes.cla()
        self.sc.axes.imshow(self.image)
        self.sc.fig.canvas.draw_idle()

    def _on_image_click(self, event):
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

    def _on_click_type_change_click(self):
        cursors_path = self.paths_manager.cursor_imgs_path()
        if self.positive_click_btn.isChecked():
            c_path = str(cursors_path / "green.png")
            self.pos_click = True
            self.sc.setCursor(QtGui.QCursor(QtGui.QPixmap(c_path)))
            self.negative_click_btn.setChecked(False)
        else:
            c_path = str(cursors_path / "red.png")
            self.pos_click = False
            self.sc.setCursor(QtGui.QCursor(QtGui.QPixmap(c_path)))
            self.positive_click_btn.setChecked(False)

    def _on_segment_click(self):
        self.segmentation_model.pos_clicks = self.pos_clicks
        self.segmentation_model.neg_clicks = self.neg_clicks
        segmented_cell = self.segmentation_model.segment(np.array(self.image))

        self.sc.axes.cla()
        self.sc.axes.imshow(segmented_cell)
        self.sc.fig.canvas.draw_idle()

    def _on_reset_click(self):
        self.pos_clicks = []
        self.neg_clicks = []
        self._refresh_image()

    def _on_file_dialog_click(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        filename, _ = QFileDialog.getOpenFileNames(
            self, "QFileDialog.getOpenFileName()", "",
            "All Files (*);;Python Files (*.py)", options=options
        )

        self.image = load_image(path=filename[0])
        self._on_reset_click()
