import tensorflow as tf

UNET_PATH = ""
UNETPP_PATH = ""


class CellSegmentator:
    def __init__(self, standard):
        self.standard = standard
        self.segmentation_model = tf.keras.models.load_model(UNET_PATH)

    def pred_cells(self):
        pass

    def postprocessing(self):
        pass
