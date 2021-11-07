import tensorflow as tf

from utils.configuaration import read_yaml_file


# ---> TBD
class CellSegmentator:
    def __init__(self, model_path):
        self.model = tf.keras.models.load_model(model_path)

    def pred_cells(self):
        pass


if __name__ == '__main__':
    config = read_yaml_file("./config.yaml")

    MODEL_PATH = config["PATH"]

    cell_segmentator = CellSegmentator(model_path=MODEL_PATH)
