import numpy as np
import tensorflow as tf

from typing import Tuple
from pathlib import Path
from cnns_pipeline.generate_data import create_guidance_map
from cells_segmentation.losses import JaccardLoss, iou, dice
from utils.constants import IMG_WIDTH, IMG_HEIGHT


class CellSegmentor:
    custom_objects = {
        JaccardLoss.__name__: JaccardLoss(),
        iou.__name__: iou,
        dice.__name__: dice
    }

    def __init__(
            self,
            model_path: Path,
            pos_clicks: list,
            neg_clicks: list,
            pos_clicks_scale: float,
            neg_clicks_scale: float
    ):
        """
        Class using pre-trained model for semi-automatic image segmentation.

        :param model_path: path to pre-trained model
        :param pos_clicks: list of positive clicks coordinates
        :param neg_clicks: list of negative clicks coordinates
        :param pos_clicks_scale: scale factor of positive clicks map
        :param neg_clicks_scale: scale factor of negative clicks map

        """
        self.pos_clicks = pos_clicks
        self.neg_clicks = neg_clicks
        self.pos_clicks_scale = pos_clicks_scale
        self.neg_clicks_scale = neg_clicks_scale

        self.model = self._load_model(model_path)

    def _load_model(
            self,
            model_path: Path
    ) -> tf.keras.models.Model:
        model = tf.keras.models.load_model(
            model_path, custom_objects=self.custom_objects)
        model.compile(
            loss=JaccardLoss(), optimizer="Adam", metrics=[iou, dice])
        return model

    def _generate_click_maps(
            self
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generates guidance maps from lists of positive and negative clicks coordinates.
        """
        pos_map = create_guidance_map(
            shape=(IMG_HEIGHT, IMG_WIDTH),
            points=self.pos_clicks,
            scale=self.pos_clicks_scale
        )

        neg_map = create_guidance_map(
            shape=(IMG_HEIGHT, IMG_WIDTH),
            points=self.neg_clicks,
            scale=self.neg_clicks_scale
        )

        return pos_map.astype(np.float32), neg_map.astype(np.float32)

    def _concat_input_tensors(
            self,
            image: np.ndarray,
    ) -> tf.Tensor:
        """
        Converts and concatenates image with guidance maps for an input of the segmentation model.
        """
        pos_click_map, neg_click_map = self._generate_click_maps()

        image_normalized = image / 255.0

        pos_click_map = tf.convert_to_tensor(
            pos_click_map.reshape((IMG_HEIGHT, IMG_WIDTH, 1)))
        pos_click_map = tf.cast(pos_click_map, dtype=tf.float32)
        pos_click_map = pos_click_map / 255.0

        neg_click_map = tf.convert_to_tensor(
            neg_click_map.reshape((IMG_HEIGHT, IMG_WIDTH, 1)))
        neg_click_map = tf.cast(neg_click_map, dtype=tf.float32)
        neg_click_map = neg_click_map / 255.0

        input_tensor = tf.concat(
            [image_normalized, pos_click_map, neg_click_map], axis=2)
        return tf.convert_to_tensor([input_tensor])

    def segment(
            self,
            image: np.ndarray
    ) -> np.ndarray:
        """
        Cuts target segmentation object from the image based on user interaction.
        """
        input_tensor = self._concat_input_tensors(image)
        prediction = self.model.predict(input_tensor)

        image, pred_mask = np.array(image, dtype=np.float32), \
            np.array(prediction, dtype=np.float32)
        pred_mask = np.where(pred_mask > 0.5, pred_mask, 0.0)

        return (image * pred_mask)[0]
