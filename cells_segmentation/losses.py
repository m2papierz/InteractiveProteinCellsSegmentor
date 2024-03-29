import tensorflow as tf
import tensorflow.keras.backend as K


@tf.function
def dice(
        y_true: tf.Tensor,
        y_pred: tf.Tensor
) -> float:
    """
    Dice coefficient.
    """
    intersection = K.sum(y_true * y_pred, axis=[1, 2, 3])
    union = K.sum(y_true, axis=[1, 2, 3]) + K.sum(y_pred, axis=[1, 2, 3])
    return K.mean((2. * intersection + 1.0) / (union + 1.0), axis=0)


@tf.function
def iou(
        y_true: tf.Tensor,
        y_pred: tf.Tensor
) -> float:
    """
    Intersection over union index.
    """
    intersection = K.sum(K.abs(y_true * y_pred), axis=[1, 2, 3])
    union = (
            K.sum(y_true, [1, 2, 3]) +
            K.sum(y_pred, [1, 2, 3]) - intersection
    )
    return K.mean((intersection + 1.0) / (union + 1.0), axis=0)


class JaccardLoss(tf.keras.losses.Loss):
    """
    Jaccard distance loss for semantic segmentation.
    """

    def call(self, y_true, y_pred):
        return 1.0 - iou(y_true, y_pred)
