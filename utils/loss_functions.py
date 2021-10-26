import tensorflow as tf
import tensorflow.keras.backend as K


@tf.function
def iou(y_true, y_pred, smooth=1) -> float:
    """
    Intersection over union loss function.

    :param y_true: truth tensor
    :param y_pred: prediction tensor
    :param smooth: parameter for numerical stability to avoid divide by zero errors
    :return: Intersection over union loss
    """
    intersection = K.sum(K.abs(y_true * y_pred), axis=[1, 2, 3])
    union = K.sum(y_true, [1, 2, 3]) + K.sum(y_pred, [1, 2, 3]) - intersection
    return K.mean((intersection + smooth) / (union + smooth), axis=0)


@tf.function
def dice(y_true, y_pred, smooth=1) -> float:
    """
    Dice loss function.

    :param y_true: truth tensor
    :param y_pred: prediction tensor
    :param smooth: parameter for numerical stability to avoid divide by zero errors
    :return: Dice loss
    """
    intersection = K.sum(y_true * y_pred, axis=[1, 2, 3])
    union = K.sum(y_true, axis=[1, 2, 3]) + K.sum(y_pred, axis=[1, 2, 3])
    return K.mean((2. * intersection + smooth) / (union + smooth), axis=0)


@tf.function
def jaccard_distance_loss(y_true, y_pred, smooth=100):
    """
    Jaccard distance for semantic segmentation also known as the intersection-over-union loss.
    This implementation is adapted for semantic segmentation.

    :param y_true: truth tensor
    :param y_pred: prediction tensor
    :param smooth: parameter for numerical stability to avoid divide by zero error
    :return: Jaccard loss
    """
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    sum_ = K.sum(K.abs(y_true) + K.abs(y_pred), axis=-1)
    jac = (intersection + smooth) / (sum_ - intersection + smooth)
    return (1 - jac) * smooth
