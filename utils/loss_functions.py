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
def combined_dice_iou_loss(y_true, y_pred, iou_weight=1, dice_weight=1):
    """
    Loss function combining binary crossentropy loss, dice loss and intersection over union loss.

    :param y_true: truth tensor
    :param y_pred: prediction tensor
    :param iou_weight: weight of the intersection over union loss
    :param dice_weight: weight of the dice loss
    :return: Combined loss
    """
    log_dice = -K.log(dice(y_true, y_pred))
    log_iou = - K.log(iou(y_true, y_pred))

    return iou_weight * log_iou + dice_weight * log_dice


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
