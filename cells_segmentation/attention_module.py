import tensorflow as tf
import tensorflow.keras.backend as K

from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Add
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.layers import multiply
from tensorflow.keras.layers import Reshape
from tensorflow.keras.layers import Lambda
from tensorflow.keras.layers import Concatenate


def channel_attention(
        input_feature: tf.Tensor,
        ratio: int
) -> tf.Tensor:
    """
    Channel Attention Module.
    """
    channel = input_feature.shape[-1]
    shared_layer_one = Dense(
        channel // ratio, activation='relu', kernel_initializer='he_normal',
        use_bias=True, bias_initializer='zeros')
    shared_layer_two = Dense(
        channel, activation='relu', kernel_initializer='he_normal',
        use_bias=True, bias_initializer='zeros')

    avg_pool = GlobalAveragePooling2D()(input_feature)
    avg_pool = Reshape((1, 1, channel))(avg_pool)
    avg_pool = shared_layer_one(avg_pool)
    avg_pool = shared_layer_two(avg_pool)

    max_pool = GlobalMaxPooling2D()(input_feature)
    max_pool = Reshape((1, 1, channel))(max_pool)
    max_pool = shared_layer_one(max_pool)
    max_pool = shared_layer_two(max_pool)

    channel_feature = Add()([avg_pool, max_pool])
    channel_feature = Activation('sigmoid')(channel_feature)

    return multiply([input_feature, channel_feature])


def spatial_attention(
        input_feature: tf.Tensor
) -> tf.Tensor:
    """
    Spatial Attention Module.
    """
    avg_pool = Lambda(
        lambda x: K.mean(x, axis=3, keepdims=True))(input_feature)
    max_pool = Lambda(
        lambda x: K.max(x, axis=3, keepdims=True))(input_feature)
    concat = Concatenate(axis=3)([avg_pool, max_pool])
    spatial_feature = Conv2D(
        filters=1, kernel_size=7, strides=1, padding='same', activation='sigmoid',
        kernel_initializer='he_normal', use_bias=False)(concat)
    return multiply([input_feature, spatial_feature])


def conv_block_attention_module(
        input_feature: tf.Tensor,
        ratio: int = 16
) -> tf.Tensor:
    """
    Convolutional Block Attention Module proposed in: https://arxiv.org/abs/1807.06521
    """
    refined_feature = channel_attention(input_feature, ratio)
    refined_feature = spatial_attention(refined_feature)
    return refined_feature
