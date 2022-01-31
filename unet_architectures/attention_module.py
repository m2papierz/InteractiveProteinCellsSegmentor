import tensorflow.keras.backend as K

from tensorflow.keras.layers import Dense, Conv2D, Add, Activation, GlobalMaxPooling2D, GlobalAveragePooling2D
from tensorflow.keras.layers import multiply, Reshape, Concatenate, Lambda


def channel_attention(input_feature, ratio):
    """
    Channel Attention Module.

    :param input_feature: input feature maps
    :param ratio: the ration of input feature maps and filters in first dense layer of the module
    :return: channel-refined feature map
    """
    channel = input_feature.shape[-1]
    shared_layer_one = Dense(channel // ratio, activation='relu', kernel_initializer='he_normal',
                             use_bias=True, bias_initializer='zeros')
    shared_layer_two = Dense(channel, activation='relu', kernel_initializer='he_normal',
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


def spatial_attention(input_feature):
    """
    Spatial Attention Module.

    :param input_feature: input feature maps
    :return: spatial-refined feature map
    """
    avg_pool = Lambda(lambda x: K.mean(x, axis=3, keepdims=True))(input_feature)
    max_pool = Lambda(lambda x: K.max(x, axis=3, keepdims=True))(input_feature)
    concat = Concatenate(axis=3)([avg_pool, max_pool])
    spatial_feature = Conv2D(filters=1, kernel_size=7, strides=1, padding='same', activation='sigmoid',
                             kernel_initializer='he_normal', use_bias=False)(concat)
    return multiply([input_feature, spatial_feature])


def conv_block_attention_module(input_feature, ratio=16):
    """
    Convolutional Block Attention Module proposed in: https://arxiv.org/abs/1807.06521

    :param input_feature: input feature maps
    :param ratio: the ration of input feature maps and filters in first dense layer of the channel module
    :return: refined feature map
    """
    refined_feature = channel_attention(input_feature, ratio)
    refined_feature = spatial_attention(refined_feature)
    return refined_feature
