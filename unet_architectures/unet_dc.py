import tensorflow as tf

from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, MaxPooling2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.layers import add, concatenate


def conv2d_bn(x, filters, num_row, num_col, activation="relu"):
    """
    2D Convolutional layers

    :param x: input layer
    :param filters: number of filters
    :param num_row: number of rows in filters
    :param num_col: number of columns in filters
    :param activation: activation function of the block
    :returns: output layer
    """

    x = Conv2D(filters, (num_row, num_col), strides=(1, 1), padding="same", use_bias=False)(x)
    x = BatchNormalization(axis=3, scale=False)(x)

    if activation is None:
        return x

    x = Activation(activation)(x)

    return x


def trans_conv2d_bn(x, filters, num_row, num_col):
    """
    2D Transposed Convolutional layers

    :param x: input layer
    :param filters: number of filters
    :param num_row: number of rows in filters
    :param num_col: number of columns in filters
    :returns: output layer
    """

    x = Conv2DTranspose(filters, (num_row, num_col), strides=(2, 2), padding="same")(x)
    x = BatchNormalization(axis=3, scale=False)(x)

    return x


def dual_channel_block(input_layer, n_filters, alpha=1.67):
    """
    Dual channel block.

    :param input_layer: input layer
    :param n_filters: number of filters
    :param alpha:
    :returns: output of the dual channel block
    """

    W = alpha * n_filters

    conv3x3_1 = conv2d_bn(x=input_layer, filters=int(W * 0.167), num_row=3, num_col=3, activation='relu')

    conv5x5_1 = conv2d_bn(x=conv3x3_1, filters=int(W * 0.333), num_row=3, num_col=3, activation='relu')

    conv7x7_1 = conv2d_bn(x=conv5x5_1, filters=int(W * 0.5), num_row=3, num_col=3, activation='relu')

    out1 = concatenate([conv3x3_1, conv5x5_1, conv7x7_1], axis=3)
    out1 = BatchNormalization(axis=3)(out1)

    conv3x3_2 = conv2d_bn(x=input_layer, filters=int(W * 0.167), num_row=3, num_col=3, activation='relu')

    conv5x5_2 = conv2d_bn(x=conv3x3_2, filters=int(W * 0.333), num_row=3, num_col=3, activation='relu')

    conv7x7_2 = conv2d_bn(x=conv5x5_2, filters=int(W * 0.5), num_row=3, num_col=3, activation='relu')
    out2 = concatenate([conv3x3_2, conv5x5_2, conv7x7_2], axis=3)
    out2 = BatchNormalization(axis=3)(out2)

    out = add([out1, out2])
    out = Activation('relu')(out)
    out = BatchNormalization(axis=3)(out)

    return out


def residual_path(input_layer, n_filters, length):
    """
    Residual path.

    :param input_layer: input layer
    :param n_filters: number of filters
    :param length: length of ResPath
    :returns:
    """

    shortcut = conv2d_bn(x=input_layer, filters=n_filters, num_row=1, num_col=1, activation=None)

    out = conv2d_bn(x=input_layer, filters=n_filters, num_row=3, num_col=3, activation='relu')

    out = add([shortcut, out])
    out = Activation('relu')(out)
    out = BatchNormalization(axis=3)(out)

    for i in range(length - 1):
        shortcut = out
        shortcut = conv2d_bn(x=shortcut, filters=n_filters, num_row=1, num_col=1, activation=None)
        out = conv2d_bn(out, filters=n_filters, num_row=3, num_col=3, activation='relu')

        out = add([shortcut, out])
        out = Activation('relu')(out)
        out = BatchNormalization(axis=3)(out)

    return out


class UnetDC:
    def __init__(self, img_height, img_width, img_channels, n_filters=32):
        """
        Dual Channel efficient unet.

        :param img_height: height of the input image tensor
        :param img_width: width of the input image tensor
        :param img_channels: number of channels of the input image tensor
        """
        input_ = Input((img_height, img_width, img_channels))

        dc_block_1 = dual_channel_block(input_layer=input_, n_filters=n_filters)
        pool1 = MaxPooling2D(pool_size=(2, 2))(dc_block_1)
        dc_block_1 = residual_path(input_layer=dc_block_1, n_filters=n_filters, length=4)

        dc_block_2 = dual_channel_block(input_layer=pool1, n_filters=n_filters * 2)
        pool2 = MaxPooling2D(pool_size=(2, 2))(dc_block_2)
        dc_block_2 = residual_path(input_layer=dc_block_2, n_filters=n_filters * 2, length=3)

        dc_block_3 = dual_channel_block(input_layer=pool2, n_filters=n_filters * 4)
        pool3 = MaxPooling2D(pool_size=(2, 2))(dc_block_3)
        dc_block_3 = residual_path(input_layer=dc_block_3, n_filters=n_filters * 4, length=2)

        dc_block_4 = dual_channel_block(input_layer=pool3, n_filters=n_filters * 8)
        pool4 = MaxPooling2D(pool_size=(2, 2))(dc_block_4)
        dc_block_4 = residual_path(input_layer=dc_block_4, n_filters=n_filters * 8, length=1)

        dc_block_5 = dual_channel_block(input_layer=pool4, n_filters=n_filters * 16)

        up6 = concatenate([Conv2DTranspose(filters=n_filters * 8, kernel_size=(2, 2), strides=(2, 2), padding='same')
                           (dc_block_5), dc_block_4], axis=3)
        dc_block_6 = dual_channel_block(input_layer=up6, n_filters=n_filters * 8)

        up7 = concatenate([Conv2DTranspose(filters=n_filters * 4, kernel_size=(2, 2), strides=(2, 2), padding='same')
                           (dc_block_6), dc_block_3], axis=3)
        dc_block_7 = dual_channel_block(input_layer=up7, n_filters=n_filters * 4)

        up8 = concatenate([Conv2DTranspose(filters=n_filters * 2, kernel_size=(2, 2), strides=(2, 2), padding='same')
                           (dc_block_7), dc_block_2], axis=3)
        dc_block_8 = dual_channel_block(input_layer=up8, n_filters=n_filters * 2)

        up9 = concatenate([Conv2DTranspose(filters=n_filters, kernel_size=(2, 2), strides=(2, 2), padding='same')
                           (dc_block_8), dc_block_1], axis=3)
        dc_block_9 = dual_channel_block(input_layer=up9, n_filters=n_filters)

        output_ = conv2d_bn(x=dc_block_9, filters=1, num_row=1, num_col=1, activation='sigmoid')

        self.model = Model(inputs=[input_], outputs=[output_])

    def compile(self, loss_function, optimizer, metrics):
        self.model.compile(loss=loss_function,
                           optimizer=optimizer,
                           metrics=metrics)

    def train(self, dataset, train_size, val_size, batch_size, epochs, callbacks):
        epoch_steps = tf.floor(train_size / batch_size)
        val_steps = tf.floor(val_size / batch_size)

        return self.model.fit(dataset['train'],
                              steps_per_epoch=epoch_steps,
                              validation_data=dataset['val'],
                              validation_steps=val_steps,
                              epochs=epochs,
                              callbacks=callbacks)
