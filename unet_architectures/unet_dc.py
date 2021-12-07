import tensorflow as tf

from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, MaxPooling2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import add, concatenate
from tensorflow.keras.models import Model


def conv2d_bn(x, filters, num_row, num_col, activation="relu"):
    """
    2D Convolutional layer.

    :param x: input layer
    :param filters: number of filters
    :param num_row: number of rows in filters
    :param num_col: number of columns in filters
    :param activation: activation function of the block
    :returns: output layer
    """

    x = Conv2D(filters, (num_row, num_col), strides=(1, 1), padding="same", kernel_initializer="he_normal")(x)
    x = BatchNormalization(axis=3)(x)

    if activation is None:
        return x
    else:
        return Activation(activation)(x)


def dual_channel_block(input_layer, n_filters, alpha=1.67):
    """
    Dual channel block.

    :param input_layer: input layer
    :param n_filters: number of filters
    :param alpha:
    :returns: output of the dual channel block
    """

    W = alpha * n_filters

    x_1_1 = conv2d_bn(x=input_layer, filters=int(W * 0.167), num_row=3, num_col=3, activation='relu')
    x_1_2 = conv2d_bn(x=x_1_1, filters=int(W * 0.333), num_row=3, num_col=3, activation='relu')
    x_1_3 = conv2d_bn(x=x_1_2, filters=int(W * 0.5), num_row=3, num_col=3, activation='relu')

    out1 = concatenate([x_1_1, x_1_2, x_1_3], axis=3)
    out1 = BatchNormalization(axis=3)(out1)

    x_2_1 = conv2d_bn(x=input_layer, filters=int(W * 0.167), num_row=3, num_col=3, activation='relu')
    x_2_2 = conv2d_bn(x=x_2_1, filters=int(W * 0.333), num_row=3, num_col=3, activation='relu')
    x_2_3 = conv2d_bn(x=x_2_2, filters=int(W * 0.5), num_row=3, num_col=3, activation='relu')

    out2 = concatenate([x_2_1, x_2_2, x_2_3], axis=3)
    out2 = BatchNormalization(axis=3)(out2)

    out = add([out1, out2])
    out = BatchNormalization(axis=3)(out)
    out = Activation('relu')(out)

    return out


def residual_path(input_layer, n_filters, length):
    """
    Residual path.

    :param input_layer: input layer
    :param n_filters: number of filters
    :param length: length of ResPath
    :returns:
    """

    x = conv2d_bn(x=input_layer, filters=n_filters, num_row=1, num_col=1)
    out = conv2d_bn(x=input_layer, filters=n_filters, num_row=3, num_col=3, activation='relu')

    out = add([x, out])
    out = BatchNormalization(axis=3)(out)
    out = Activation('relu')(out)

    for i in range(length - 1):
        x = conv2d_bn(x=out, filters=n_filters, num_row=1, num_col=1)
        out = conv2d_bn(out, filters=n_filters, num_row=3, num_col=3, activation='relu')

        out = add([x, out])
        out = BatchNormalization(axis=3)(out)
        out = Activation('relu')(out)

    return out


class UnetDC:
    def __init__(self, img_height, img_width, img_channels, n_filters=16):
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
