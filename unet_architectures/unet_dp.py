import tensorflow as tf

from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Conv2DTranspose
from tensorflow.keras.layers import UpSampling2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import concatenate, add
from tensorflow.keras.models import Model


def conv2d_block(input_tensor, n_filters, kernel):
    # First layer
    x = Conv2D(filters=n_filters, kernel_size=kernel, kernel_initializer='he_normal', padding='same')(input_tensor)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # Second layer
    x = Conv2D(filters=n_filters, kernel_size=kernel, kernel_initializer='he_normal',  padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    return x


def con2d_down_block(input_tensor, n_filters, kernel):
    # First layer
    x1 = Conv2D(filters=n_filters, kernel_size=kernel, strides=(2, 2), kernel_initializer='he_normal',
                padding='same')(input_tensor)
    x1 = BatchNormalization()(x1)
    x1 = Activation('relu')(x1)

    # Second layer
    x1 = Conv2D(filters=n_filters, kernel_size=kernel, kernel_initializer='he_normal', padding='same')(x1)
    x1 = BatchNormalization()(x1)
    x1 = Activation('relu')(x1)

    x0 = Conv2D(filters=n_filters, kernel_size=(1, 1), strides=(2, 2), kernel_initializer="he_normal")(input_tensor)
    x0 = BatchNormalization()(x0)
    x0 = Activation('relu')(x0)

    x = add([x0, x1])
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    return x


def con2d_up_block(input_tensor, n_filters):
    x0 = Conv2DTranspose(filters=n_filters, kernel_size=(3, 3), strides=(2, 2), kernel_initializer='he_normal',
                         padding='same')(input_tensor)
    x1 = UpSampling2D()(input_tensor)
    x = add([x0, x1])
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    return x


class UnetDP:
    def __init__(self, img_height, img_width, img_channels, n_filters=16):
        """
        Dual Path Unet proposed in: https://arxiv.org/abs/2011.02880

        :param img_height: height of the input image tensor
        :param img_width: width of the input image tensor
        :param img_channels: number of channels of the input image tensor=
        :param n_filters: base number of filters in the convolutional layers
        """
        input_ = Input((img_height, img_width, img_channels))

        x00 = conv2d_block(input_tensor=input_, n_filters=1 * n_filters, kernel=(3, 3))
        d00 = con2d_down_block(input_tensor=x00, n_filters=1 * n_filters, kernel=(3, 3))

        x10 = conv2d_block(input_tensor=d00, n_filters=2 * n_filters, kernel=(3, 3))
        d10 = con2d_down_block(input_tensor=x10, n_filters=2 * n_filters, kernel=(3, 3))

        x20 = conv2d_block(input_tensor=d10, n_filters=4 * n_filters, kernel=(3, 3))
        d20 = con2d_down_block(input_tensor=x20, n_filters=4 * n_filters, kernel=(3, 3))

        x30 = conv2d_block(input_tensor=d20, n_filters=8 * n_filters, kernel=(3, 3))
        d30 = con2d_down_block(input_tensor=x30, n_filters=8 * n_filters, kernel=(3, 3))

        x01 = conv2d_block(input_tensor=x00, n_filters=1 * n_filters, kernel=(3, 3))
        d01 = con2d_down_block(input_tensor=x01, n_filters=1 * n_filters, kernel=(3, 3))

        m = conv2d_block(input_tensor=d30, n_filters=16 * n_filters, kernel=(3, 3))

        x11 = concatenate([d01, x10])
        x11 = conv2d_block(input_tensor=x11, n_filters=2 * n_filters, kernel=(3, 3))
        d11 = con2d_down_block(input_tensor=x11, n_filters=2 * n_filters, kernel=(3, 3))

        u01 = con2d_up_block(input_tensor=x30, n_filters=n_filters * 8)

        x21 = concatenate([d11, x20, u01])
        x21 = conv2d_block(input_tensor=x21, n_filters=4 * n_filters, kernel=(3, 3))
        d21 = con2d_down_block(input_tensor=x21, n_filters=4 * n_filters, kernel=(3, 3))

        u00 = con2d_up_block(input_tensor=m, n_filters=n_filters * 16)

        x31 = concatenate([u00, x30, d21])
        x31 = conv2d_block(input_tensor=x31, n_filters=8 * n_filters, kernel=(3, 3))

        u10 = con2d_up_block(input_tensor=x31, n_filters=n_filters * 8)

        x22 = concatenate([u10, x21, x20])
        x22 = conv2d_block(input_tensor=x22, n_filters=4 * n_filters, kernel=(3, 3))

        u11 = con2d_up_block(input_tensor=x21, n_filters=n_filters * 4)

        x12 = concatenate([u11, x11, x10])
        x12 = conv2d_block(input_tensor=x12, n_filters=2 * n_filters, kernel=(3, 3))

        u20 = con2d_up_block(input_tensor=x22, n_filters=n_filters * 4)

        x13 = concatenate([u20, x12, x11, x10])
        x13 = conv2d_block(input_tensor=x13, n_filters=2 * n_filters, kernel=(3, 3))

        u21 = con2d_up_block(input_tensor=x12, n_filters=n_filters * 2)

        x02 = concatenate([u21, x01, x00])
        x02 = conv2d_block(input_tensor=x02, n_filters=1 * n_filters, kernel=(3, 3))

        u30 = con2d_up_block(input_tensor=x13, n_filters=n_filters * 2)

        x03 = concatenate([u30, x02, x01, x00])
        x03 = conv2d_block(input_tensor=x03, n_filters=1 * n_filters, kernel=(3, 3))

        output_ = Conv2D(1, (1, 1), activation="sigmoid")(x03)

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
