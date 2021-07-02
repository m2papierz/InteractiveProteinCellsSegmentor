import tensorflow as tf

from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Conv2DTranspose
from tensorflow.keras.layers import concatenate
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import ELU
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import UpSampling2D
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.applications import MobileNetV2

filters_num = [32, 64, 128, 256, 512]
skip_connection_names = [
    "input_image",
    "block_1_expand_relu",
    "block_3_expand_relu",
    "block_6_expand_relu",
    "block_13_expand_relu"
]


def conv2d_block(input_tensor: tf.Tensor, n_filters: int, kernel_size: int, regularizer_factor: float,
                 batch_norm: bool) -> tf.Tensor:
    """
    Block of two convolutional layers.

    :param input_tensor: input tensor of the convolutional block
    :param n_filters: number of filters in the convolutional layer
    :param kernel_size: kernel size of the convolutional layer
    :param regularizer_factor:
    :param batch_norm: boolean flag for setting batch normalization
    :return: Output tensor of two convolutional layers
    """
    # First layer
    x = Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size),
               kernel_initializer='he_normal', padding='same')(input_tensor)
    if batch_norm:
        x = BatchNormalization()(x)
        
    x = Activation('elu')(x)

    # Second layer
    x = Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size),
               kernel_initializer='he_normal', padding='same')(x)
    if batch_norm:
        x = BatchNormalization()(x)
    x = Activation('elu')(x)

    return x


class Unet:
    """
    Standard U-net architecture.

    :param img_height: height of the input image tensor
    :param img_width: width of the input image tensor
    :param img_channels: number of channels of the input image tensor
    :param dropout: dropout rate
    :param n_filters: base number of filters in the convolutional layers
    :param kernel_size: kernel size of the convolutional layer
    :param batch_norm: boolean flag for setting batch normalization
    """

    def __init__(self, img_height, img_width, img_channels, dropout=0.05, n_filters=16, kernel_size=3,
                 regularizer_factor=1e-5, batch_norm=True):
        # Contracting path
        input_ = Input((img_height, img_width, img_channels))
        c1 = conv2d_block(input_tensor=input_, n_filters=n_filters * 1, kernel_size=kernel_size,
                          regularizer_factor=regularizer_factor, batch_norm=batch_norm)
        p1 = MaxPooling2D((2, 2))(c1)
        p1 = Dropout(dropout)(p1)

        c2 = conv2d_block(input_tensor=p1, n_filters=n_filters * 2, kernel_size=kernel_size,
                          regularizer_factor=regularizer_factor, batch_norm=batch_norm)
        p2 = MaxPooling2D((2, 2))(c2)
        p2 = Dropout(dropout)(p2)

        c3 = conv2d_block(input_tensor=p2, n_filters=n_filters * 4, kernel_size=kernel_size,
                          regularizer_factor=regularizer_factor, batch_norm=batch_norm)
        p3 = MaxPooling2D((2, 2))(c3)
        p3 = Dropout(dropout)(p3)

        c4 = conv2d_block(input_tensor=p3, n_filters=n_filters * 8, kernel_size=kernel_size,
                          regularizer_factor=regularizer_factor, batch_norm=batch_norm)
        p4 = MaxPooling2D((2, 2))(c4)
        p4 = Dropout(dropout)(p4)

        c5 = conv2d_block(input_tensor=p4, n_filters=n_filters * 16, kernel_size=kernel_size,
                          regularizer_factor=regularizer_factor, batch_norm=batch_norm)

        # Expansive path
        u6 = Conv2DTranspose(n_filters * 8, (3, 3), strides=(2, 2), padding='same',
                             kernel_initializer="he_normal")(c5)
        u6 = concatenate([u6, c4])
        u6 = Dropout(dropout)(u6)
        c6 = conv2d_block(input_tensor=u6, n_filters=n_filters * 8, kernel_size=kernel_size,
                          regularizer_factor=regularizer_factor, batch_norm=batch_norm)

        u7 = Conv2DTranspose(n_filters * 4, (3, 3), strides=(2, 2), padding='same',
                             kernel_initializer="he_normal")(c6)
        u7 = concatenate([u7, c3])
        u7 = Dropout(dropout)(u7)
        c7 = conv2d_block(input_tensor=u7, n_filters=n_filters * 4, kernel_size=kernel_size,
                          regularizer_factor=regularizer_factor, batch_norm=batch_norm)

        u8 = Conv2DTranspose(n_filters * 2, (3, 3), strides=(2, 2), padding='same',
                             kernel_initializer="he_normal")(c7)
        u8 = concatenate([u8, c2])
        u8 = Dropout(dropout)(u8)
        c8 = conv2d_block(input_tensor=u8, n_filters=n_filters * 2, kernel_size=kernel_size,
                          regularizer_factor=regularizer_factor, batch_norm=batch_norm)

        u9 = Conv2DTranspose(n_filters * 1, (3, 3), strides=(2, 2), padding='same',
                             kernel_initializer="he_normal")(c8)
        u9 = concatenate([u9, c1])
        u9 = Dropout(dropout)(u9)
        c9 = conv2d_block(input_tensor=u9, n_filters=n_filters * 1, kernel_size=kernel_size,
                          regularizer_factor=regularizer_factor, batch_norm=batch_norm)
        output_ = Conv2D(1, (1, 1), activation="sigmoid")(c9)

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


class UnetPP:
    """
    U-net++ architecture.

    :param img_height: height of the input image tensor
    :param img_width: width of the input image tensor
    :param img_channels: number of channels of the input image tensor
    :param dropout: dropout rate
    :param n_filters: base number of filters in the convolutional layers
    """
    def __init__(self, img_height, img_width, img_channels, dropout=0.2, n_filters=16, regularizer_factor=1e-5):
        input_ = Input((img_height, img_width, img_channels))
        x00 = Conv2D(filters=2 * n_filters, kernel_size=(3, 3), padding='same',
                     kernel_regularizer=l2(regularizer_factor))(input_)
        x00 = BatchNormalization()(x00)
        x00 = ELU()(x00)
        x00 = Dropout(dropout)(x00)
        x00 = Conv2D(filters=2 * n_filters, kernel_size=(3, 3), padding='same',
                     kernel_regularizer=l2(regularizer_factor))(x00)
        x00 = BatchNormalization()(x00)
        x00 = ELU()(x00)
        x00 = Dropout(dropout)(x00)
        p0 = MaxPooling2D(pool_size=(2, 2))(x00)

        x10 = Conv2D(filters=4 * n_filters, kernel_size=(3, 3), padding='same',
                     kernel_regularizer=l2(regularizer_factor))(p0)
        x10 = BatchNormalization()(x10)
        x10 = ELU()(x10)
        x10 = Dropout(dropout)(x10)
        x10 = Conv2D(filters=4 * n_filters, kernel_size=(3, 3), padding='same',
                     kernel_regularizer=l2(regularizer_factor))(x10)
        x10 = BatchNormalization()(x10)
        x10 = ELU()(x10)
        x10 = Dropout(dropout)(x10)
        p1 = MaxPooling2D(pool_size=(2, 2))(x10)

        x01 = Conv2DTranspose(filters=2 * n_filters, kernel_size=(2, 2), strides=(2, 2), padding='same')(x10)
        x01 = concatenate([x00, x01])
        x01 = Conv2D(filters=2 * n_filters, kernel_size=(3, 3), padding='same',
                     kernel_regularizer=l2(regularizer_factor))(x01)
        x01 = BatchNormalization()(x01)
        x01 = ELU()(x01)
        x01 = Conv2D(filters=2 * n_filters, kernel_size=(3, 3), padding='same',
                     kernel_regularizer=l2(regularizer_factor))(x01)
        x01 = BatchNormalization()(x01)
        x01 = ELU()(x01)
        x01 = Dropout(dropout)(x01)

        x20 = Conv2D(filters=8 * n_filters, kernel_size=(3, 3), padding='same',
                     kernel_regularizer=l2(regularizer_factor))(p1)
        x20 = BatchNormalization()(x20)
        x20 = ELU()(x20)
        x20 = Dropout(dropout)(x20)
        x20 = Conv2D(filters=8 * n_filters, kernel_size=(3, 3), padding='same',
                     kernel_regularizer=l2(regularizer_factor))(x20)
        x20 = BatchNormalization()(x20)
        x20 = ELU()(x20)
        x20 = Dropout(dropout)(x20)
        p2 = MaxPooling2D(pool_size=(2, 2))(x20)

        x11 = Conv2DTranspose(filters=2 * n_filters, kernel_size=(2, 2), strides=(2, 2), padding='same')(x20)
        x11 = concatenate([x10, x11])
        x11 = Conv2D(filters=2 * n_filters, kernel_size=(3, 3), padding='same',
                     kernel_regularizer=l2(regularizer_factor))(x11)
        x11 = BatchNormalization()(x11)
        x11 = ELU()(x11)
        x11 = Conv2D(filters=2 * n_filters, kernel_size=(3, 3), padding='same',
                     kernel_regularizer=l2(regularizer_factor))(x11)
        x11 = BatchNormalization()(x11)
        x11 = ELU()(x11)
        x11 = Dropout(dropout)(x11)

        x02 = Conv2DTranspose(filters=2 * n_filters, kernel_size=(2, 2), strides=(2, 2), padding='same')(x11)
        x02 = concatenate([x00, x01, x02])
        x02 = Conv2D(filters=2 * n_filters, kernel_size=(3, 3), padding='same',
                     kernel_regularizer=l2(regularizer_factor))(x02)
        x02 = BatchNormalization()(x02)
        x02 = ELU()(x02)
        x02 = Conv2D(filters=2 * n_filters, kernel_size=(3, 3), padding='same',
                     kernel_regularizer=l2(regularizer_factor))(x02)
        x02 = BatchNormalization()(x02)
        x02 = ELU()(x02)
        x02 = Dropout(dropout)(x02)

        x30 = Conv2D(filters=16 * n_filters, kernel_size=(3, 3), padding='same',
                     kernel_regularizer=l2(regularizer_factor))(p2)
        x30 = BatchNormalization()(x30)
        x30 = ELU()(x30)
        x30 = Dropout(0.2)(x30)
        x30 = Conv2D(filters=16 * n_filters, kernel_size=(3, 3), padding='same',
                     kernel_regularizer=l2(regularizer_factor))(x30)
        x30 = BatchNormalization()(x30)
        x30 = ELU()(x30)
        x30 = Dropout(dropout)(x30)
        p3 = MaxPooling2D(pool_size=(2, 2))(x30)

        x21 = Conv2DTranspose(filters=2 * n_filters, kernel_size=(2, 2), strides=(2, 2), padding='same')(x30)
        x21 = concatenate([x20, x21])
        x21 = Conv2D(filters=2 * n_filters, kernel_size=(3, 3), padding='same',
                     kernel_regularizer=l2(regularizer_factor))(x21)
        x21 = BatchNormalization()(x21)
        x21 = ELU()(x21)
        x21 = Conv2D(filters=2 * n_filters, kernel_size=(3, 3), padding='same',
                     kernel_regularizer=l2(regularizer_factor))(x21)
        x21 = BatchNormalization()(x21)
        x21 = ELU()(x21)
        x21 = Dropout(dropout)(x21)

        x12 = Conv2DTranspose(filters=2 * n_filters, kernel_size=(2, 2), strides=(2, 2), padding='same')(x21)
        x12 = concatenate([x10, x11, x12])
        x12 = Conv2D(filters=2 * n_filters, kernel_size=(3, 3), padding='same',
                     kernel_regularizer=l2(regularizer_factor))(x12)
        x12 = BatchNormalization()(x12)
        x12 = ELU()(x12)
        x12 = Conv2D(filters=2 * n_filters, kernel_size=(3, 3), padding='same',
                     kernel_regularizer=l2(regularizer_factor))(x12)
        x12 = BatchNormalization()(x12)
        x12 = ELU()(x12)
        x12 = Dropout(dropout)(x12)

        x03 = Conv2DTranspose(filters=2 * n_filters, kernel_size=(2, 2), strides=(2, 2), padding='same')(x12)
        x03 = concatenate([x00, x01, x02, x03])
        x03 = Conv2D(filters=2 * n_filters, kernel_size=(3, 3), padding='same',
                     kernel_regularizer=l2(regularizer_factor))(x03)
        x03 = BatchNormalization()(x03)
        x03 = ELU()(x03)
        x03 = Conv2D(filters=2 * n_filters, kernel_size=(3, 3), padding='same',
                     kernel_regularizer=l2(regularizer_factor))(x03)
        x03 = BatchNormalization()(x03)
        x03 = ELU()(x03)
        x03 = Dropout(dropout)(x03)

        m = Conv2D(filters=32 * n_filters, kernel_size=(3, 3), padding='same',
                   kernel_regularizer=l2(regularizer_factor))(p3)
        m = BatchNormalization()(m)
        m = ELU()(m)
        m = Conv2D(filters=32 * n_filters, kernel_size=(3, 3), padding='same',
                   kernel_regularizer=l2(regularizer_factor))(m)
        m = BatchNormalization()(m)
        m = ELU()(m)
        m = Dropout(dropout)(m)

        x31 = Conv2DTranspose(filters=16 * n_filters, kernel_size=(2, 2), strides=(2, 2), padding='same')(m)
        x31 = concatenate([x31, x30])
        x31 = Conv2D(filters=16 * n_filters, kernel_size=(3, 3), padding='same',
                     kernel_regularizer=l2(regularizer_factor))(x31)
        x31 = BatchNormalization()(x31)
        x31 = ELU()(x31)
        x31 = Conv2D(filters=16 * n_filters, kernel_size=(3, 3), padding='same',
                     kernel_regularizer=l2(regularizer_factor))(x31)
        x31 = BatchNormalization()(x31)
        x31 = ELU()(x31)
        x31 = Dropout(dropout)(x31)

        x22 = Conv2DTranspose(filters=8 * n_filters, kernel_size=(2, 2), strides=(2, 2), padding='same')(x31)
        x22 = concatenate([x22, x20, x21])
        x22 = Conv2D(filters=8 * n_filters, kernel_size=(3, 3), padding='same',
                     kernel_regularizer=l2(regularizer_factor))(x22)
        x22 = BatchNormalization()(x22)
        x22 = ELU()(x22)
        x22 = Conv2D(filters=8 * n_filters, kernel_size=(3, 3), padding='same',
                     kernel_regularizer=l2(regularizer_factor))(x22)
        x22 = BatchNormalization()(x22)
        x22 = ELU()(x22)
        x22 = Dropout(dropout)(x22)

        x13 = Conv2DTranspose(filters=4 * n_filters, kernel_size=(2, 2), strides=(2, 2), padding='same')(x22)
        x13 = concatenate([x13, x10, x11, x12])
        x13 = Conv2D(filters=4 * n_filters, kernel_size=(3, 3), padding='same',
                     kernel_regularizer=l2(regularizer_factor))(x13)
        x13 = BatchNormalization()(x13)
        x13 = ELU()(x13)
        x13 = Conv2D(filters=4 * n_filters, kernel_size=(3, 3), padding='same')(x13)
        x13 = BatchNormalization()(x13)
        x13 = ELU()(x13)
        x13 = Dropout(dropout)(x13)

        x04 = Conv2DTranspose(filters=2 * n_filters, kernel_size=(2, 2), strides=(2, 2), padding='same')(x13)
        x04 = concatenate([x04, x00, x01, x02, x03], axis=3)
        x04 = Conv2D(filters=2 * n_filters, kernel_size=(3, 3), padding='same',
                     kernel_regularizer=l2(regularizer_factor))(x04)
        x04 = BatchNormalization()(x04)
        x04 = ELU()(x04)
        x04 = Conv2D(filters=2 * n_filters, kernel_size=(3, 3), padding='same',
                     kernel_regularizer=l2(regularizer_factor))(x04)
        x04 = BatchNormalization()(x04)
        x04 = ELU()(x04)
        x04 = Dropout(dropout)(x04)
        output_ = Conv2D(1, kernel_size=(1, 1), kernel_initializer='he_normal', activation='sigmoid')(x04)

        self.model = tf.keras.Model(inputs=[input_], outputs=[output_])

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


class UnetFT:
    def __init__(self, img_height, img_width, img_channels):
        input_ = Input(shape=(img_height, img_width, img_channels), name="input_image")
        encoder = MobileNetV2(input_tensor=input_, weights="imagenet", include_top=False, alpha=1.4)
        encoder_output = encoder.get_layer("block_16_project").output
        encoder.trainable = True

        x = encoder_output
        for i in range(1, len(skip_connection_names) + 1, 1):
            x_skip = encoder.get_layer(skip_connection_names[-i]).output
            x = UpSampling2D((2, 2))(x)
            x = Concatenate()([x, x_skip])

            x = Conv2D(filters_num[-i], (3, 3), padding="same", kernel_initializer="he_normal")(x)
            x = BatchNormalization()(x)
            x = Activation("relu")(x)

            x = Conv2D(filters_num[-i], (3, 3), padding="same", kernel_initializer="he_normal")(x)
            x = BatchNormalization()(x)
            x = Activation("relu")(x)

        output_ = Conv2D(1, (1, 1), activation="sigmoid", padding="same", kernel_initializer="he_normal")(x)

        self.model = tf.keras.Model(inputs=[input_], outputs=[output_])

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
