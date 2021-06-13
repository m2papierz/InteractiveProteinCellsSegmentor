import tensorflow as tf

from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Conv2DTranspose
from tensorflow.keras.layers import concatenate
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.models import Model


def conv2d_block(input_tensor: tf.Tensor, n_filters: int, kernel_size: int, batch_norm: bool) -> tf.Tensor:
    """
    Block of two convolutional layers.

    :param input_tensor: input tensor of the convolutional block
    :param n_filters: number of filters in the convolutional layer
    :param kernel_size: kernel size of the convolutional layer
    :param batch_norm: boolean flag for setting batch normalization
    :return: Output tensor of two convolutional layers
    """
    # First layer
    x = Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size),
               kernel_initializer='he_normal', padding='same')(input_tensor)
    if batch_norm:
        x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # Second layer
    x = Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size),
               kernel_initializer='he_normal', padding='same')(input_tensor)
    if batch_norm:
        x = BatchNormalization()(x)
    x = Activation('relu')(x)

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

    def __init__(self, img_height, img_width, img_channels, dropout=0.05, n_filters=16, kernel_size=3, batch_norm=True):
        # Contracting path
        input_ = Input((img_height, img_width, img_channels))
        c1 = conv2d_block(input_tensor=input_, n_filters=n_filters * 1, kernel_size=kernel_size,
                          batch_norm=batch_norm)
        p1 = MaxPooling2D((2, 2))(c1)
        p1 = Dropout(dropout)(p1)

        c2 = conv2d_block(input_tensor=p1, n_filters=n_filters * 2, kernel_size=kernel_size,
                          batch_norm=batch_norm)
        p2 = MaxPooling2D((2, 2))(c2)
        p2 = Dropout(dropout)(p2)

        c3 = conv2d_block(input_tensor=p2, n_filters=n_filters * 4, kernel_size=kernel_size,
                          batch_norm=batch_norm)
        p3 = MaxPooling2D((2, 2))(c3)
        p3 = Dropout(dropout)(p3)

        c4 = conv2d_block(input_tensor=p3, n_filters=n_filters * 8, kernel_size=kernel_size,
                          batch_norm=batch_norm)
        p4 = MaxPooling2D((2, 2))(c4)
        p4 = Dropout(dropout)(p4)

        c5 = conv2d_block(input_tensor=p4, n_filters=n_filters * 16, kernel_size=kernel_size,
                          batch_norm=batch_norm)

        # Expansive path
        u6 = Conv2DTranspose(n_filters * 8, (3, 3), strides=(2, 2), padding='same', kernel_initializer="he_normal")(c5)
        u6 = concatenate([u6, c4])
        u6 = Dropout(dropout)(u6)
        c6 = conv2d_block(input_tensor=u6, n_filters=n_filters * 8, kernel_size=kernel_size,
                          batch_norm=batch_norm)

        u7 = Conv2DTranspose(n_filters * 4, (3, 3), strides=(2, 2), padding='same', kernel_initializer="he_normal")(c6)
        u7 = concatenate([u7, c3])
        u7 = Dropout(dropout)(u7)
        c7 = conv2d_block(input_tensor=u7, n_filters=n_filters * 4, kernel_size=kernel_size,
                          batch_norm=batch_norm)

        u8 = Conv2DTranspose(n_filters * 2, (3, 3), strides=(2, 2), padding='same', kernel_initializer="he_normal")(c7)
        u8 = concatenate([u8, c2])
        u8 = Dropout(dropout)(u8)
        c8 = conv2d_block(input_tensor=u8, n_filters=n_filters * 2, kernel_size=kernel_size,
                          batch_norm=batch_norm)

        u9 = Conv2DTranspose(n_filters * 1, (3, 3), strides=(2, 2), padding='same', kernel_initializer="he_normal")(c8)
        u9 = concatenate([u9, c1])
        u9 = Dropout(dropout)(u9)
        c9 = conv2d_block(input_tensor=u9, n_filters=n_filters * 1, kernel_size=kernel_size,
                          batch_norm=batch_norm)
        output_ = Conv2D(1, (1, 1), activation='sigmoid')(c9)

        self.model = Model(inputs=[input_], outputs=[output_])

    def compile(self, loss_function="sparse_categorical_crossentropy", optimizer="Adam", metric="accuracy"):
        self.model.compile(loss=loss_function, optimizer=optimizer, metrics=[metric])

        return self.model

    def train(self, dataset, epochs, callbacks):
        epoch_steps = dataset['train'].shape / epochs
        val_steps = dataset['val'].shape / epochs

        self.model.compile()

        return self.model.fit(dataset['train'],
                              steps_per_epoch=epoch_steps,
                              validation_data=dataset['val'],
                              validation_steps=val_steps,
                              epochs=epochs,
                              callbacks=callbacks)
