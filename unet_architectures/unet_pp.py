import tensorflow as tf

from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Conv2DTranspose
from tensorflow.keras.layers import concatenate
from tensorflow.keras.layers import ELU
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.regularizers import l2


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
