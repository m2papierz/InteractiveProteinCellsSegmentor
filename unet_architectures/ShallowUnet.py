import tensorflow as tf

from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Conv2DTranspose
from tensorflow.keras.layers import concatenate
from tensorflow.keras.layers import ReLU
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.models import Model

from unet_architectures.attention_module import conv_block_attention_module


class ShallowUnet:
    """
    The U-net architecture proposed in: https://arxiv.org/abs/1505.04597

    :param img_height: height of the input image tensor
    :param img_width: width of the input image tensor
    :param img_channels: number of channels of the input image tensor
    :param n_filters: base number of filters in the convolutional layers
    :param attention: flag indicating if to apply attention module
    """

    def __init__(self, img_height, img_width, img_channels, n_filters=16, attention=True):
        self.attention = attention

        # Contracting path
        input_ = Input((img_height, img_width, img_channels))
        c1 = self.conv2d_block(input_tensor=input_, n_filters=n_filters * 1)
        p1 = MaxPooling2D((2, 2))(c1)

        c2 = self.conv2d_block(input_tensor=p1, n_filters=n_filters * 2)
        p2 = MaxPooling2D((2, 2))(c2)

        c3 = self.conv2d_block(input_tensor=p2, n_filters=n_filters * 4)
        p3 = MaxPooling2D((2, 2))(c3)

        c4 = self.conv2d_block(input_tensor=p3, n_filters=n_filters * 8)
        p4 = MaxPooling2D((2, 2))(c4)

        c5 = self.conv2d_block(input_tensor=p4, n_filters=n_filters * 16)

        # Expansive path
        u6 = Conv2DTranspose(n_filters * 8, kernel=(3, 3), strides=(2, 2), padding='same',
                             kernel_initializer="he_normal")(c5)
        u6 = concatenate([u6, c4])
        c6 = self.conv2d_block(input_tensor=u6, n_filters=n_filters * 8)

        u7 = Conv2DTranspose(n_filters * 4, kernel=(3, 3), strides=(2, 2), padding='same',
                             kernel_initializer="he_normal")(c6)
        u7 = concatenate([u7, c3])
        c7 = self.conv2d_block(input_tensor=u7, n_filters=n_filters * 4)

        u8 = Conv2DTranspose(n_filters * 2, kernel=(3, 3), strides=(2, 2), padding='same',
                             kernel_initializer="he_normal")(c7)
        u8 = concatenate([u8, c2])
        c8 = self.conv2d_block(input_tensor=u8, n_filters=n_filters * 2)

        u9 = Conv2DTranspose(n_filters * 1, kernel=(3, 3), strides=(2, 2), padding='same',
                             kernel_initializer="he_normal")(c8)
        u9 = concatenate([u9, c1])
        c9 = self.conv2d_block(input_tensor=u9, n_filters=n_filters * 1)
        output_ = Conv2D(1, kernel=(1, 1), activation="sigmoid")(c9)

        self.model = Model(inputs=[input_], outputs=[output_])

    def conv2d_block(self, input_tensor: tf.Tensor, n_filters: int) -> tf.Tensor:
        """
        Block of two convolutional layers.

        :param input_tensor: input tensor of the convolutional block
        :param n_filters: number of filters in the convolutional layer
        :return: Output tensor of two convolutional layers
        """
        # First layer
        x = Conv2D(filters=n_filters, kernel_size=(3, 3), kernel_initializer='he_normal', padding='same')(
            input_tensor)
        x = BatchNormalization()(x)
        x = ReLU()(x)

        # Second layer
        x = Conv2D(filters=n_filters, kernel_size=(3, 3), kernel_initializer='he_normal', padding='same')(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)

        if self.attention:
            return conv_block_attention_module(x)
        return x

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
