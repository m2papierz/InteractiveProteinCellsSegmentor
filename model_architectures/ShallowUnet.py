import tensorflow as tf

from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Conv2DTranspose
from tensorflow.keras.layers import concatenate
from tensorflow.keras.layers import ReLU
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.models import Model

from utils.configuaration import read_yaml_file


def conv2d_block(input_tensor: tf.Tensor, n_filters: int) -> tf.Tensor:
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

    return x


class ShallowUnet:
    """
    The U-net architecture proposed in: https://arxiv.org/abs/1505.04597

    :param img_height: height of the input image tensor
    :param img_width: width of the input image tensor
    :param img_channels: number of channels of the input image tensor
    :param n_filters: base number of filters in the convolutional layers
    """

    def __init__(self, img_height, img_width, img_channels, n_filters=16):
        # Contracting path
        input_ = Input((img_height, img_width, img_channels))
        c1 = conv2d_block(input_tensor=input_, n_filters=n_filters * 1)
        p1 = MaxPooling2D((2, 2))(c1)

        c2 = conv2d_block(input_tensor=p1, n_filters=n_filters * 2)
        p2 = MaxPooling2D((2, 2))(c2)

        c3 = conv2d_block(input_tensor=p2, n_filters=n_filters * 4)
        p3 = MaxPooling2D((2, 2))(c3)

        c4 = conv2d_block(input_tensor=p3, n_filters=n_filters * 8)
        p4 = MaxPooling2D((2, 2))(c4)

        c5 = conv2d_block(input_tensor=p4, n_filters=n_filters * 16)

        # Expansive path
        u6 = Conv2DTranspose(n_filters * 8, kernel_size=(3, 3), strides=(2, 2), padding='same',
                             kernel_initializer="he_normal")(c5)
        u6 = concatenate([u6, c4])
        c6 = conv2d_block(input_tensor=u6, n_filters=n_filters * 8)

        u7 = Conv2DTranspose(n_filters * 4, kernel_size=(3, 3), strides=(2, 2), padding='same',
                             kernel_initializer="he_normal")(c6)
        u7 = concatenate([u7, c3])
        c7 = conv2d_block(input_tensor=u7, n_filters=n_filters * 4)

        u8 = Conv2DTranspose(n_filters * 2, kernel_size=(3, 3), strides=(2, 2), padding='same',
                             kernel_initializer="he_normal")(c7)
        u8 = concatenate([u8, c2])
        c8 = conv2d_block(input_tensor=u8, n_filters=n_filters * 2)

        u9 = Conv2DTranspose(n_filters * 1, kernel_size=(3, 3), strides=(2, 2), padding='same',
                             kernel_initializer="he_normal")(c8)
        u9 = concatenate([u9, c1])
        c9 = conv2d_block(input_tensor=u9, n_filters=n_filters * 1)
        output_ = Conv2D(1, kernel_size=(1, 1), activation="sigmoid")(c9)

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


if __name__ == '__main__':
    config = read_yaml_file("../pipeline/config.yaml")

    image_height = config['image_height']
    image_width = config['image_width']
    input_channels = config['input_channels']

    model = ShallowUnet(
        img_height=image_height,
        img_width=image_width,
        img_channels=input_channels
    )

    model.model.summary()