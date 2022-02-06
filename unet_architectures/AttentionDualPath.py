import tensorflow as tf

from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Conv2DTranspose
from tensorflow.keras.layers import UpSampling2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import ReLU
from tensorflow.keras.layers import concatenate, Add
from tensorflow.keras.models import Model

from utils.configuaration import read_yaml_file
from unet_architectures.attention_module import conv_block_attention_module


def conv2d_block(input_tensor: tf.Tensor, n_filters: int) -> tf.Tensor:
    """
    Block of two convolutional layers.

    :param input_tensor: input tensor of the convolutional block
    :param n_filters: number of filters in the convolutional layer
    :return: output tensor of two convolutional layers
    """
    # First layer
    x0 = BatchNormalization()(input_tensor)
    x0 = ReLU()(x0)
    x0 = Conv2D(filters=n_filters, kernel_size=(3, 3), strides=(1, 1), kernel_initializer='he_normal',
                padding='same')(x0)

    # Second layer
    x0 = BatchNormalization()(x0)
    x0 = ReLU()(x0)
    x0 = Conv2D(filters=n_filters, kernel_size=(3, 3), strides=(1, 1), kernel_initializer='he_normal',
                padding='same')(x0)

    return x0


def con2d_down_block(input_tensor, n_filters) -> tf.Tensor:
    """
    Down-sampling convolutional block.
    :param input_tensor: input tensor of the convolutional block
    :param n_filters: number of filters in the convolutional layer
    :return: result of down-sampling
    """
    # First layer
    x0 = BatchNormalization()(input_tensor)
    x0 = ReLU()(x0)
    x0 = Conv2D(filters=n_filters, kernel_size=(3, 3), strides=(2, 2), kernel_initializer='he_normal',
                padding='same')(x0)

    # Second layer
    x0 = BatchNormalization()(x0)
    x0 = ReLU()(x0)
    x0 = Conv2D(filters=n_filters, kernel_size=(3, 3), strides=(1, 1), kernel_initializer='he_normal',
                padding='same')(x0)

    # Attention mechanism
    x0 = conv_block_attention_module(x0)

    x1 = Conv2D(filters=n_filters, kernel_size=(1, 1), strides=(2, 2), kernel_initializer="he_normal")(input_tensor)

    return Add()([x0, x1])


def con2d_up_block(input_tensor, n_filters):
    """
    Up-sampling convolutional block.

    :param input_tensor: input tensor of the convolutional block
    :param n_filters: number of filters in the convolutional layer
    :return: result of up-sampling
    """
    x0 = Conv2DTranspose(filters=n_filters, kernel_size=(3, 3), strides=(2, 2), kernel_initializer='he_normal',
                         padding='same')(input_tensor)
    x1 = UpSampling2D()(input_tensor)

    return Add()([x0, x1])


class AttentionDualPathUnet:
    def __init__(self, img_height, img_width, img_channels, n_filters=16, attention=True):
        """
        Dual Path Unet based on architecture proposed in: https://arxiv.org/abs/2011.02880

        :param img_height: height of the input image tensor
        :param img_width: width of the input image tensor
        :param img_channels: number of channels of the input image tensor
        :param n_filters: base number of filters in the convolutional layers
        :param attention: flag indicating if to apply attention module
        """
        self.attention = attention

        input_ = Input((img_height, img_width, img_channels))

        # Contracting path
        x00 = conv2d_block(input_tensor=input_, n_filters=1 * n_filters)
        d00 = con2d_down_block(input_tensor=x00, n_filters=1 * n_filters)

        x10 = conv2d_block(input_tensor=d00, n_filters=2 * n_filters)
        d10 = con2d_down_block(input_tensor=x10, n_filters=2 * n_filters)

        x20 = conv2d_block(input_tensor=d10, n_filters=4 * n_filters)
        d20 = con2d_down_block(input_tensor=x20, n_filters=4 * n_filters)

        x30 = conv2d_block(input_tensor=d20, n_filters=8 * n_filters)
        d30 = con2d_down_block(input_tensor=x30, n_filters=8 * n_filters)

        x01 = conv2d_block(input_tensor=x00, n_filters=1 * n_filters)
        d01 = con2d_down_block(input_tensor=x01, n_filters=1 * n_filters)

        x11 = concatenate([d01, x10])
        x11 = conv2d_block(input_tensor=x11, n_filters=2 * n_filters)
        d11 = con2d_down_block(input_tensor=x11, n_filters=2 * n_filters)

        u01 = con2d_up_block(input_tensor=x30, n_filters=n_filters * 8)

        x21 = concatenate([d11, x20, u01])
        x21 = conv2d_block(input_tensor=x21, n_filters=4 * n_filters)
        d21 = con2d_down_block(input_tensor=x21, n_filters=4 * n_filters)

        m = conv2d_block(input_tensor=d30, n_filters=16 * n_filters)
        m = conv_block_attention_module(m)

        # Expanding path
        u00 = con2d_up_block(input_tensor=m, n_filters=n_filters * 16)

        x31 = concatenate([u00, x30, d21])
        x31 = conv2d_block(input_tensor=x31, n_filters=8 * n_filters)

        u10 = con2d_up_block(input_tensor=x31, n_filters=n_filters * 8)

        x22 = concatenate([u10, x21, x20])
        x22 = conv2d_block(input_tensor=x22, n_filters=4 * n_filters)

        u11 = con2d_up_block(input_tensor=x21, n_filters=n_filters * 4)

        x12 = concatenate([u11, x11, x10])
        x12 = conv2d_block(input_tensor=x12, n_filters=2 * n_filters)

        u20 = con2d_up_block(input_tensor=x22, n_filters=n_filters * 4)

        x13 = concatenate([u20, x12, x11, x10])
        x13 = conv2d_block(input_tensor=x13, n_filters=2 * n_filters)

        u21 = con2d_up_block(input_tensor=x12, n_filters=n_filters * 2)

        x02 = concatenate([u21, x01, x00])
        x02 = conv2d_block(input_tensor=x02, n_filters=1 * n_filters)

        u30 = con2d_up_block(input_tensor=x13, n_filters=n_filters * 2)

        x03 = concatenate([u30, x02, x01, x00])
        x03 = conv2d_block(input_tensor=x03, n_filters=1 * n_filters)

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


if __name__ == '__main__':
    config = read_yaml_file("../pipeline/config.yaml")

    image_height = config['image_height']
    image_width = config['image_width']
    input_channels = config['input_channels']

    model = AttentionDualPathUnet(
        img_height=image_height,
        img_width=image_width,
        img_channels=input_channels
    )

    model.model.summary()
