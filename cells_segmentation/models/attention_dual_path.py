import tensorflow as tf

from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Conv2DTranspose
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import ReLU
from tensorflow.keras.layers import concatenate, Add
from tensorflow.keras.models import Model

from cells_segmentation.attention_module import conv_block_attention_module


def conv2d_block(
        input_tensor: tf.Tensor,
        n_filters: int
) -> tf.Tensor:
    """
    Block of two convolutional layers with bath normalization and ReLU activation.
    """
    x0 = BatchNormalization()(input_tensor)
    x0 = ReLU()(x0)
    x0 = Conv2D(
        filters=n_filters, kernel_size=(3, 3), strides=(1, 1), kernel_initializer='he_normal',
        padding='same')(x0)

    x0 = BatchNormalization()(x0)
    x0 = ReLU()(x0)
    x0 = Conv2D(
        filters=n_filters, kernel_size=(3, 3), strides=(1, 1), kernel_initializer='he_normal',
        padding='same')(x0)

    return x0


def con2d_down_block(
        input_tensor: tf.Tensor,
        n_filters: int
) -> tf.Tensor:
    """
    Down-sampling convolutional block.
    """
    # First convolution layer with 2x2 kernel
    x0 = BatchNormalization()(input_tensor)
    x0 = ReLU()(x0)
    x0 = Conv2D(
        filters=n_filters, kernel_size=(3, 3), strides=(2, 2), kernel_initializer='he_normal',
        padding='same')(x0)

    x0 = BatchNormalization()(x0)
    x0 = ReLU()(x0)
    x0 = Conv2D(
        filters=n_filters, kernel_size=(3, 3), strides=(1, 1), kernel_initializer='he_normal',
        padding='same')(x0)

    x1 = Conv2D(
        filters=n_filters, kernel_size=(1, 1), strides=(2, 2), kernel_initializer="he_normal")(input_tensor)

    return Add()([x0, x1])


def con2d_up_block(
        input_tensor: tf.Tensor,
        n_filters: int,
        attention: bool = False
) -> tf.Tensor:
    """
    Up-sampling convolutional block.
    """
    x0 = BatchNormalization()(input_tensor)
    x0 = ReLU()(x0)
    x0 = Conv2DTranspose(
        filters=n_filters, kernel_size=(3, 3), strides=(2, 2), kernel_initializer='he_normal',
        padding='same')(x0)

    x0 = BatchNormalization()(x0)
    x0 = ReLU()(x0)
    x0 = Conv2D(
        filters=n_filters, kernel_size=(3, 3), strides=(1, 1), kernel_initializer='he_normal',
        padding='same')(x0)

    x1 = Conv2DTranspose(
        filters=n_filters, kernel_size=(1, 1), strides=(2, 2), kernel_initializer="he_normal",
        padding="same")(input_tensor)
    x1 = conv_block_attention_module(x1) if attention else x1

    return Add()([x0, x1])


class AttentionDualPathUnet:
    def __init__(
            self,
            img_height: int = 512,
            img_width: int = 512,
            input_channels: int = 5,
            n_filters=16
    ):
        """
        Dual Path Unet based on architecture proposed in: https://arxiv.org/abs/2011.02880
        """
        input_ = Input((img_height, img_width, input_channels))

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

        u30 = con2d_up_block(input_tensor=x30, n_filters=n_filters * 8)

        x21 = concatenate([d11, x20, u30])
        x21 = conv2d_block(input_tensor=x21, n_filters=4 * n_filters)
        d21 = con2d_down_block(input_tensor=x21, n_filters=8 * n_filters)

        # Bottom block
        m = conv2d_block(input_tensor=d30, n_filters=16 * n_filters)

        # Extracting path
        um = con2d_up_block(input_tensor=m, n_filters=n_filters * 16, attention=True)

        x31 = concatenate([um, x30, d21])
        x31 = conv2d_block(input_tensor=x31, n_filters=8 * n_filters)

        u31 = con2d_up_block(input_tensor=x31, n_filters=n_filters * 8, attention=True)

        x22 = concatenate([u31, x21, x20])
        x22 = conv2d_block(input_tensor=x22, n_filters=4 * n_filters)

        u21 = con2d_up_block(input_tensor=x21, n_filters=n_filters * 4)

        x12 = concatenate([u21, x11, x10])
        x12 = conv2d_block(input_tensor=x12, n_filters=2 * n_filters)

        u22 = con2d_up_block(input_tensor=x22, n_filters=n_filters * 4, attention=True)

        x13 = concatenate([u22, x12, x11, x10])
        x13 = conv2d_block(input_tensor=x13, n_filters=2 * n_filters)

        u12 = con2d_up_block(input_tensor=x12, n_filters=n_filters * 2)

        x02 = concatenate([u12, x01, x00])
        x02 = conv2d_block(input_tensor=x02, n_filters=1 * n_filters)

        u13 = con2d_up_block(input_tensor=x13, n_filters=n_filters * 2, attention=True)

        x03 = concatenate([u13, x02, x01, x00])
        x03 = conv2d_block(input_tensor=x03, n_filters=1 * n_filters)

        output_ = Conv2D(1, (1, 1), activation="sigmoid")(x03)

        self.model = Model(inputs=[input_], outputs=[output_])

    def compile(self, loss_function, optimizer, metrics):
        self.model.compile(
            loss=loss_function,
            optimizer=optimizer,
            metrics=metrics
        )

    def train(self, dataset, train_size, val_size, batch_size, epochs, callbacks):
        epoch_steps = tf.floor(train_size / batch_size)
        val_steps = tf.floor(val_size / batch_size)

        return self.model.fit(
            dataset['train'],
            steps_per_epoch=epoch_steps,
            validation_data=dataset['val'],
            validation_steps=val_steps,
            epochs=epochs,
            callbacks=callbacks
        )


if __name__ == '__main__':
    model = AttentionDualPathUnet()
    model.model.summary()
