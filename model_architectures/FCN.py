import tensorflow as tf

from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Conv2DTranspose
from tensorflow.keras.layers import Add
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import ReLU
from tensorflow.keras.models import Model

from utils.configuaration import read_yaml_file


def conv2d_block(input_tensor: tf.Tensor, n_filters: int, n_blocks=2) -> tf.Tensor:
    x = BatchNormalization()(input_tensor)
    x = ReLU()(x)
    x = Conv2D(filters=n_filters, kernel_size=(3, 3), strides=(1, 1), kernel_initializer='he_normal',
               padding='same')(x)

    for i in range(n_blocks - 1):
        x = BatchNormalization()(x)
        x = ReLU()(x)
        x = Conv2D(filters=n_filters, kernel_size=(3, 3), strides=(1, 1), kernel_initializer='he_normal',
                   padding='same')(x)

    return MaxPooling2D((2, 2), strides=(2, 2))(x)


class FCN:
    def __init__(self, img_height, img_width, img_channels):
        """
        Fully Convolutional Network proposed in: https://arxiv.org/abs/1411.4038

        :param img_height: height of the input image tensor
        :param img_width: width of the input image tensor
        :param img_channels: number of channels of the input image tensor
        """
        input_ = Input((img_height, img_width, img_channels))

        # Encoder part
        conv1 = conv2d_block(input_tensor=input_, n_filters=64, n_blocks=2)
        conv2 = conv2d_block(input_tensor=conv1, n_filters=128, n_blocks=2)
        conv3 = conv2d_block(input_tensor=conv2, n_filters=256, n_blocks=3)
        conv4 = conv2d_block(input_tensor=conv3, n_filters=512, n_blocks=3)
        conv5 = conv2d_block(input_tensor=conv4, n_filters=512, n_blocks=3)

        # Decoder part
        conv5 = BatchNormalization()(conv5)
        conv5 = ReLU()(conv5)
        output_ = (Conv2D(filters=4096, kernel_size=(7, 7), kernel_initializer='he_normal', padding='same'))(conv5)

        conv7 = BatchNormalization()(output_)
        conv7 = ReLU()(conv7)
        conv7 = (Conv2D(filters=4096, kernel_size=(1, 1), kernel_initializer='he_normal', padding='same'))(conv7)
        conv7_up = Conv2DTranspose(filters=2, kernel_size=(4, 4), strides=(4, 4), use_bias=False)(conv7)

        conv4 = BatchNormalization()(conv4)
        conv4 = ReLU()(conv4)
        conv4_1x1 = (Conv2D(filters=2, kernel_size=(1, 1), kernel_initializer='he_normal', padding='same'))(conv4)
        pool4_up = (Conv2DTranspose(filters=2, kernel_size=(2, 2), strides=(2, 2), use_bias=False))(conv4_1x1)

        conv3 = BatchNormalization()(conv3)
        conv3 = ReLU()(conv3)
        conv3_1x1 = (Conv2D(filters=2, kernel_size=(1, 1), kernel_initializer='he_normal', padding='same'))(conv3)

        output_ = Add()([pool4_up, conv3_1x1, conv7_up])
        output_ = Conv2DTranspose(filters=2, kernel_size=(8, 8), strides=(8, 8), use_bias=False)(output_)
        output_ = Conv2D(1, kernel_size=(1, 1), activation="sigmoid")(output_)

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

    model = FCN(
        img_height=image_height,
        img_width=image_width,
        img_channels=input_channels
    )

    model.model.summary()
