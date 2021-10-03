import tensorflow as tf

from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import UpSampling2D
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.applications import MobileNetV2

filters_num = [32, 64, 128, 256, 512]
skip_connection_names = [
    "input_image",
    "block_1_expand_relu",
    "block_3_expand_relu",
    "block_6_expand_relu",
    "block_13_expand_relu"
]


class UnetMobilenet:
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
