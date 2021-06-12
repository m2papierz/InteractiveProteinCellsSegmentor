from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Conv2DTranspose
from tensorflow.keras.layers import concatenate
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.models import Model


def conv2d_block(input_tensor, n_filters, kernel_size, batch_norm):
    """
    Block of two convolutional layers.

    :param input_tensor: input tensor of the convolutional block
    :param n_filters: number of filters in the convolutional layer
    :param kernel_size: kernel size of the convolutional layer
    :param batch_norm: boolean flag for setting batch normalization
    :return: Output of two convolutional layers
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
    Class representing the deep neural network U-net architecture for medical image segmentation.

    :param dropout: dropout rate
    :param n_filters: base number of filters in the convolutional layers
    :param kernel_size: kernel size of the convolutional layer
    :param batch_norm: boolean flag for setting batch normalization
    """

    def __init__(self, dropout=0.05, n_filters=16, kernel_size=3, batch_norm=True):
        self.dropout = dropout
        self.n_filters = n_filters
        self.kernel_size = kernel_size
        self.batch_norm = batch_norm

    def build(self, img_height, img_width, img_channels):
        """
        Builds U-net architecture model.

        :param img_height: height the input image
        :param img_width: width of the input image
        :param img_channels: number of channels of the input image
        :return: Unet model
        """
        # Contracting path
        input_ = Input((img_height, img_width, img_channels))
        c1 = conv2d_block(input_tensor=input_, n_filters=self.n_filters * 1, kernel_size=self.kernel_size,
                          batch_norm=self.batch_norm)
        p1 = MaxPooling2D((2, 2))(c1)
        p1 = Dropout(self.dropout)(p1)

        c2 = conv2d_block(input_tensor=p1, n_filters=self.n_filters * 2, kernel_size=self.kernel_size,
                          batch_norm=self.batch_norm)
        p2 = MaxPooling2D((2, 2))(c2)
        p2 = Dropout(self.dropout)(p2)

        c3 = conv2d_block(input_tensor=p2, n_filters=self.n_filters * 4, kernel_size=self.kernel_size,
                          batch_norm=self.batch_norm)
        p3 = MaxPooling2D((2, 2))(c3)
        p3 = Dropout(self.dropout)(p3)

        c4 = conv2d_block(input_tensor=p3, n_filters=self.n_filters * 8, kernel_size=self.kernel_size,
                          batch_norm=self.batch_norm)
        p4 = MaxPooling2D((2, 2))(c4)
        p4 = Dropout(self.dropout)(p4)

        c5 = conv2d_block(input_tensor=p4, n_filters=self.n_filters * 16, kernel_size=self.kernel_size,
                          batch_norm=self.batch_norm)

        # Expansive path
        u6 = Conv2DTranspose(self.n_filters * 8, (3, 3), strides=(2, 2), padding='same')(c5)
        u6 = concatenate([u6, c4])
        u6 = Dropout(self.dropout)(u6)
        c6 = conv2d_block(input_tensor=u6, n_filters=self.n_filters * 8, kernel_size=self.kernel_size,
                          batch_norm=self.batch_norm)

        u7 = Conv2DTranspose(self.n_filters * 4, (3, 3), strides=(2, 2), padding='same')(c6)
        u7 = concatenate([u7, c3])
        u7 = Dropout(self.dropout)(u7)
        c7 = conv2d_block(input_tensor=u7, n_filters=self.n_filters * 4, kernel_size=self.kernel_size,
                          batch_norm=self.batch_norm)

        u8 = Conv2DTranspose(self.n_filters * 2, (3, 3), strides=(2, 2), padding='same')(c7)
        u8 = concatenate([u8, c2])
        u8 = Dropout(self.dropout)(u8)
        c8 = conv2d_block(input_tensor=u8, n_filters=self.n_filters * 2, kernel_size=self.kernel_size,
                          batch_norm=self.batch_norm)

        u9 = Conv2DTranspose(self.n_filters * 1, (3, 3), strides=(2, 2), padding='same')(c8)
        u9 = concatenate([u9, c1])
        u9 = Dropout(self.dropout)(u9)
        c9 = conv2d_block(input_tensor=u9, n_filters=self.n_filters * 1, kernel_size=self.kernel_size,
                          batch_norm=self.batch_norm)
        output_ = Conv2D(1, (1, 1), activation='sigmoid')(c9)

        model = Model(inputs=[input_], outputs=[output_])
        return model
