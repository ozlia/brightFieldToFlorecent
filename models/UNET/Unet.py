from tensorflow.keras.layers import Conv2D, Conv2DTranspose, Dropout, concatenate, Input, \
    MaxPool2D, BatchNormalization, ReLU
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import Model

from helpers import utils
from models.ICNN import ICNN
import getpass
import os


class Unet(ICNN):

    def __init__(self, input_dim=(128, 128, 6), batch_size=32, epochs=1000):
        inputs = Input(shape=input_dim)

        self._filter_size = 3
        self._stride = (2, 2)
        self._num_start_filters = 32

        # encoder: contracting path - downsample
        # 1 - downsample
        f1, p1 = self.downsample_block(inputs, 32)
        # 2 - downsample
        f2, p2 = self.downsample_block(p1, 64)
        # 3 - downsample
        f3, p3 = self.downsample_block(p2, 128)
        # 4 - downsample
        f4, p4 = self.downsample_block(p3, 256)
        # 5 - bottleneck
        bottleneck = self.double_conv_block(p4, 512)
        # decoder: expanding path - upsample
        # 6 - upsample
        u6 = self.upsample_block(bottleneck, f4, 256)
        # 7 - upsample
        u7 = self.upsample_block(u6, f3, 128)
        # 8 - upsample
        u8 = self.upsample_block(u7, f2, 64)
        # 9 - upsample
        u9 = self.upsample_block(u8, f1, 32)
        # outputs
        outputs = Conv2D(input_dim[2], (1, 1), activation='sigmoid', name='decoder_output')(u9)
        # unet model with Keras Functional API
        assert outputs.dtype.name == 'float16', 'mixed_float policy not operational'
        model = Model(inputs, outputs, name="U-Net")

        opt = Adam()
        model.compile(optimizer=opt, loss='mse')
        self.model = model
        self.input_dim = input_dim
        self.batch_size = batch_size
        self.epochs = epochs

        self.USER = getpass.getuser().split("@")[0]
        print(self.USER)

        self.dir = utils.DIRECTORY
        if not os.path.exists(self.dir):
            os.makedirs(self.dir)

    def double_conv_block(self, x, n_filters):
        # Conv2D then ReLU activation
        x = Conv2D(n_filters, self._filter_size, padding="same")(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        # Conv2D then ReLU activation
        x = Conv2D(n_filters, self._filter_size, padding="same")(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        return x

    def downsample_block(self, x, n_filters):
        f = self.double_conv_block(x, n_filters)
        p = MaxPool2D(self._stride)(f)
        p = Dropout(0.3)(p)
        return f, p

    def upsample_block(self, x, conv_features, n_filters):
        # upsample
        x = Conv2DTranspose(n_filters, self._filter_size, self._stride, padding="same")(x)
        # concatenate
        x = concatenate([x, conv_features])
        # dropout
        x = Dropout(0.3)(x)
        # Conv2D twice with ReLU activation
        x = self.double_conv_block(x, n_filters)
        return x
