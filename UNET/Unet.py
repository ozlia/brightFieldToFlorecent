from tensorflow.keras.layers import Conv2D, Conv2DTranspose, LeakyReLU, Dropout, concatenate, MaxPooling2D, Input, \
    MaxPool2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import Model
from ICNN import ICNN
import getpass
import os
from tensorflow.keras.losses import mean_squared_error


class Unet(ICNN):

    def __init__(self, input_dim=(128, 128, 6), batch_size=32, epochs=1000):
        inputs = Input(shape=input_dim)

        self._filter_size = (3, 3)
        self._stride = (2, 2)

        # encoder: contracting path - downsample
        # 1 - downsample
        f1, p1 = self.downsample_block(inputs, 16)
        # 2 - downsample
        f2, p2 = self.downsample_block(p1, 32)
        # 3 - downsample
        f3, p3 = self.downsample_block(p2, 64)
        # 4 - downsample
        f4, p4 = self.downsample_block(p3, 128)
        # 5 - bottleneck
        bottleneck = self.double_conv_block(p4, 256)
        # decoder: expanding path - upsample
        # 6 - upsample
        u6 = self.upsample_block(bottleneck, f4, 128)
        # 7 - upsample
        u7 = self.upsample_block(u6, f3, 64)
        # 8 - upsample
        u8 = self.upsample_block(u7, f2, 32)
        # 9 - upsample
        u9 = self.upsample_block(u8, f1, 16)
        # outputs
        outputs = Conv2D(input_dim[2], (1, 1), activation='sigmoid', name='decoder_output')(u9)
        # unet model with Keras Functional API
        model = Model(inputs, outputs, name="U-Net")

        self.optimizer = Adam()
        self.loss_fn = mean_squared_error
        model.compile(optimizer=self.optimizer, loss=self.loss_fn)
        self.model = model
        self.input_dim = input_dim
        self.batch_size = batch_size
        self.epochs = epochs

        self.USER = getpass.getuser().split("@")[0]
        print(self.USER)

        self.dir = "/home/%s/%s" % (self.USER, "Unet")
        if not os.path.exists(self.dir):
            os.makedirs(self.dir)

    def double_conv_block(self, x, n_filters):
        # Conv2D then ReLU activation
        x = Conv2D(n_filters, self._filter_size, padding="same", activation="relu")(x)
        # Conv2D then ReLU activation
        x = Conv2D(n_filters, self._filter_size, padding="same", activation='relu')(x)
        return x

    def downsample_block(self, x, n_filters):
        f = self.double_conv_block(x, n_filters)
        p = MaxPool2D(self._stride)(f)
        # p = Dropout(0.3)(p)
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
