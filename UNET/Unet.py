from tensorflow.keras.layers import Conv2D, Conv2DTranspose, LeakyReLU, Dropout, concatenate, MaxPooling2D, Input, MaxPool2D
from tensorflow.keras.callbacks import CSVLogger
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import Model, models, callbacks
from patchify import unpatchify
from datetime import datetime
import utils
from ICNN import ICNN
import getpass
import os

class Unet(ICNN):

    def __init__(self, input_dim=(128, 128, 6), batch_size=32, epochs=1000):
        inputs = Input(shape=input_dim)

        # encoder: contracting path - downsample
        # 1 - downsample
        f1, p1 = Unet.downsample_block(inputs, 16)
        # 2 - downsample
        f2, p2 = Unet.downsample_block(p1, 32)
        # 3 - downsample
        f3, p3 = Unet.downsample_block(p2, 64)
        # 4 - downsample
        f4, p4 = Unet.downsample_block(p3, 128)
        # 5 - bottleneck
        bottleneck = Unet.double_conv_block(p4, 256)
        # decoder: expanding path - upsample
        # 6 - upsample
        u6 = Unet.upsample_block(bottleneck, f4, 128)
        # 7 - upsample
        u7 = Unet.upsample_block(u6, f3, 64)
        # 8 - upsample
        u8 = Unet.upsample_block(u7, f2, 32)
        # 9 - upsample
        u9 = Unet.upsample_block(u8, f1, 16)
        # outputs
        outputs = Conv2D(input_dim[2], (1, 1), activation='sigmoid', name='decoder_output')(u9)
        # unet model with Keras Functional API
        model = Model(inputs, outputs, name="U-Net")

        opt = Adam()
        model.compile(optimizer=opt, loss='mse')
        self.model = model
        self.input_dim = input_dim
        self.batch_size = batch_size
        self.epochs = epochs

        self.USER = getpass.getuser().split("@")[0]
        print(self.USER)

        self.dir = "/home/%s/%s" % (self.USER, "Unet")
        if not os.path.exists(self.dir):
            os.makedirs(self.dir)

    def train(self, train_x, train_label, val_set=0.0, model_dir="model3D_full"):
        save_time = datetime.now().strftime("%d-%m-%Y_%H-%M")
        model_dir = "%s/%s_%s/" % (self.dir, model_dir, save_time)
        # train_x = utils.transform_dimensions(train_x, [0, 2, 3, 1])
        # train_label = utils.transform_dimensions(train_label, [0, 2, 3, 1])
        # model_dir = self.dir + model_dir
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        model = self.model
        model.summary()
        callback = [
            callbacks.ModelCheckpoint("%sUnetModel3D.h5" % model_dir, save_best_only=True),
            CSVLogger('%slog_%s.csv' % (model_dir, save_time), append=True, separator=';')
        ]
        model.fit(train_x, train_label, batch_size=self.batch_size, epochs=self.epochs, verbose=1,
                  validation_split=val_set,
                  callbacks=callback)
        model.save(model_dir)

    def predict_patches(self, test_data_input):
        """

        @param test_data_input: list of bright_field patches
        @return: list of predicted fluorescent patches
        """
        return self.model.predict(test_data_input)

    def predict(self, img):
        """
        predicts list of full scaled bright_field images
        @param path_to_tiff: list of full scaled predicted fluorescent images
        @return:
        """
        bright_field = utils.utils_patchify(img, self.input_dim)
        for row in bright_field:
            for col in row:
                pred_img = self.model.predict(col)
                col[0] = pred_img[0]
        size = img[0].shape
        return unpatchify(bright_field, size)

    def load_model(self, model_dir='/model/'):
        """
        loads model
        @param model_dir: path in cluster of saved model
        @return: nothing, saves the model in class
        """
        path = self.dir + model_dir
        self.model = models.load_model(path)

    @staticmethod
    def double_conv_block(x, n_filters):
        # Conv2D then ReLU activation
        x = Conv2D(n_filters, 3, padding="same", activation=LeakyReLU())(x)
        # Conv2D then ReLU activation
        x = Conv2D(n_filters, 3, padding="same", activation=LeakyReLU())(x)
        return x

    @staticmethod
    def downsample_block(x, n_filters):
        f = Unet.double_conv_block(x, n_filters)
        p = MaxPool2D(2)(f)
        # p = Dropout(0.3)(p)
        return f, p

    @staticmethod
    def upsample_block(x, conv_features, n_filters):
        # upsample
        x = Conv2DTranspose(n_filters, 3, 2, padding="same")(x)
        # concatenate
        x = concatenate([x, conv_features])
        # dropout
        # x = Dropout(0.3)(x)
        # Conv2D twice with ReLU activation
        x = Unet.double_conv_block(x, n_filters)
        return x
