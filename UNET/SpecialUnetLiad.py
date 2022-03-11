from keras.optimizers import Adam
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, LeakyReLU, Dropout, concatenate, MaxPooling2D, Input
from tensorflow.keras.callbacks import CSVLogger
from tensorflow.keras import Model, models, callbacks
from patchify import unpatchify
from datetime import datetime
import utils
from UNET.iUNet import iUNet
import getpass
import os


class SpecialUNet:

    def __init__(self, input_dim=(128, 128, 6), batch_size=32, epochs=1000):
        inputs = Input(shape=input_dim)

        # encoder
        c1 = Conv2D(16, (3, 3,), activation='relu', padding='same')(inputs)
        c1 = Conv2D(16, (3, 3), activation='relu', padding='same')(c1)
        c2 = MaxPooling2D((2, 2))(c1)

        c2 = Conv2D(32, (3, 3), activation='relu', padding='same')(c2)
        c2 = Conv2D(32, (3, 3), activation='relu', padding='same')(c2)
        c3 = MaxPooling2D((2, 2))(c2)

        c3 = Conv2D(64, (3, 3), activation='relu', padding='same')(c3)
        c3 = Conv2D(64, (3, 3), activation='relu', padding='same')(c3)
        c4 = MaxPooling2D((2, 2))(c3)

        c4 = Conv2D(128, (3, 3), activation='relu', padding='same')(c4)
        c4 = Conv2D(128, (3, 3), activation='relu', padding='same')(c4)
        c5 = MaxPooling2D((2, 2))(c4)

        # embedding
        c5 = Conv2D(256, (3, 3), activation='relu', padding='same')(c5)
        u6 = Conv2DTranspose(256, (3, 3), strides=2, activation='relu', padding='same')(c5)

        # decoder
        u6 = concatenate([u6, c4])
        c6 = Conv2DTranspose(128, (3, 3), padding='same')(u6)
        c6 = LeakyReLU()(c6)
        u7 = Conv2DTranspose(128, (3, 3), strides=2, padding='same')(c6)
        u7 = LeakyReLU()(u7)

        u7 = concatenate([u7, c3])
        c7 = Conv2DTranspose(64, (3, 3), padding='same')(u7)
        c7 = LeakyReLU()(c7)
        u8 = Conv2DTranspose(64, (3, 3), strides=2, padding='same')(c7)
        u8 = LeakyReLU()(u8)

        u8 = concatenate([u8, c2])
        c8 = Conv2DTranspose(32, (3, 3), padding='same')(u8)
        c8 = LeakyReLU()(c8)
        u9 = Conv2DTranspose(32, (3, 3), strides=2, activation='relu', padding='same')(c8)
        u9 = LeakyReLU()(u9)

        u9 = concatenate([u9, c1])
        c9 = Conv2DTranspose(16, (3, 3), padding='same')(u9)
        c9 = LeakyReLU()(c9)
        c9 = Conv2DTranspose(16, (3, 3), padding='same')(c9)
        c9 = LeakyReLU()(c9)
        outputs = Conv2D(input_dim[2], (3, 3), activation='sigmoid', padding='same', name='decoder_output')(c9)

        model = Model(inputs, outputs)
        # optimizer = Adam(0.0002, 0.5)
        model.compile(optimizer='adam', loss='mae')
        self.model = model
        self.input_dim = input_dim
        self.batch_size = batch_size
        self.epochs = epochs

        self.USER = getpass.getuser().split("@")[0]
        print(self.USER)

        self.dir = "/home/%s/%s" % (self.USER, "basicAE")
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
            callbacks.ModelCheckpoint("%sBasicAEModel3D.h5" % model_dir, save_best_only=True),
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
