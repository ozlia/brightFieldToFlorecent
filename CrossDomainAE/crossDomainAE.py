from tensorflow import keras
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, LeakyReLU
from tensorflow.keras.callbacks import CSVLogger
from patchify import unpatchify
from datetime import datetime
import utils
from ICNN import ICNN
import getpass
import os

class AutoEncoderCrossDomain(ICNN):

    def __init__(self, input_dim=(128, 128, 6), batch_size=32, epochs=100):
        stride = 2
        inputs = keras.Input(shape=input_dim)

        # Encoder implementation
        x = Conv2D(32, (2, 2), strides=stride, activation="relu", padding="same")(inputs)
        x = Conv2D(64, (2, 2), strides=stride, activation="relu", padding="same")(x)
        x = Conv2D(128, (2, 2), strides=stride, activation="relu", padding="same")(x)

        # decoder implementation
        x = (Conv2DTranspose(128, (2, 2), strides=stride, padding="same"))(x)
        x = LeakyReLU()(x)
        x = (Conv2DTranspose(64, (2, 2), strides=stride, padding="same"))(x)
        x = LeakyReLU()(x)
        x = (Conv2DTranspose(32, (2, 2), strides=stride, padding="same"))(x)
        x = LeakyReLU()(x)
        outputs = Conv2DTranspose(input_dim[2], (2, 2), activation='sigmoid', padding='same', name='decoder_output')(x)

        model = keras.Model(inputs, outputs)
        model.compile(optimizer="adam", loss='mse')
        self.model = model
        self.input_dim = input_dim
        self.batch_size = batch_size
        self.epochs = epochs

        self.USER = getpass.getuser().split("@")[0]
        print(self.USER)

        self.dir = utils.DIRECTORY
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
        callbacks = [
            keras.callbacks.ModelCheckpoint("%sBasicAEModel3D.h5" % model_dir, save_best_only=True),
            CSVLogger('%slog_%s.csv' % (model_dir, save_time), append=True, separator=';')
        ]
        model.fit(train_x, train_label, batch_size=self.batch_size, epochs=self.epochs, verbose=1,
                  validation_split=val_set,
                  callbacks=callbacks)
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
        self.model = keras.models.load_model(path)


