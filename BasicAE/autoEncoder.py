from tensorflow import keras
from keras import layers
from patchify import patchify, unpatchify
import data_prepere
import utils
from ICNN import ICNN
import getpass
import os


class AutoEncoder(ICNN):

    def __init__(self, input_dim=(1, 128, 128), batch_size=32, epochs=1000):
        inputs = keras.Input(shape=input_dim)

        # [First half of the network: downSampling inputs]
        x = layers.Conv2D(32, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same")(inputs)
        x = layers.MaxPooling2D((2, 2), padding="same")(x)
        x = layers.Conv2D(8, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same")(x)
        x = layers.MaxPooling2D((2, 2), padding="same")(x)
        x = layers.Conv2D(8, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same")(x)
        x = layers.MaxPooling2D((2, 2), padding="same")(x)
        x = layers.Conv2D(8, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same")(x)

        # [second half of the net work upSampling]
        x = layers.UpSampling2D(2)(x)
        x = layers.Conv2DTranspose(8, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same")(x)
        x = layers.UpSampling2D((2, 2))(x)
        x = layers.Conv2DTranspose(32, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same")(x)
        x = layers.UpSampling2D((2, 2))(x)
        outputs = layers.Conv2D(input_dim[0], 3, activation="sigmoid", padding="same")(x)

        model = keras.Model(inputs, outputs)
        model.compile(optimizer="adam", loss="binary_crossentropy")
        self.model = model
        self.input_dim = input_dim
        self.batch_size = batch_size
        self.epochs = epochs

        self.USER = getpass.getuser().split("@")[0]
        print(self.USER)

        self.dir = "/home/%s/%s" % (self.USER, "basicAE")
        if not os.path.exists(self.dir):
            os.makedirs(self.dir)

    def train(self, train_x, train_label, valid_x=None, valid_label=None, model_dir="/model2D_full/"):
        model_dir = self.dir + model_dir
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        model = self.model
        model.summary()
        callbacks = [
            keras.callbacks.ModelCheckpoint("%s/BasicAEModel2D.h5" % self.dir, save_best_only=True)
        ]
        validation = (valid_x, valid_label) if valid_x and valid_label else None
        model.fit(train_x, train_label, batch_size=self.batch_size, epochs=self.epochs, verbose=1,
                  validation_data=validation,
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
        # bright_field = data_prepere.pack_unpack_data(path_to_tiff, self.input_dim)
        bright_field = utils.utils_patchify(img, self.input_dim)
        for row in bright_field:
            for col in row:
                pred_img = self.model.predict(col)
                col[0] = pred_img[0]
        size = (640, 1024, 1)
        return unpatchify(bright_field, size)

    def load_model(self, model_dir='/model/'):
        """
        loads model
        @param model_dir: path in cluster of saved model
        @return: nothing, saves the model in class
        """
        path = self.dir + model_dir
        self.model = keras.models.load_model(path)
