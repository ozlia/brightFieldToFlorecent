from tensorflow import keras
from keras import layers

from ICNN import ICNN


class AutoEncoder(ICNN):

    def __init__(self, input_dim=(128, 128, 1), batch_size=16, epochs=1000):
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
        outputs = layers.Conv2D(input_dim[2], 3, activation="sigmoid", padding="same")(x)

        model = keras.Model(inputs, outputs)
        model.compile(optimizer="adam", loss="binary_crossentropy")
        self.model = model
        self.input_dim = input_dim
        self.batch_size = batch_size
        self.epochs = epochs

    def train(self, train_x, train_label, valid_x=None, valid_label=None, model_dir="/home/ozlia/basicAE/model2D/"):
        model = self.model
        model.summary()
        callbacks = [
            keras.callbacks.ModelCheckpoint("/home/ozlia/BasicAEModel2D.h5", save_best_only=True)
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

    def predict(self, test_x):
        """
        predicts list of full scaled bright_field images
        @param test_x: list of full scaled predicted fluorescent images
        @return:
        """
        for img in test_x:
            q = 0
        return 0

    def load_model(self, path='/home/ozlia/basicAE/model/'):
        """
        loads model
        @param path: path in cluster of saved model
        @return: nothing, saves the model in class
        """
        self.model = keras.models.load_model(path)
