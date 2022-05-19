from tensorflow import keras
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, LeakyReLU, BatchNormalization, Dropout
from tensorflow.keras.callbacks import CSVLogger
from patchify import unpatchify
from datetime import datetime
import utils
from ICNN import ICNN
import getpass
import os
import numpy as np


class AutoEncoderCrossDomain(ICNN):

    def __init__(self, input_dim=(128, 128, 6), batch_size=32, epochs=100):
        stride = 2
        filter_size = 2
        inputs = keras.Input(shape=input_dim)

        # Encoder implementation
        en_inputs = inputs
        x = Conv2D(32, filter_size, strides=stride, activation=LeakyReLU(), padding="same")(en_inputs)
        x = BatchNormalization()(x)
        x = Conv2D(64, filter_size, strides=stride, activation=LeakyReLU(), padding="same")(x)
        x = BatchNormalization()(x)
        x = Conv2D(128, filter_size, strides=stride, activation=LeakyReLU(), padding="same")(x)
        x = BatchNormalization()(x)
        x = Conv2D(256, filter_size, strides=stride, activation="relu", padding="same")(x)
        x = BatchNormalization()(x)
        x = Conv2D(512, filter_size, strides=stride, activation="relu", padding="same")(x)
        x = BatchNormalization()(x)
        latent_layer = Conv2D(512, filter_size, strides=stride, activation=LeakyReLU(), padding="same")(x)
        encoder = keras.Model(inputs, latent_layer, name="Encoder")

        # decoder implementation
        latent_layer_shape = latent_layer.get_shape()
        decoder_inputs = keras.Input(shape=(latent_layer_shape[1], latent_layer_shape[2], latent_layer_shape[3]))
        x = (Conv2DTranspose(512, filter_size, strides=stride, padding="same", activation=LeakyReLU()))(decoder_inputs)
        x = BatchNormalization()(x)
        x = (Conv2DTranspose(512, filter_size, strides=stride, padding="same", activation="relu"))(x)
        x = BatchNormalization()(x)
        x = (Conv2DTranspose(256, filter_size, strides=stride, padding="same", activation="relu"))(x)
        x = BatchNormalization()(x)
        x = (Conv2DTranspose(128, filter_size, strides=stride, padding="same", activation=LeakyReLU()))(x)
        x = BatchNormalization()(x)
        x = (Conv2DTranspose(64, filter_size, strides=stride, padding="same", activation=LeakyReLU()))(x)
        x = BatchNormalization()(x)
        x = (Conv2DTranspose(32, filter_size, strides=stride, padding="same", activation=LeakyReLU()))(x)
        x = BatchNormalization()(x)
        outputs = Conv2DTranspose(input_dim[2], filter_size, activation='sigmoid', padding='same',name='decoder_output')(x)
        decoder = keras.Model(decoder_inputs, outputs, name="Decoder")

        # AutoEncoder
        model = keras.Model(inputs, decoder(encoder(inputs)), name='AutoEncoder')
        # model = keras.Model(inputs, outputs, name='AutoEncoder')


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

