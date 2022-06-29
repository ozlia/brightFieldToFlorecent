from models.ICNN import ICNN
import getpass
import os
from tensorflow.keras.layers import Conv2D
from tensorflow.keras import Model, Input, backend, models
from helpers import utils


class CrossDomain(ICNN):

    def __init__(self, input_dim=(128, 128, 6), batch_size=32, epochs=100):
        self.input_dim = input_dim
        self.batch_size = batch_size
        self.epochs = epochs
        self.USER = getpass.getuser().split("@")[0]
        self.dir = utils.DIRECTORY
        if not os.path.exists(self.dir):
            os.makedirs(self.dir)

        inputs = Input(shape=input_dim)
        encoder_path = '/home/ozlia/B2B_Mitochondria/B2B_Mitochondria_15-05-2022_22-08'
        decoder_path = '/home/ozlia/F2F_Mitochondria/F2F_Mitochondria_15-05-2022_19-14'
        self.encoder = models.load_model(encoder_path).get_layer('Encoder')
        self.decoder = models.load_model(decoder_path).get_layer('Decoder')
        self.encoder.trainable = False
        self.decoder.trainable = False

        # mapping_layer
        mapping_layer_input_shape = self.encoder.output_shape
        mapping_layer_inputs = Input(
            shape=(mapping_layer_input_shape[1], mapping_layer_input_shape[2], mapping_layer_input_shape[3]))
        x = Conv2D(128, 3, activation='relu', padding='same')(mapping_layer_inputs)
        mapping_layer_outputs = Conv2D(512, 3, activation='relu', padding='same')(x)
        self.mapping_layer = Model(mapping_layer_inputs, mapping_layer_outputs, name='feature_mapping_layer')

        model = Model(inputs, self.decoder(self.mapping_layer(self.encoder(inputs))), name="Auto_Encoder")
        model.summary()
        self.model = model
        model.compile(optimizer="adam", loss='mse')


def sampler(layers):
    std_norm = backend.random_normal(shape=(backend.shape(layers[0])[0], 128), mean=0, stddev=1)
    return layers[0] + layers[1] * std_norm
