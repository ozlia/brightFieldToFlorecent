from dcae_blocks.DCAE import dcae, dcae_squeeze_unit
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, LeakyReLU, concatenate
from tensorflow import keras


class DCAE_Chain:

    def __init__(self, num_blocks, input_shape=(128, 128, 3)):
        inputs = keras.Input(shape=input_shape)

        # feature initialization stage
        h_minus1 = Conv2D(32, (3, 3), strides=1, activation="relu", padding="same")(inputs)
        h_0 = Conv2D(32, (3, 3), strides=1, activation="relu", padding="same")(h_minus1)
        x = h_0

        # multi-scale mapping stage
        skip_connections_output = []
        for _ in range(num_blocks):
            dcae_output = dcae(2, x)
            skip_connections_output.append(dcae_output)
            if len(skip_connections_output) > 1:
                x = concatenate(skip_connections_output)
            else:
                x = dcae_output

        # The reconstruction stage
        h_lt = dcae_squeeze_unit(h_minus1, x)
        i_sr = Conv2D(6, (3, 3), strides=1, activation="relu", padding="same")(h_lt)

        model = keras.Model(inputs, i_sr)
        model.compile(optimizer="adam", loss='mse')
        self.model = model

    def train(self, train_x, train_label, val_set=0.0, model_dir="model3D_full"):
        model = self.model
        model.summary()
        callbacks = [
            keras.callbacks.ModelCheckpoint("%sBasicAEModel3D.h5" % model_dir, save_best_only=True)
        ]
        model.fit(train_x, train_label, batch_size=32, epochs=10, verbose=1,
                  validation_split=val_set,
                  callbacks=callbacks)
        model.save(model_dir)
