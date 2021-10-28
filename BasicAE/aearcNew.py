from tensorflow import keras
from keras import layers


def get_model(img_size, color):
    inputs = keras.Input(shape=img_size + (color,))  ## (128, 128, 1)

    ### [First half of the network: downsampling inputs] ###
    x = layers.Conv2D(32, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same")(inputs)
    x = layers.MaxPooling2D((2, 2), padding="same")(x)
    x = layers.Conv2D(8, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same")(x)
    x = layers.MaxPooling2D((2, 2), padding="same")(x)
    x = layers.Conv2D(8, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same")(x)
    x = layers.MaxPooling2D((2, 2), padding="same")(x)
    x = layers.Conv2D(8, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same")(x)

    ### [seconed half of the net work upsampling]
    x = layers.UpSampling2D(2)(x)
    x = layers.Conv2DTranspose(8, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same")(x)
    x = layers.UpSampling2D((2, 2))(x)
    x = layers.Conv2DTranspose(32, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same")(x)
    x = layers.UpSampling2D((2, 2))(x)
    outputs = layers.Conv2D(color, 3, activation="sigmoid", padding="same")(x)

    model = keras.Model(inputs, outputs)
    model.compile(optimizer="adam", loss="binary_crossentropy")
    return model
