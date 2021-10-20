from tensorflow import keras
from keras import layers
from keras.models import Sequential 

def get_model(img_size):
    inputs = keras.Input(shape = img_size + (3,))
#     ### [First half of the network: downsampling inputs] ###
    x = createConLayer(16, inputs)
    x = layers.MaxPooling2D(3, strides=2, padding="same")(x)
    x = createConLayer(32, x)
    x = layers.MaxPooling2D(3, strides=2, padding="same")(x)
    x = createConLayer(64, x)
    x = layers.MaxPooling2D(3, strides=2, padding="same")(x)
    x = createConLayer(128, x)
    x = layers.MaxPooling2D(3, strides=2, padding="same")(x)
    x = createConLayer(256, x)

#     ### [seconed half of the net work upsampling]
    x = layers.UpSampling2D(2)(x)
    x = createTransposeConLayer(128, x)
    x = layers.UpSampling2D(2)(x)
    x = createTransposeConLayer(64, x)
    x = layers.UpSampling2D(2)(x)
    x = createTransposeConLayer(32, x)
    x = layers.UpSampling2D(2)(x)
    x = createTransposeConLayer(16, x)
    outputs = layers.Conv2D(1, 3, activation="sigmoid", padding="same")(x)

    model = keras.Model(inputs, outputs)
    model.compile(optimizer="adam", loss="mean_squared_error", metrics = ["accuracy"])
    return model



def createConLayer(dim, inputs):
    x = layers.Conv2D(dim, (3,3), activation="relu", kernel_initializer ="he_normal", padding="same")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(dim, (3,3), activation="relu", kernel_initializer ="he_normal", padding="same")(x)
    x = layers.BatchNormalization()(x)
    return x
    
def createTransposeConLayer(dim, inputs):
    x = layers.Conv2DTranspose(dim, 3, padding="same")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Conv2DTranspose(dim, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    return x