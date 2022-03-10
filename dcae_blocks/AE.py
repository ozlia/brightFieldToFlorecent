from tensorflow.keras.layers import Conv2D, Conv2DTranspose, LeakyReLU, concatenate


def ae(input):
    stride = 2

    # encoder
    x = Conv2D(32, (2, 2), strides=stride, activation="relu", padding="same")(input)
    ius = x

    x = Conv2D(32, (2, 2), strides=stride, activation="relu", padding="same")(x)
    # decoder
    x = (Conv2DTranspose(32, (3, 3), strides=stride, padding="same"))(x)
    x = LeakyReLU()(x)

    x = concatenate([x, ius])

    x = (Conv2DTranspose(32, (3, 3), strides=stride, padding="same"))(x)
    x = LeakyReLU()(x)

    return x
