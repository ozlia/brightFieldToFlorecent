from tensorflow.keras.layers import Conv2D, concatenate
from dcae_blocks.AE import ae


def dcae(num_ae, inputs):
    prev_layer = inputs
    iud = []
    for i in range(num_ae):
        ae_output = ae(prev_layer)
        iud.append(ae_output)
        if len(iud) > 1:
            prev_layer = concatenate(iud)
        else:
            prev_layer = ae_output

    iud_g = prev_layer
    h_g = dcae_squeeze_unit(inputs, iud_g)
    return h_g


def dcae_squeeze_unit(h_g_minus1, iud_g):
    x = Conv2D(32, (1, 1), activation="relu", padding="same")(iud_g)
    x = concatenate([h_g_minus1, x])
    h_g = Conv2D(6, (3, 3), activation="relu", padding="same")(x)
    return h_g
