# from sporco.metric import psnr
# import numpy as np
# print(psnr(np.array([1,2,3]),np.array([1,0,3])))

import keras.backend as K
from tensorflow.keras.losses import MeanSquaredError as mse,binary_crossentropy as bce
from tensorflow import sqrt


# def psnr(y_true, y_pred):
#     max_pixel = 255.0
#     RMSE = sqrt(mse(y_true, y_pred))
#     return 20 * log10(max_pixel / RMSE)



# implementation of wasserstein loss
def wasserstein_loss(y_true, y_pred):
    return K.mean(y_true * y_pred)