import numpy as np
from tensorflow import keras


def corr_coef(y_true, y_pred) -> float:
    """Calculates the Pearson correlation coefficient between the inputs.
    Parameters
    ----------
    y_true
        First input.
    y_pred
        Second input.
    Returns
    -------
    float
        Pearson correlation coefficient between the inputs.
    """
    if y_true is None or y_pred is None:
        return None
    y_pred = y_pred[:,:,0]
    y_true = y_true[:,:,0]
    assert y_true.shape == y_pred.shape, "Inputs must be same shape"
    mean_a = np.mean(y_true)
    mean_b = np.mean(y_pred)
    std_a = np.std(y_true)
    std_b = np.std(y_pred)
    cc = np.mean((y_true - mean_a) * (y_pred - mean_b)) / (std_a * std_b)
    # np.corrcoef
    return cc


@staticmethod
def weighted_mse(yTrue, yPred):
    ones = keras.backend.ones_like(yTrue[0, :])
    idx = keras.backend.cumsum(ones)

    return keras.backend.mean((1 / idx) * keras.backend.square(yTrue - yPred))


def np_corr(y_true, y_pred):
    return np.corrcoef(y_true.flatten(), y_pred.flatten())