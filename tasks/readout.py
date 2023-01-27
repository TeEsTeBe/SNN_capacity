import numpy as np
from sklearn.linear_model import LinearRegression


def train_readout(states, target_signal):
    regression = LinearRegression(n_jobs=-1, fit_intercept=True, copy_X=True).fit(states, target_signal)
    reconstructed_signal = regression.predict(states)

    return reconstructed_signal, regression


def mean_squared_error(signal1, signal2):
    return np.mean((signal1 - signal2) ** 2)


def squared_correlation_coefficient(signal1, signal2):
    return np.corrcoef(signal1, signal2)[0][1]**2

