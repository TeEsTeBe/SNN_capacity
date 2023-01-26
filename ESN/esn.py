import numpy as np
import matplotlib.pyplot as plt

import numpy.linalg as alg
from sklearn.decomposition import PCA


def test_loading():
    print("ESN library loaded")


# all kinds of connection matrices

def CM_Initialise_Uniform(in_shape, out_shape, scale=1.0, bias=0.0):
    # uniform in [-1.0,1.0]
    CM = 2.0 * np.random.rand(in_shape, out_shape) - 1.0
    return scale * CM + bias


def CM_Initialise_Normal(in_shape, out_shape, scale=1.0, bias=0.0):
    CM = scale * np.random.randn(in_shape, out_shape) + bias
    return CM


def CM_Initialise_Orthogonal(in_shape, out_shape):
    n = max(in_shape, out_shape)
    H = np.random.randn(n, n)
    Q, R = alg.qr(H)
    return Q[:in_shape, :out_shape]


# my own implementation
def set_abs_small_vals_to_zero(matrix, density):
    n_zeros = int(matrix.size * (1 - density))
    zero_threshold = np.sort(np.abs(matrix).flatten())[n_zeros - 1]
    matrix[np.abs(matrix) <= zero_threshold] = 0.
    return matrix


def set_randomly_to_zero(matrix, density):
    mask = (np.random.random(size=matrix.shape) > density)
    matrix[mask] = 0.
    return matrix


def CM_Initialise_Sparse_Orthogonal(in_shape, out_shape, density_denominator):
    n = max(in_shape, out_shape)
    # n_q = int(np.ceil(n * (1/density_denominator)))

    CM = np.zeros(shape=(n, n))

    for offset in range(density_denominator):
        n_q = int(np.ceil((n - offset) / density_denominator))
        H = np.random.randn(n_q, n_q)
        Q, R = alg.qr(H)
        real_q_n = n - offset
        CM[offset::density_denominator, offset::density_denominator] = Q

    np.random.shuffle(CM)
    np.random.shuffle(CM.T)

    return CM[:in_shape, :out_shape]


# Nonlinearities & other tools

def ReLu(x, slope=1, slope_start=0):
    y = x.copy() * slope - slope * slope_start
    y[y <= 0] = 0.

    return y


def Lin(x):
    return x.copy()


# spectral radius scaling

def CM_scale_specrad(CM, SR):
    # CM is the original connection matrix, SR is the desired spectral radius
    nCM = CM.copy()
    EV = np.max(np.absolute(alg.eigvals(CM)))
    return SR * nCM / EV


# Generic ESN base class
# Requires external generation of the connection matrices:
#  * Input to reservoir (I2R)
#  * Bias to reservoir (B2R)
#  * Reservoir to reservoir (R2R)
#  * Output to reservoir (O2R) - for output feedback only
#
# Other arguments:
#  * nonlinearity (function)
#  * stdev of noise (standard normal) on state values, input values (only for Batch processing) 
#    and output values

class ESN:
    def __init__(self, I2R=np.array([]), B2R=np.array([]), R2R=np.array([]), O2R=np.array([]),
                 nonlinearity=lambda x: np.tanh(x), statenoise=0.0, inputnoise=0.0, readoutnoise=0.0):
        # number of neurons
        self.NN = R2R.shape[0]
        # number of inputs
        self.NI = I2R.shape[1]
        self.NLfun = nonlinearity

        # Connection matrices: convention Row-to-Column !!
        self.I2R = I2R.copy()
        self.B2R = B2R.copy()
        self.R2R = R2R.copy()
        self.O2R = O2R.copy()

        self.inputnoise = inputnoise
        self.statenoise = statenoise
        self.readoutnoise = readoutnoise

        self.state = np.zeros((1, self.NN))

    def Reset(self):
        self.state = np.zeros((1, self.NN))

    def Copy(self):
        NewRes = ESN(I2R=self.I2R.copy(), B2R=self.B2R.copy(), R2R=self.R2R.copy(), O2R=self.O2R.copy(),
                     nonlinearity=self.NLfun, statenoise=self.statenoise, inputnoise=self.inputnoise,
                     readoutnoise=self.readoutnoise)
        return NewRes

    def State(self):
        return self.state.copy()

    def _Update(self, inputs):
        self.state = self.NLfun(
            np.dot(inputs, self.I2R) + np.dot(self.state, self.R2R) + self.B2R + self.statenoise * np.random.randn(1,
                                                                                                                   self.NN))

    def _UpdateWithFeedback(self, inputs, feedback):
        self.state = self.NLfun(np.dot(inputs, self.I2R) + np.dot(self.state, self.R2R) + np.dot(feedback,
                                                                                                 self.O2R) + self.B2R + self.statenoise * np.random.randn(
            1, self.NN))

    def Batch(self, inputs):
        steps = inputs.shape[0]
        states = np.zeros((steps, self.NN))
        for tt in range(steps):
            n_inp = inputs[tt:tt + 1, :]
            if self.inputnoise > 0:
                n_inp = n_inp + self.inputnoise * np.random.randn(n_inp.shape[0], n_inp.shape[1])
            self._Update(n_inp)
            states[tt:tt + 1, :] = self.State()
        if self.readoutnoise > 0.0:
            states = states + self.readoutnoise * np.random.randn(states.shape[0].states.shape[1])
        return states

    def BatchWithFeedback(self, inputs, feedback):
        steps = inputs.shape[0]
        states = np.zeros((steps, self.NN))
        for tt in range(steps):
            self._UpdateWithFeedback(inputs[tt:tt + 1, :], feedback[tt:tt + 1, :])
            states[tt:tt + 1, :] = self.State()
        return states
