import numpy as np
import numpy.linalg as alg


def CM_Initialise_Uniform(in_shape, out_shape, scale=1.0, bias=0.0):
    """
    Initialize a connection matrix with uniform distribution.

    Parameters
    ----------
    in_shape : int
        Number of rows in the connection matrix (input dimension).
    out_shape : int
        Number of columns in the connection matrix (output dimension).
    scale : float
        Scale factor for the connection matrix.
    bias : float
        Bias for the connection matrix.

    Returns
    -------
    ndarray
        Connection matrix with uniform distribution in [-1.0, 1.0].
    """

    CM = 2.0 * np.random.rand(in_shape, out_shape) - 1.0
    return scale * CM + bias


def CM_Initialise_Normal(in_shape, out_shape, scale=1.0, bias=0.0):
    """
    Initialize a connection matrix with normal distribution.

    Parameters
    ----------
    in_shape : int
        Number of rows in the connection matrix (input dimension).
    out_shape : int
        Number of columns in the connection matrix (output dimension).
    scale : float
        Scale factor for the connection matrix.
    bias : float
        Bias for the connection matrix.

    Returns
    -------
    ndarray
        Connection matrix with normal distribution.
    """

    CM = scale * np.random.randn(in_shape, out_shape) + bias
    return CM


def CM_Initialise_Orthogonal(in_shape, out_shape):
    """
    Initialize an orthogonal connection matrix.

    Parameters
    ----------
    in_shape : int
        Number of rows in the connection matrix (input dimension).
    out_shape : int
        Number of columns in the connection matrix (output dimension).

    Returns
    -------
    ndarray
        Orthogonal connection matrix.
    """
    n = max(in_shape, out_shape)
    H = np.random.randn(n, n)
    Q, R = alg.qr(H)
    return Q[:in_shape, :out_shape]


def set_randomly_to_zero(matrix, density):
    mask = (np.random.random(size=matrix.shape) > density)
    matrix[mask] = 0.
    return matrix


def CM_Initialise_Sparse_Orthogonal(in_shape, out_shape, density_denominator):
    """
    Initialize a sparse orthogonal connection matrix (density=1/density_denominator).

    Parameters
    ----------
    in_shape : int
        Number of rows in the connection matrix (input dimension).
    out_shape : int
        Number of columns in the connection matrix (output dimension).
    density_denominator : int
        Density denominator for the connection matrix.

    Returns
    -------
    ndarray
        Sparse orthogonal connection matrix with a density of 1/density_denominator
    """

    n = max(in_shape, out_shape)

    CM = np.zeros(shape=(n, n))

    for offset in range(density_denominator):
        n_q = int(np.ceil((n - offset) / density_denominator))
        H = np.random.randn(n_q, n_q)
        Q, R = alg.qr(H)
        CM[offset::density_denominator, offset::density_denominator] = Q

    np.random.shuffle(CM)
    np.random.shuffle(CM.T)

    return CM[:in_shape, :out_shape]


def ReLu(x, slope=1, slope_start=0):
    y = x.copy() * slope - slope * slope_start
    y[y <= 0] = 0.

    return y


def Lin(x):
    return x.copy()


def CM_scale_specrad(CM, SR):
    """
    Scale the spectral radius of a connection matrix.

    Parameters
    ----------
    CM : ndarray
        Connection matrix to be scaled.
    SR : float
        Desired spectral radius.

    Returns
    -------
    ndarray
        Scaled connection matrix with spectral radius of SR.
    """

    nCM = CM.copy()
    EV = np.max(np.absolute(alg.eigvals(CM)))
    return SR * nCM / EV


class ESN:
    """
    Echo State Network (ESN) class.

    This class implements an ESN and requires external generation of the connection matrices: Input to reservoir (I2R),
    Bias to reservoir (B2R), Reservoir to reservoir (R2R), and Output to reservoir (O2R) - for output feedback only.
    """

    def __init__(self, I2R=np.array([]), B2R=np.array([]), R2R=np.array([]), O2R=np.array([]),
                 nonlinearity=lambda x: np.tanh(x), statenoise=0.0, inputnoise=0.0, readoutnoise=0.0):
        """
        Initialize an ESN

        Parameters
        ----------
        I2R: ndarray
            Connection matrix from input to reservoir.
        B2R: ndarray
            Connection matrix from bias to reservoir.
        R2R: ndarray
            Connection matrix from reservoir to reservoir.
        O2R: ndarray
            Connection matrix from output to reservoir (for output feedback only)
        nonlinearity: function
            Nonlinearity function
        statenoise: float
            Standard deviation of noise on state values
        inputnoise: float
            Standard deviation of noise on input values
        readoutnoise: float
            Standard deviation of noise on output values
        """

        # number of neurons
        self.NN = R2R.shape[0]
        # number of inputs
        self.NI = I2R.shape[1]
        self.NLfun = nonlinearity

        # Connection matrices: convention Row-to-Column
        self.I2R = I2R.copy()
        self.B2R = B2R.copy()
        self.R2R = R2R.copy()
        self.O2R = O2R.copy()

        self.inputnoise = inputnoise
        self.statenoise = statenoise
        self.readoutnoise = readoutnoise

        self.state = np.zeros((1, self.NN))

    def State(self):
        return self.state.copy()

    def _Update(self, inputs):
        """
        Update the state of the reservoir.

        Parameters
        ----------
        inputs : ndarray
            Inputs to be processed.
        """

        self.state = self.NLfun(
            np.dot(inputs, self.I2R) + np.dot(self.state, self.R2R) + self.B2R + self.statenoise * np.random.randn(1,
                                                                                                                   self.NN))

    def _UpdateWithFeedback(self, inputs, feedback):
        """
        Update the state of the reservoir with feedback.

        Parameters
        ----------
        inputs : ndarray
            Inputs to be processed.
        feedback : ndarray
            Feedback to be used during processing.
        """

        self.state = self.NLfun(np.dot(inputs, self.I2R) + np.dot(self.state, self.R2R) + np.dot(feedback,
                                                                                                 self.O2R) + self.B2R + self.statenoise * np.random.randn(
            1, self.NN))

    def Batch(self, inputs):
        """
        Process the given inputs.

        Parameters
        ----------
        inputs : ndarray
            Inputs to be processed.

        Returns
        -------
        ndarray
            States of the reservoir after processing the inputs.
        """

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
        """
        Process the given inputs with feedback.

        Parameters
        ----------
        inputs : ndarray
            Inputs to be processed.
        feedback: ndarray
            Feedback to be used during processing

        Returns
        -------
        ndarray
            States of the reservoir after processing the inputs.
        """
        steps = inputs.shape[0]
        states = np.zeros((steps, self.NN))
        for tt in range(steps):
            self._UpdateWithFeedback(inputs[tt:tt + 1, :], feedback[tt:tt + 1, :])
            states[tt:tt + 1, :] = self.State()
        return states
