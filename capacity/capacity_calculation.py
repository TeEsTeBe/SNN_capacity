import scipy
import scipy.io
import scipy.integrate
import scipy.interpolate
import scipy.linalg
import scipy.signal

import copy
from scipy.stats import chi2
import numpy as np


def legendre_step(n, x, nm1, nm2):
    """ Compute Legendre n+1 function from Legendre n and n-1 functions These are orthogonal for uniform random inputs
    in [-1,1] """

    if n == 0:
        l = np.ones_like(x)
    elif n == 1:
        l = x.copy()
    else:
        l = ((2.0 * n - 1.0) * x * nm1 - (n - 1.0) * nm2) / float(n)

    return [l, nm1]


def legendre_incremental(n, x):
    """ Recursively compute Legendre polynomial of order n, and return as a function
        These are orthogonal for uniform random inputs in [-1,1] """

    if n == 0:
        return np.ones_like(x)
    elif n == 1:
        return x
    else:
        l = x.copy()
        prev = np.ones_like(x)

        for index in range(int(n) - 1):
            [l, prev] = legendre_step(index + 2, x, l, prev)

    return l


def generate_task(taskfun=legendre_incremental, input_values=None, variables=1, positions=None, delay=1,
                  powerlist=None):
    """ Calculate the desired output for the current iterator position and given input """

    if input_values is None:
        input_values = np.array([])
    if positions is None:
        positions = np.array([0])
    if powerlist is None:
        powerlist = np.array([1])

    inp = np.atleast_2d(input_values.flatten()).T
    output = np.ones((inp.shape[0], 1))
    for i in np.arange(variables):
        pos = positions[i] + delay - 1
        if pos == 0:
            task = taskfun(powerlist[i], inp)
        else:
            task = taskfun(powerlist[i], np.concatenate((np.zeros((pos, inp.shape[1])), inp[:-pos, :])))
        output = output * task
    return output


# capacity metrics

def cov_capacity(states, target, return_all_scores=False, R_inv=None):  # ,zm_target=True,zm_states=True):
    """ Compute the non-linear memory capacity based on correlations """

    score = np.zeros(target.shape[1])

    if not (R_inv.any()):
        R_inv, crank = scipy.linalg.pinv(np.dot(states.T, states) / states.shape[0], return_rank=True)
        print("Estimated rank of state covariance matrix = ", crank)

    for k in range(target.shape[1]):
        P = np.dot(states.T, target[:, k:k + 1]) / (states.shape[0])
        score[k] = (np.dot(np.dot(P.T, R_inv), P)) / np.mean(target[:, k:k + 1] * target[:, k:k + 1])

    totalscore = np.sum(score)

    if return_all_scores:
        return [totalscore, score]
    else:
        return totalscore


class CapacityIterator():

    def __init__(self, basis=legendre_incremental, mindel=1, maxdel=100000, delskip=0,
                 mindeg=1, maxdeg=100,
                 minwindow=1, maxwindow=10000, windowskip=0,
                 minvars=1, maxvars=100,
                 orth_factor=2.0, score_leak=1.0,
                 m_delay=True, m_windowpos=False, m_window=True,
                 m_powerlist=False, m_variables=False, m_degrees=True,
                 maxbases=1000000, verbosity=0, debug=False):
        """ Iterator to measure capacity profiles

        Parameters
        ----------
        basis: function
            basis functions to be used; default = legendre polynomials
        mindel: int
            lower bound for delays
        maxdel: int
            upper bound for delays
        delskip: int
            threshold is ignored for delays up to this value (for systems with an initial response delay)
        mindeg: int
            lower bound for degrees
        maxdeg: int
            upper bound for degrees
        minwindow: int
            lower bound for window length
        maxwindow: int
            upper bound for window length
        windowskip: int
            threshold is ignored in window loop up to this value (for systems with feedback delay)
        minvars: int
            minimal number of variables
        maxvars: int
            maximum number of variables
        orth_factor: float
            Factor that increases the cutoff value
        score_leak: float
            for values <1.0, the current score is lowpassed before applying the halting threshold. Also useful for
            systems with feedback delay and/or with periodic response
        m_delay: bool
            Whether to assume a monotonous decrease of capacity with increasing delay
        m_windowpos: bool
            Whether to assume a monotonous decrease of capacity with increasing window positions
        m_window: bool
            Whether to assume a monotonous decrease of capacity with increasing window length
        m_powerlist: bool
            Whether to assume a monotonous decrease of capacity with longer power lists
        m_variables: bool
            Whether to assume a monotonous decrease of capacity with increasing number of variables
        m_degrees: bool
            Whether to assume a monotonous decrease of capacity with increasing degree
        maxbases: int
            maximum number of target functions that should be evaluated
        verbosity: int
            -1 only gives total capacity, 0 (default) gives total capacity per degree, 1 gives one line per capacity
        debug: bool
            whether to enable additional debug print outs
        """

        # Set ranges for sweeps
        self.mindel = mindel
        self.maxdel = maxdel
        self.delskip = delskip
        self.delhold = 0
        self.holding = 0
        self.score_leak = score_leak
        self.orth_factor = orth_factor
        self.profile = True
        self.debug = debug
        self.p_threshold = 0.999

        self.a_threshold = 1.0
        self.d_threshold = 0.001
        self.corr_cond = None
        self.mindeg = mindeg
        self.maxdeg = maxdeg
        self.minvars = minvars
        self.maxvars = maxvars
        self.minwindow = minwindow
        self.maxwindow = maxwindow
        self.windowskip = windowskip
        self.taskfun = basis

        # set monotonicity variables:
        # True means that if the total score for a given loop level is below threshold, 
        # further iterations accross this loop will be skipped
        self.monotonous_delay = m_delay
        self.monotonous_windowpos = m_windowpos
        self.monotonous_window = m_window
        self.monotonous_powerlist = m_powerlist
        self.monotonous_variables = m_variables
        self.monotonous_degree = m_degrees

        # Initialize iterations
        self.delay = self.mindel
        self.window = self.minwindow
        self.variables = self.minvars
        if self.window < self.variables:
            self.window = self.variables
        self.degree = self.mindeg
        if self.degree < self.variables:
            self.degree = self.variables

        if self.window == 1:
            self.positions = np.array([0])
            self.powerlist = np.array([self.mindeg])
        else:
            self.positions = range(self.variables - 1)
            self.positions.append(self.window - 1)
            if self.degree == self.variables:
                self.powerlist = np.ones(self.variables)
            else:
                self.powerlist = np.array([self.degree - self.variables + 1])
                self.powerlist.append(np.ones(self.variables - 1))

        # initialize cumulative scores
        self.delay_score = 0.0
        self.windowpos_score = 0.0
        self.window_score = 0.0
        self.powerlist_score = 0.0
        self.variables_score = 0.0
        self.degrees_score = 0.0
        self.prev_var_score = 1000.0

        self.scored = False  # variable to keep track of scoring use
        self.use_scores = True
        self.leaky_score = 0.0
        self.current_score = 0.0
        self.cumscore = 0.0

        self.maxbases = maxbases
        self.bases = 1

        self.tags = []

        self.verbose = verbosity

    def reset(self):
        """ Initialize iterations """

        self.delay = self.mindel
        self.holding = 0
        self.window = self.minwindow
        self.variables = self.minvars
        if self.window < self.variables:
            self.window = self.variables
        self.degree = self.mindeg
        if self.degree < self.variables:
            self.degree = self.variables

        if self.window == 1:
            self.positions = np.array([0])
            self.powerlist = np.array([self.mindeg])
        else:
            self.positions = range(self.variables - 1)
            self.positions.append(self.window - 1)
            if self.degree == self.variables:
                self.powerlist = np.ones(self.variables)
            else:
                self.powerlist = np.array([self.degree - self.variables + 1])
                self.powerlist.append(np.ones(self.variables - 1))

        # initialize cumulative scores
        self.delay_score = 0.0
        self.windowpos_score = 0.0
        self.window_score = 0.0
        self.powerlist_score = 0.0
        self.variables_score = 0.0
        self.degrees_score = 0.0
        self.prev_var_score = 1000.0

        self.scored = False  # variable to keep track of scoring use
        self.use_scores = True
        self.leaky_score = 0.0
        self.current_score = 0.0

        self.bases = 0
        self.cumscore = 0.0

        self.tags = []

    def collect(self, inputs, estates):
        """ Calculates the results for all capacity functions

        Parameters
        ----------
        inputs: ndarray
            input values
        estates: ndarray
            state matrix

        """

        self.reset()
        donext = True

        # Normalise state matrix
        estates -= np.mean(estates, axis=0)
        estates /= np.std(estates, axis=0)

        self.dimensions = estates.shape[1]

        if self.verbose > 1:
            print('State space dimensions = ' + str(self.dimensions))

        if self.dimensions > 1:
            covmat = np.dot(estates.T, estates) / estates.shape[0]
            self.corrmat, crank = scipy.linalg.pinv(covmat, return_rank=True, cond=self.corr_cond)
            print("Estimated rank of state covariance matrix = ", crank)
        else:
            self.corrmat = np.ones((1, 1))

        while donext:
            outputs = self.task(inputs)
            samples = float(estates[self.delay + self.window - 2:, :].shape[0])
            score = cov_capacity(estates[self.delay + self.window - 2:, :], outputs[self.delay + self.window - 2:, :],
                                 R_inv=self.corrmat)

            p_threshold = self.orth_factor * chi2.isf(1 - self.p_threshold, self.dimensions) / samples
            if self.a_threshold >= 1.0:
                threshold = p_threshold
            else:
                if p_threshold > self.a_threshold:
                    threshold = p_threshold
                else:
                    threshold = self.a_threshold

            self.score(score, threshold)
            donext = self.next()

        if self.verbose >= -1:
            tag = 'Total capacity=%(score).3f (%(perc).2f percent)' % {'score': self.totalscore(),
                                                                       'perc': 100.0 * self.totalscore() /
                                                                               estates.shape[1]}
            print(tag)
        return self.totalscore(), self.alltags(), self.bases, self.dimensions

    def task(self, input_values):
        output = generate_task(taskfun=self.taskfun,
                               input_values=input_values, variables=self.variables, positions=self.positions,
                               delay=self.delay, powerlist=self.powerlist)
        output -= output.mean()
        output /= output.std()
        return output

    def score(self, score, threshold):
        if not (self.use_scores):
            print('WARNING: scoring has been switched off, any further scores will be ignored')
        if self.scored:
            print('WARNING: duplicate scoring of individual, any further scores will be ignored')
        self.scored = True

        self.current_score = score

        if self.current_score > self.leaky_score:
            self.leaky_score = self.current_score
        else:
            self.leaky_score = (1.0 - self.score_leak) * self.leaky_score + self.score_leak * score

        if self.leaky_score >= threshold:
            self.cumscore += score
            if self.profile:
                self.tags.append({'degree': copy.deepcopy(self.degree),
                                  'variables': copy.deepcopy(self.variables),
                                  'powerlist': copy.deepcopy(self.powerlist),
                                  'window': copy.deepcopy(self.window),
                                  'window_positions': copy.deepcopy(self.positions),
                                  'delay': copy.deepcopy(self.delay),
                                  'score': copy.deepcopy(score),
                                  'l_score': copy.deepcopy(self.leaky_score),
                                  'threshold': threshold})

        else:
            self.current_score = 0.0
            self.leaky_score = 0.0

        if self.verbose > 0:
            tag = self.print_tag(threshold)
            if self.leaky_score >= threshold:
                print(tag)
            else:
                if self.debug:
                    print('S: ', tag)

    def totalscore(self):
        return self.cumscore

    def tag(self, threshold):
        tag = {'degree': self.degree,
               'variables': self.variables,
               'powerlist': self.powerlist,
               'window': self.window,
               'window_positions': self.positions,
               'delay': self.delay,
               'score': self.current_score,
               'l_score': self.leaky_score,
               'threshold': threshold}
        return tag

    def alltags(self):
        return self.tags

    def print_tag(self, threshold):
        tag = 'deg=%(degree)d' % {"degree": self.degree}
        tag += ', var=%(variables)d' % {"variables": self.variables}
        tag += ', pow=' + str(self.powerlist)
        tag += ', win=' + str(self.window)
        tag += ', pos=' + str(self.positions)
        tag += ', del=' + str(self.delay)
        tag += ', score=%(score).2e' % {"score": self.current_score}
        tag += ', leaky score=%(score).2e' % {"score": self.leaky_score}
        tag += ', cumscore=%(score).3f' % {"score": self.cumscore}
        tag += ', threshold=%(score).2e' % {"score": threshold}
        return tag

    def next(self):

        if self.bases >= self.maxbases:
            return False
        if not (self.scored) and self.use_scores:
            self.use_scores = False
            print("Warning: current task has not been scored -- scoring is incomplete and has been switched off")
        self.scored = False

        self.delay_score += self.current_score

        if self.delay <= self.delskip or not (self.use_scores) or not (self.monotonous_delay) or (
                self.use_scores and self.monotonous_delay and (self.current_score > 0.0 or (
                self.delay > self.delskip and self.current_score <= 0.0 and self.holding < self.delhold))):
            if self.delay + self.window - 1 < self.maxdel:
                if self.holding < self.delhold and self.delay > self.delskip and self.current_score <= 0.0:
                    self.holding += 1
                else:
                    self.holding = 0

                self.delay += 1
                self.current_score = 0.0
                self.bases += 1
                return True
        self.current_score = 0.0
        self.delay = self.mindel
        self.subs = np.zeros(self.dimensions, )

        self.windowpos_score += self.delay_score
        self.holding = 0

        if not (self.use_scores) or not (self.monotonous_windowpos) or (self.use_scores and self.monotonous_windowpos):
            if self.use_scores and self.monotonous_windowpos and (self.delay_score == 0.0) and (self.variables > 2):
                maxpos = np.arange(self.window - self.variables + 1, self.window - 1, 1.0)
                for index in np.arange(self.variables - 3, -1.0, -1.0).astype(int):
                    # find last index that is not at its maximal value
                    if self.positions[index + 1] < maxpos[index]:
                        # determine number of indices before that
                        if index == 0:
                            index2 = index
                        else:
                            for index2 in np.arange(index - 1, -1.0, -1.0).astype(int):
                                if self.positions[index2 + 1] < self.positions[index2 + 2] - 1:
                                    break
                        for ind in np.arange(index2, index + 1, 1).astype(int):
                            self.positions[ind + 1] = maxpos[ind]
                        break

            newpos = self.__nextpos()
            if len(newpos) > 0:
                self.positions = newpos
                self.delay = self.mindel
                self.delay_score = 0.0
                self.bases += 1
                return True
        self.delay_score = 0.0
        self.window_score += self.windowpos_score

        if self.window <= self.windowskip or not (self.use_scores) or not (self.monotonous_window) or (
                self.use_scores and self.monotonous_window and (self.windowpos_score > 0.0)):
            if (self.window < self.maxwindow) and (
                    (self.window < self.maxdel - self.delay + 1) or not (self.monotonous_window)):
                if self.variables > 1:
                    self.window += 1
                    self.positions = np.arange(self.variables)
                    self.positions[self.variables - 1] = self.window - 1
                    self.delay = self.mindel
                    self.windowpos_score = 0.0
                    self.bases += 1
                    return True
        self.windowpos_score = 0.0
        self.powerlist_score += self.window_score

        if not (self.use_scores) or not (self.monotonous_powerlist) or (
                self.use_scores and self.monotonous_powerlist and (self.window_score > 0.0)):
            newpowers = self.__nextpowers()
            if len(newpowers) > 0:
                self.powerlist = newpowers
                self.window = self.variables
                self.positions = np.arange(self.variables)
                self.positions[self.variables - 1] = self.window - 1
                self.delay = self.mindel
                self.window_score = 0.0
                self.bases += 1
                return True
        self.window_score = 0.0
        self.variables_score += self.powerlist_score

        if not (self.use_scores) or not (self.monotonous_variables) or (
                self.use_scores and self.monotonous_variables and (
                (self.powerlist_score > 0.0) or self.variables == 1)):
            if self.variables < self.maxvars:
                if self.variables < self.degree:
                    self.variables += 1
                    self.powerlist = np.ones(self.variables)
                    self.powerlist[0] = self.degree - (self.variables - 1)
                    self.window = self.variables
                    self.positions = np.arange(self.variables)
                    self.positions[self.variables - 1] = self.window - 1
                    self.delay = self.mindel
                    self.powerlist_score = 0.0
                    self.bases += 1
                    return True
        self.powerlist_score = 0.0
        self.degrees_score += self.variables_score

        if self.verbose >= 0:
            print('Total score for degree ', str(self.degree), ' = ', str(self.variables_score), '(cumulative total = ',
                  str(self.cumscore), ')')
        if not (self.use_scores) or not (self.monotonous_degree) or (self.use_scores and self.monotonous_degree and (
                (self.variables_score >= self.d_threshold) or (self.prev_var_score >= self.d_threshold))):
            if self.degree < self.maxdeg:
                self.degree += 1
                self.variables = self.minvars
                self.powerlist = np.ones(self.variables)
                self.powerlist[0] = self.degree - (self.variables - 1)
                self.window = self.variables
                self.positions = np.arange(self.variables)
                self.positions[self.variables - 1] = self.window - 1
                self.prev_var_score = self.variables_score
                self.delay = self.mindel
                self.variables_score = 0.0
                self.bases += 1
                return True
        self.variables_score = 0.0

        return False

    def __nextpos(self):
        nextpos = np.zeros(shape=(0, 0))
        for i in np.arange(self.variables - 2, 0, -1).astype(int):
            if self.positions[i] < self.window - (self.variables - i):
                nextpos = self.positions
                nextpos[i] += 1
                pos = nextpos[i] + 1
                for j in np.arange(i + 1, self.variables - 1).astype(int):
                    nextpos[j] = pos
                    pos += 1
                break
        return nextpos

    def __nextpowers(self):
        nextpowers = np.zeros(shape=(0, 0))
        if self.variables == 1:
            return nextpowers
        if self.degree == self.variables:
            return nextpowers
        freepowers = self.degree - self.variables
        powers = np.zeros(freepowers)
        index = 0
        for i in np.arange(self.variables).astype(int):
            if self.powerlist[i] > 1:
                for j in np.arange(1, self.powerlist[i]):
                    powers[index] = i
                    index += 1
        for i in np.arange(freepowers - 1, -1, -1).astype(int):
            if powers[i] < self.variables - 1:
                powers[i] += 1
                for j in np.arange(i + 1, freepowers).astype(int):
                    powers[j] = powers[i]
                nextpowers = np.ones(self.variables)
                for j in np.arange(freepowers).astype(int):
                    nextpowers[int(powers[j])] += 1
                break
        return nextpowers
