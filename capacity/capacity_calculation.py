import scipy
import scipy.io
import scipy.integrate
import scipy.interpolate
import scipy.linalg
import scipy.signal

import copy
from scipy.stats import chi2
import numpy as np


def test_loading():
    print("Capacity library loaded")


# helper functions

def polyval(x, c):
    out = np.zeros(x.shape)
    for i in range(c.__len__()):
        if not (c[i] == 0.0):
            out += c[i] * x ** (i)
    return out


def extract_mean(states):
    N = states.shape[1]
    zmstates = np.zeros_like(states)
    for i in range(N):
        zmstates[:, i:i + 1] = states[:, i:i + 1] - scipy.mean(states[:, i:i + 1])
    return zmstates


# Basis functions for continuous-valued reservoirs

def legendre_step(n, x, nm1, nm2):
    ''' Compute Legendre n+1 function from Legendre n and n-1 functions 
        These are orthogonal for uniform random inputs in [-1,1]
    '''

    if n == 0:
        l = np.ones_like(x)
    elif n == 1:
        l = x.copy()
    else:
        l = ((2.0 * n - 1.0) * x * nm1 - (n - 1.0) * nm2) / float(n)

    return [l, nm1]


def legendre_incremental(n, x):
    ''' Recursively compute Legendre polynomial of order n, and return as a function
        These are orthogonal for uniform random inputs in [-1,1]
    '''
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


def generate_task(taskfun=legendre_incremental, input=[], variables=1, positions=[0], delay=1, powerlist=[1]):
    # calculate the desired output for the current iterator position and given input
    inp = scipy.atleast_2d(input.flatten()).T
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
    ''' Compute the non-linear memory capacity based on correlations
    '''

    score = np.zeros(target.shape[1])

    # if R_inv==None:  # gives an error in recent versions of Python
    if not (R_inv.any()):
        R_inv, crank = scipy.linalg.pinv(np.dot(states.T, states) / states.shape[0], return_rank=True)
        print("Estimated rank of state covariance matrix = ", crank)

    for k in range(target.shape[1]):
        P = np.dot(states.T, target[:, k:k + 1]) / (states.shape[0])
        score[k] = (np.dot(np.dot(P.T, R_inv), P)) / np.mean(target[:, k:k + 1] * target[:, k:k + 1])

        # P = states.T @ target / steps
        # (P.T @ R_inv) @ P  / var(target)

        # target is standardized (mean = 0, std = 1) -> mean(target^2) = var(target)
        # see def task()

    totalscore = np.sum(score)

    if return_all_scores:
        return [totalscore, score]
    else:
        return totalscore


# capacity iterator

class capacity_iterator():
    # Iterator class to measure capacity profiles -- with lots of parameters.
    # Usage: 
    #    nlmc=nlmc_iterator(options)
    #    totalscore,tags,numbases,dimensions = nlmc.collect(inputs,outputs)
    #
    # tags is the data structure containing all detailed information about the measured profile
    # NOTE: first sample of states is assumed to be response to first input sample, NOT initial state
    #
    # verbosity: -1 only gives total capacity, 0 (default) gives total capacity per degree, 1 gives thousands of lines output (one line per capacity, only for debugging)
    #
    # Typical parameters to set for measuring capacity profiles of dynamical systems:
    #
    #  * orth_factor: between 2.0 (long datasets) and 10.0 (shorter datasets)
    #  * delskip: threshold is ignored for delays up to this value (for systems with an initial response delay)
    #  * windowskip: threshold is ignored in window loop up to this value (for systems with feedback delay)
    #  * score_leak: for values <1.0, the current score is lowpassed before applying the halting threshold
    #                also useful for systems with feedback delay and/or with periodic response
    #  * basis: basis functions to be used; default = legendre polynomials (for uniform [0,1] input data)
    #           this can also be a GrammSchmittbasis class member
    #  
    # Additional bounded search parameters for measuring task profiles or when automatic search becomes too extensive: 
    #
    #  * for bounded delay search, set mindel and maxdel to desired values (additionally, set m_delay to False to switch off thresholding)
    #  * for bounded degree search, set mindeg and maxdeg to desired values (additionally, set m_degrees to False to switch off thresholding)
    #  * for bounded variables search, set minvars and maxvars  to desired values (additionally, set m_variables to False to switch off thresholding)
    #  * for bounded window search, set minwindow and maxwindow (additionally, set m_window and m_windowpos to False to switch off thresholding)
    #  * to completely switch off thresholding, set all 'm_'-parameters to False 

    def __init__(self, basis=legendre_incremental, mindel=1, maxdel=100000, delskip=0, delhold=0,
                 mindeg=1, maxdeg=100,
                 minwindow=1, maxwindow=10000, windowskip=0,
                 minvars=1, maxvars=100,
                 corr_cond=None,
                 orth_factor=2.0, score_leak=1.0,
                 p_threshold=0.999, d_threshold=0.001, a_threshold=1.0,  # PCA_threshold=1.0e-10,
                 m_delay=True, m_windowpos=False, m_window=True,
                 m_powerlist=False, m_variables=False, m_degrees=True,
                 maxbases=1000000, verbose=0, debug=False, profile=True):
        # Set ranges for sweeps
        self.mindel = mindel
        self.maxdel = maxdel
        self.delskip = delskip
        self.delhold = delhold
        self.holding = 0
        self.score_leak = score_leak
        self.orth_factor = orth_factor
        self.profile = profile
        self.debug = debug
        if p_threshold <= 1.0:
            self.p_threshold = float(p_threshold)
        else:  # assume threshold is specified as a percentage
            self.p_threshold = float(p_threshold) / 100.0

        self.a_threshold = float(a_threshold)
        self.d_threshold = d_threshold
        self.corr_cond = corr_cond
        # self.PCA_th=PCA_threshold
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

        self.verbose = verbose

    def reset(self):
        # Initialize iterations
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
        # inputs=inputs[:,0]
        self.reset()
        donext = True

        # Normalise state matrix
        estates -= np.mean(estates, axis=0)
        estates /= np.std(estates, axis=0)

        samples = estates.shape[0]
        self.dimensions = estates.shape[1]

        # estates=states+self.statenoise*np.random.randn(samples,self.dimensions)

        if self.verbose > 1:
            print('State space dimensions = ' + str(self.dimensions))

        if self.dimensions > 1:
            # self.corrmat=np.cov(estates)
            covmat = np.dot(estates.T, estates) / estates.shape[0]
            self.corrmat, crank = scipy.linalg.pinv(covmat, return_rank=True, cond=self.corr_cond)
            print("Estimated rank of state covariance matrix = ", crank)
        else:
            self.corrmat = np.ones((1, 1))

        # states=states

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

    def task(self, input):
        output = generate_task(taskfun=self.taskfun,
                               input=input, variables=self.variables, positions=self.positions,
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
        # self.leaky_score=1.0
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
