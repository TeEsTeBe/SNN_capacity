import numpy as np

from tasks.base_task import BaseTask
from evaluators.analog_evaluator import AnalogEvaluator


class NARMA(BaseTask):
    default_params_gteq10 = {
        'alpha': 0.3,
        'beta': 0.05,
        'gamma': 1.5,
        'eps': 0.1
    }
    default_params_smaller10 = {
        'alpha': 0.2,
        'beta': 0.004,
        'gamma': 1.5,
        'eps': 0.001
    }

    def __init__(self, steps, inputs=None, n=10, alpha=None, beta=None, gamma=None, eps=None, delay=None):
        assert delay is None, "delay is not implemented for NARMA task"
        super().__init__()
        self.steps = steps
        self.n = n
        if n >= 10:
            self.defaults = self.default_params_gteq10.copy()
        else:
            self.defaults = self.default_params_smaller10.copy()
        self.alpha = self.defaults['alpha'] if alpha is None else alpha
        self.beta = self.defaults['beta'] if beta is None else beta
        self.gamma = self.defaults['gamma'] if gamma is None else gamma
        self.eps = self.defaults['eps'] if eps is None else eps
        self.delay = delay
        self.input_dimension = 1
        if inputs is None:
            self.input = self._generate_input()
        else:
            inputs = self._scale_input(inputs)
            self._check_input(inputs)
            self.input = inputs

        self.target = self._generate_target()

    def _scale_input(self, input):
        print(f'input is scaled to values between 0 and 0.5. (Previous min: {input.min()}, previous max: {input.max()}')
        input = np.interp(input, (input.min(), input.max()), (0., 0.5))

        return input

    def _check_input(self, inputs):
        assert inputs.min() >= 0., f"All inputs should be between 0. and 0.5, but they are between {inputs.min()} and {inputs.max()}!"
        assert inputs.min() <= 0.5, f"All inputs should be between 0. and 0.5, but they are between {inputs.min()} and {inputs.max()}!"
        assert len(inputs.shape) == 1, "The input for the NARMA tasks needs to be one dimensional"

    def _generate_input(self):
        inputs = np.random.uniform(0., 0.5, size=self.steps)

        return inputs

    def _generate_target(self):
        target = np.zeros_like(self.input)

        for t in range(self.n - 1, len(self.input) - 1):
            target[t + 1] = self.alpha * target[t] + self.beta * target[t] * np.sum(target[t - (self.n - 1):t + 1]) \
                            + self.gamma * self.input[t - (self.n - 1)] * self.input[t] + self.eps
            assert (not np.any(np.isinf(target))), "Unstable solution"

        target = target[self.n:]

        return target

    def get_default_evaluator(self):
        return AnalogEvaluator()

    def get_state_filter(self):
        state_filter = np.ones(shape=(self.steps), dtype=np.bool)
        state_filter[:self.n] = False

        return state_filter
