import numpy as np

from tasks.base_task import BaseTask
from evaluators.binary_evaluator import BinaryEvaluator


class XOR(BaseTask):

    def __init__(self, steps, inputs=None, delay=0):
        # super().__init__(inputs=inputs)
        super().__init__()
        self.steps = steps
        self.delay = delay
        if inputs is None:
            self.input = self._generate_input()
        else:
            self._check_input(inputs)
            self.input = inputs

        self.target = self._generate_target()

        self.input_dimension = 2

    def get_state_filter(self):
        state_filter = np.ones(shape=(self.steps), dtype=np.bool)
        if self.delay > 0:
            state_filter[:self.delay] = False

        return state_filter

    def _check_input(self, inputs):
        assert len(inputs.shape) == 2 and inputs.shape[1] == 2, "The input for the XOR tasks needs to be two dimensional"
        unique_vals = np.unique(inputs)
        assert unique_vals[0] == 0 and unique_vals[1] == 1, "The input should contain only ones and zeros"

    def _generate_input(self):
        inputs = np.random.uniform(-1, 1, self.steps * 2)
        inputs[inputs>=0] = 1
        inputs[inputs<=0] = 0
        # inputs[inputs<=0] = -1
        inputs = inputs.reshape(self.steps, 2)

        return inputs

    def _generate_target(self):
        target = np.logical_xor(self.input[:, 0], self.input[:, 1]).astype(int)
        if self.delay > 0:
            target = target[:-self.delay]

        return target

    def get_default_evaluator(self):
        return BinaryEvaluator()
