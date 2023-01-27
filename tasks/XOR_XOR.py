import numpy as np

from tasks.base_task import BaseTask
from evaluators.binary_evaluator import BinaryEvaluator


class XORXOR(BaseTask):

    def __init__(self, steps, inputs=None, delay=0):
        self.steps = steps
        # super().__init__(inputs=inputs)
        super().__init__()
        self.delay = delay
        if inputs is None:
            self.input = self._generate_input()
        else:
            self._check_input(inputs)
            self.input = inputs

        self.target = self._generate_target()
        self.input_dimension = 4

    def get_state_filter(self):
        state_filter = np.ones(shape=(self.steps), dtype=np.bool)
        if self.delay > 0:
            state_filter[:self.delay] = False

        return state_filter

    def _check_input(self, inputs):
        assert len(inputs.shape) == 2 and inputs.shape[1] == 4, "The input for the XOR tasks needs to be 4 dimensional"
        unique_vals = np.unique(inputs)
        assert unique_vals[0] == 0 and unique_vals[1] == 1, "The input should contain only ones and zeros"

    def _generate_input(self):
        inputs = np.random.uniform(-1, 1, self.steps * 4)
        inputs[inputs>=0] = 1
        inputs[inputs<=0] = 0
        # inputs[inputs<=0] = -1
        inputs = inputs.reshape(self.steps, 4)

        return inputs

    def _generate_target(self):
        xor1 = np.logical_xor(self.input[:, 0], self.input[:, 1])
        xor2 = np.logical_xor(self.input[:, 2], self.input[:, 3])
        xorxor = np.logical_xor(xor1, xor2).astype(int)

        if self.delay > 0:
            xorxor = xorxor[:-self.delay]

        return xorxor

    def get_default_evaluator(self):
        return BinaryEvaluator()
