import numpy as np

from tasks.base_task import BaseTask
from evaluators.binary_evaluator import BinaryEvaluator


class TemporalContinuousXOR(BaseTask):

    def _check_input(self, inputs):
        assert np.max(inputs) <= 1, "The inputs should be uniformly distributed between -1 and 1"
        assert np.max(inputs) >= -1, "The inputs should be uniformly distributed between -1 and 1"

    def _generate_input(self):
        inputs = np.random.uniform(-1, 1, self.steps)

        return inputs

    def _generate_target(self):
        inputs1 = self.inputs[:-1]
        inputs1[inputs1 > 0] = 1
        inputs1[inputs1 <= 0] = 0

        inputs2 = self.inputs[1:]
        inputs2[inputs2 > 0] = 1
        inputs2[inputs2 <= 0] = 0

        target = np.logical_xor(inputs1, inputs2).astype(int)

        return target

    def get_default_evaluator(self):
        return BinaryEvaluator()

    def get_state_filter(self):
        state_filter = np.ones(shape=(self.steps), dtype=np.bool)
        state_filter[:1] = False

        return state_filter

    def __init__(self, steps, inputs):
        super().__init__()
        self.steps = steps
        if inputs is None:
            self.inputs = self._generate_input()
        else:
            self.inputs = inputs

        self.target = self._generate_target()

        self.input_dimension = 1
