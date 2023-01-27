import numpy as np

from tasks.base_task import BaseTask
from evaluators.classification_evaluator import ClassificationEvaluator


class DelayedClassification(BaseTask):

    def __init__(self, steps, n_classes, delay=0, input=None):
        super().__init__()
        self.steps = steps
        self.n_classes = n_classes
        self.input_dimension = n_classes
        self.delay = delay
        if input is not None and len(input.shape) == 2:
            self.input = input
        else:
            self.input = self._generate_input(signal=input)
        self.target = self._generate_target()

    def _check_input(self, inputs):
        pass

    def _generate_input(self, signal=None):
        if signal is not None:
            unique_signals = np.unique(signal)
            assert len(unique_signals) == self.n_classes, "The given signal needs to have the same number of different values as you want to have classes"

        # one-hot-encoding
        inputs = np.zeros(shape=(self.steps, self.input_dimension))
        for step_idx in range(self.steps):

            if signal is None:
                class_idx = np.random.randint(0, self.n_classes)
            else:
                class_idx = np.where(unique_signals==signal[step_idx])[0][0]

            inputs[step_idx, class_idx] = 1

        return inputs

    def _generate_target(self):
        if self.delay > 0:
            target = self.input[:-self.delay]
        else:
            target = self.input[:]

        return target

    def get_default_evaluator(self):
        return ClassificationEvaluator()

    def get_state_filter(self):
        state_filter = np.ones(self.steps, dtype=np.bool)
        state_filter[:self.delay] = False

        return state_filter

