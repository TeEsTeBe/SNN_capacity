import numpy as np

from evaluators.base_evaluator import BaseEvaluator


class ClassificationEvaluator(BaseEvaluator):

    def evaluate(self, targets, reconstruction):
        reconstruction_correct = np.argmax(reconstruction, axis=1) == np.argmax(targets, axis=1)
        n_correct = np.count_nonzero(reconstruction_correct)
        results = {
            'accuracy': n_correct/len(targets),
        }

        return results

    def __init__(self):
        super().__init__()
