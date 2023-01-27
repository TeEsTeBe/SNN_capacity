import numpy as np
from sklearn.metrics import cohen_kappa_score

from evaluators.base_evaluator import BaseEvaluator


class BinaryEvaluator(BaseEvaluator):

    def __init__(self):
        super().__init__()

    def evaluate(self, targets, reconstruction):
        reconstruction[reconstruction > 0.5] = 1
        reconstruction[reconstruction <= 0.5] = 0

        results = {
            'mse': float(np.mean((targets - reconstruction) ** 2)),
            'squared_corr_coeff': float(np.corrcoef(targets, reconstruction)[0][1] ** 2),
            'kappa': float(cohen_kappa_score(targets, reconstruction)),
            'accuracy': np.count_nonzero(reconstruction==targets)/len(targets)
        }

        return results


