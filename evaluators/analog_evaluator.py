import numpy as np

from evaluators.base_evaluator import BaseEvaluator


class AnalogEvaluator(BaseEvaluator):

    def __init__(self):
        super().__init__()

    def evaluate(self, targets, reconstruction):
        mse = np.mean((targets - reconstruction) ** 2)
        rmse = np.sqrt(mse)
        nmse = mse / np.std(targets)  # w is missing? (see Goudarzi et al. 2014, p. 4)
        rnmse = np.sqrt(nmse)
        nrmse = np.sqrt(mse) / (np.max(targets) - np.min(targets))
        corr_coeff = np.corrcoef(targets, reconstruction)[0][1]
        results = {
            'mse': float(mse),
            'squared_corr_coeff': float(corr_coeff ** 2),
            'rmse': float(rmse),
            'rnmse': float(rnmse),
            'nrmse': float(nrmse),
        }

        return results
