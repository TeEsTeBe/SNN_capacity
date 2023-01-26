import unittest
from argparse import Namespace

import run_ESN


class TestESN(unittest.TestCase):

    def test_run_ESN(self):
        args_dict = {
            'input': None,
            'runname': 'testrun',
            'groupname': 'testgroup',
            'data_path': 'data',
            'steps': 5000,
            'nodes': 50,
            'input_scaling': 0.5,
            'spectral_radius': 0.95,
            'n_warmup': 500,
            'init_normal': False,
            'orthogonalize': True,
            'ortho_density_denominator': 1,
            'use_relu': False,
            'relu_slope': 1,
            'relu_start': 0,
            'use_linear_activation': False,
            'results_file': None,
            'figures_path': None,
            'recurrence_density': None,
            'trial': 1,
            'seed': None
        }
        args_namespace = Namespace(**args_dict)
        run_ESN.main(args_namespace)


if __name__ == "__main__":
    unittest.main()
