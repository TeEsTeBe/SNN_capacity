import os
import unittest
from argparse import Namespace

import numpy as np
from scipy.special import eval_legendre

import run_capacity


class TestCapacity(unittest.TestCase):

    def test_capacity(self):
        n_steps = 10000
        n_units = 50
        input_signal = np.random.uniform(-1, 1, n_steps)
        legendre_deg2 = eval_legendre(2, input_signal)
        legendre_deg3 = eval_legendre(3, input_signal)
        legendre_deg4 = eval_legendre(4, input_signal)
        state_matrix = np.random.uniform(-1, 1, n_units * n_steps).reshape((n_units, n_steps))

        state_matrix[0, :] = input_signal
        state_matrix[1, :] = legendre_deg2
        state_matrix[2, :] = legendre_deg3
        state_matrix[3, :] = legendre_deg4

        state_matrix[:, 1:] = state_matrix[:, 1:] * 0.75 + state_matrix[:, :-1] * 0.25

        os.makedirs('data', exist_ok=True)
        input_path = os.path.join('data', 'test_inputs.npy')
        states_path = os.path.join('data', 'test_states.npy')
        np.save(input_path, input_signal)
        np.save(states_path, state_matrix)

        args_dict = {
            'name': 'test_run',
            'input': input_path,
            'states_path': states_path,
            'results_file': os.path.join('data', 'test_results.tsv'),
            'capacity_results': os.path.join('data', 'test_capacity_results'),
            'max_degree': 1000,
            'max_delay': 1000,
            'm_variables': True,
            'm_powerlist': True,
            'm_windowpos': True,
            'orth_factor': 2,
            'figures_path': os.path.join('data', 'test_figures'),
            'n_warmup': 500,
            'use_scipy': False,
            'sample_ids': None,
            'sample_size': None,
            'sample_step': None,
            'delskip': 0,
            'windowskip': 0,
            'verbosity': 0
        }
        args_namespace = Namespace(**args_dict)
        run_capacity.main(args_namespace)


if __name__ == "__main__":
    unittest.main()
