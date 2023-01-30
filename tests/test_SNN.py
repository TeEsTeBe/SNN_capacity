import os
import unittest

import yaml
import nest

from SNN.simulation.simulation_runner import SimulationRunner


class TestSNN(unittest.TestCase):

    @staticmethod
    def run_input_and_noise_combinations(parameters):
        input_type_list = ['spatial_DC', 'spatial_rate', 'uniform_DC', 'uniform_rate', 'step_DC', 'step_rate']
        noise_loop_list = [None, parameters['step_duration']]
        for noise_loop_duration in noise_loop_list:
            for input_type in input_type_list:
                print(f'\n\n ------- input_type: {input_type}, noise loop duration: {noise_loop_duration}\n')
                nest.ResetKernel()
                nest.set_verbosity('M_ERROR')
                parameters['input_type'] = input_type
                parameters['noise_loop_duration'] = noise_loop_duration
                simulation_runner = SimulationRunner(**parameters)
                simulation_runner.run()

    def test_microcircuit_run(self):
        print('\n----------------------------------\n\tMicrocircuit Tests\n----------------------------------\n')
        nest.ResetKernel()
        nest.set_verbosity('M_ERROR')
        parameter_file_path = os.path.join('SNN_parameter_files', 'microcircuit_test_params.yaml')
        with open(parameter_file_path, 'r') as parameter_file:
            parameters = yaml.safe_load(parameter_file)
        parameters['paramfile'] = parameter_file_path
        del parameters['trial']
        self.run_input_and_noise_combinations(parameters)

    def test_brunel_run(self):
        print('\n----------------------------------\n\tBRN Tests\n----------------------------------\n')
        parameter_file_path = os.path.join('SNN_parameter_files', 'brunel_test_params.yaml')
        with open(parameter_file_path, 'r') as parameter_file:
            parameters = yaml.safe_load(parameter_file)
        parameters['paramfile'] = parameter_file_path
        del parameters['trial']
        self.run_input_and_noise_combinations(parameters)


if __name__ == "__main__":
    unittest.main()
