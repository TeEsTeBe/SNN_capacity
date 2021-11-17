import argparse
import yaml
from simulation.simulation_runner import SimulationRunner


def parse_cmd():
    parser = argparse.ArgumentParser()
    parser.add_argument('parameter_file', help="Path to the parameter yaml file.")

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_cmd()
    with open(args.parameter_file, 'r') as param_file:
        params = yaml.safe_load(param_file)
    params['paramfile'] = args.parameter_file
    simulationrunner = SimulationRunner(**params)
    simulationrunner.run()
