import argparse
import yaml
from SNN.simulation import SimulationRunner


def parse_cmd():
    parser = argparse.ArgumentParser()
    parser.add_argument('parameter_file', help="Path to the parameter yaml file.")
    parser.add_argument('--num_threads', type=int, default=1, help="Number of threads used for the simulation.")

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_cmd()
    with open(args.parameter_file, 'r') as param_file:
        params = yaml.safe_load(param_file)
    params['paramfile'] = args.parameter_file
    del params['trial']
    simulationrunner = SimulationRunner(num_threads=args.num_threads, **params)
    simulationrunner.run()
