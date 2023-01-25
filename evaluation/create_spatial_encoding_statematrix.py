import os
import argparse

import numpy as np
import yaml

from SNN.utils.input_utils import get_gaussian_input_values


def parse_cmd():
    parser = argparse.ArgumentParser()

    parser.add_argument('--parameters', required=True, help='Path to the parameter file of the simulation.')
    parser.add_argument('--inputs', required=True, help='Path to the inputs file of the simulation.')
    parser.add_argument('--output', help='Path to store the resulting state matrix to.')
    parser.add_argument('--N', type=int, help='Number of encoding variables.')

    return parser.parse_args()


def main():
    args = parse_cmd()

    inputs = np.load(args.inputs)

    with open(args.parameters, 'r') as params_file:
        parameters = yaml.safe_load(params_file)

    std = parameters['spatial_std_factor']
    max_value = parameters['input_max_value']
    min_value = parameters['input_min_value']

    if args.N is None:
        n_encoder = 447 if parameters['network_type'] == 'microcircuit' else 1000
    else:
        n_encoder = args.N

    enc_state_matrix = get_gaussian_input_values(input_values=inputs, n_encoder=n_encoder, min_value=min_value,
                                                 max_value=max_value, std=std)

    if args.output is None:
        args.output = os.path.join(os.path.dirname(args.parameters), 'encoder_state_matrix.npy')

    np.save(args.output, enc_state_matrix)
    print(f'Encoder state matrix saved to {args.output}')


if __name__ == "__main__":
    main()
