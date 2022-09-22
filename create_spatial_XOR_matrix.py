import os
import argparse

import numpy as np

from utils.input_utils import get_gaussian_XOR_input_values


def parse_cmd():
    parser = argparse.ArgumentParser()

    parser.add_argument('--output_folder')
    parser.add_argument('--N', type=int, default=1000)
    parser.add_argument('--num_steps', type=int, default=6000)
    parser.add_argument('--std', type=float, default=1.)
    parser.add_argument('--min', type=float, default=0.)
    parser.add_argument('--max', type=float, default=1.)

    return parser.parse_args()


def main():
    args = parse_cmd()

    os.makedirs(args.output_folder, exist_ok=True)

    signal = np.random.choice(np.arange(0, 4), size=args.num_steps, replace=True)
    states = get_gaussian_XOR_input_values(signal, max_value=args.max, min_value=args.min, n_generators=args.N,
                                           std=args.std)

    signal_path = os.path.join(args.output_folder, 'input_signal.npy')
    np.save(signal_path, signal)
    print(f'Signal stored to "{signal_path}"')

    states_path = os.path.join(args.output_folder, 'spatialXOR_statemat.npy')
    np.save(states_path, states)
    print(f'State matrix stored to "{states_path}"')


if __name__ == "__main__":
    main()
