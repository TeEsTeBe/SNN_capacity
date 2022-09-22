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
    parser.add_argument('--trial', type=int, default=1)

    return parser.parse_args()


def main():
    args = parse_cmd()

    os.makedirs(args.output_folder, exist_ok=True)

    signal_path = os.path.join(args.output_folder, f'input_signal__steps={args.num_steps}__trial={args.trial}.npy')
    if os.path.exists(signal_path):
        signal = np.load(signal_path)
        print(f'Loaded existing signal from "{signal_path}"')
    else:
        signal = np.random.choice(np.arange(0, 4), size=args.num_steps, replace=True)
        np.save(signal_path, signal)
        print(f'Signal stored to "{signal_path}"')
    states = get_gaussian_XOR_input_values(signal, max_value=args.max, min_value=args.min, n_generators=args.N,
                                           std=args.std)

    states_path = os.path.join(args.output_folder,
                               f'spatialXOR_statemat__N={args.N}__steps={args.num_steps}__std={args.std}'
                               f'__min={args.min}__max={args.max}__trial={args.trial}.npy')
    np.save(states_path, states)
    print(f'State matrix stored to "{states_path}"')


if __name__ == "__main__":
    main()
