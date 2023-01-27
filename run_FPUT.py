#!/usr/env python
from FPUT.FPUT_input import run_multi_input_experiment, filename, check_convergence
import os
import argparse

import numpy as np
import yaml
from pathlib import Path


def parse_cmd():
    parser = argparse.ArgumentParser()

    parser.add_argument("--alpha", type=float, default=0.25)
    parser.add_argument("--tau_relax", type=int, default=10)
    parser.add_argument("--nbr_batches", type=int, default=100000)
    parser.add_argument("--warmup_batches", type=int, default=10)
    parser.add_argument("--init_epsilon", type=float, default=0.05)
    parser.add_argument("--trial", type=int, default=0)
    parser.add_argument("--discrete", default=False, action="store_true")
    parser.add_argument("--force", default=False, action="store_true")
    parser.add_argument("--in_dim", type=int, default=1)
    parser.add_argument(
        "--uniques",
        type=int,
        default=4,
        help="how many unique values are to be presented in case of discrete input. ignored if --discrete is not given",
    )
    parser.add_argument("--in_width", type=int, default=64)
    parser.add_argument("--in_variance", type=float, default=1 / 3)
    parser.add_argument("--osc", type=int, default=64)
    return parser.parse_args()


def pretty_print(params, pre_text="these parameters:", post_text=""):
    print(pre_text)
    for k, v in params.items():
        print(f"{k}: {v}")
    print(post_text)


def main(args):
    print(args)
    if "discrete" not in args or not args["discrete"]:
        del args["uniques"]
    overwrite = args["force"]
    del args["force"]
    print(args)

    fput_file_prefix = filename(prefix="./Data/FPUT/", postfix="/", **args)
    fput_file_prefix = Path(fput_file_prefix)
    metadata_file = fput_file_prefix / "metadata.yaml"
    trajectory_file = fput_file_prefix / "trajectories.npy"
    input_file = fput_file_prefix / "scaled_input_seq.npy"
    try:
        if overwrite:
            raise FileNotFoundError
        if not check_convergence(metadata_file):
            pretty_print(
                args,
                post_text="lead to diverging trajectories. Please try something else",
            )
            # raise OverflowError
        else:
            print(
                f"File {trajectory_file} has been found. if you would you like to overwrite, pass --force"
            )
    except FileNotFoundError:
        print(f"File {trajectory_file} not Found, will run simulation for")
        pretty_print(args)
        run_multi_input_experiment(file_prefix=fput_file_prefix, **args)
        if not check_convergence(metadata_file):
            pretty_print(
                args,
                post_text="lead to diverging trajectories. Please try something else",
            )
            # raise OverflowError


if __name__ == "__main__":
    main(args=vars((parse_cmd())))
