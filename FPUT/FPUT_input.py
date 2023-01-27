from FPUT.FPUT_analysis import analyse_damped
from FPUT.Simulation import DampedSimulation
import FPUT.utils

import numpy as np
from contextlib import contextmanager
from pathlib import Path
import yaml
from itertools import product


class FixedInputDampedSimulation(DampedSimulation):
    def __init__(self, input_seq, warmup_batches=10, **simparams):

        super().__init__(**simparams)
        self.setInputSeq(input_seq=np.array([0.0 for i in range(warmup_batches)]))
        self.simulate_batches(nbr_batches=warmup_batches)

        self.setInputSeq(input_seq=np.array(input_seq))


def used_input_seq_and_trajectories(
    input_seq=None,
    alpha=0.25,
    tau_relax=100,
    input_amplitude=0.01,
    input_duration=10,
    nbr_batches=90,
    warmup_batches=10,
    readout_every="stim_offset",
    init_epsilon=0.05,
    init_seed=200,
    input_seed=88111111,
    input_save_file=None,
    trajectory_save_file=None,
    **add_simparams,
):
    simparams = dict(
        init_epsilon=init_epsilon,
        input_seed=input_seed,
        alpha=alpha,
        tau_relax=tau_relax,
        input_amplitude=input_amplitude,
        input_duration=input_duration,
        init_seed=init_seed,
        **add_simparams,
    )
    if input_seq is None:
        input_seq = np.random.uniform(-1, 1, size=nbr_batches)

    simulation = FixedInputDampedSimulation(
        input_seq=input_seq, warmup_batches=warmup_batches, **simparams
    )
    simulation.startReadout(readout_every=readout_every, n_readouts=1)
    input_seq = simulation.simulate_batches(nbr_batches=nbr_batches)
    if input_save_file is not None:
        np.save(file=input_save_file, arr=input_seq)
    trajectories = np.array(simulation.readout)
    if trajectory_save_file is not None:
        np.save(file=trajectory_save_file, arr=trajectories)

    return input_seq, trajectories


def equidistant_zero_mean(seq, distance=2):
    values = sorted(set(seq.flatten()))
    diff = np.diff(values)
    assert len(set(diff)) == 1
    uniques = len(values)
    min_value = np.min(seq)
    seq -= min_value
    return (seq - (uniques - 1) / 2) * distance


def equidistant_fix_mean_and_var(seq, mean, var):
    values = sorted(set(seq.flatten()))
    diff = np.diff(values)
    assert len(set(diff)) == 1
    step = diff[0]
    uniques = len(values)
    cur_var = step ** 2 * (uniques ** 2 - 1) / 12
    seq = equidistant_zero_mean(
        seq=seq, distance=np.sqrt(12 * var / (uniques ** 2 - 1))
    )
    seq += mean

    return seq


def stretch_and_buffer(
    seq,
    axis,
    stretch_length,
    total_length,
    buffer_value=0,
):
    assert len(seq.shape) <= 2
    number_of_buffers = total_length - seq.shape[axis] * stretch_length
    assert number_of_buffers >= 0
    buffer_sections = max(1, seq.shape[axis] - 1)
    buffer_positions = []
    for i in range(number_of_buffers):
        factor = (i % buffer_sections) + 1
        buffer_positions.append(factor * stretch_length)

    idx = sorted(buffer_positions)
    seq = np.repeat(a=seq, repeats=stretch_length, axis=axis)
    return np.insert(arr=seq, obj=idx, values=buffer_value, axis=axis)


def run_input_experiment(file_prefix, **simparams):

    if simparams["discrete"]:
        scaled_input = np.random.randint(
            0, simparams["uniques"], size=simparams["osc"] * simparams["nbr_batches"]
        )
        input_seq = (scaled_input * 2 - 3) * simparams["discretization_factor"]

        print("input_seq", set(input_seq))
    else:
        input_seq = np.random.uniform(-1, 1, size=simparams["nbr_batches"])
    input_seq, trajectories = used_input_seq_and_trajectories(
        input_seq=input_seq, **simparams
    )
    if simparams["discrete"]:
        scaled_input = scaled_input[: len(input_seq)]
        with writeable_file(file_prefix / "scaled_input_seq.npy") as f:
            np.save(file=f, arr=scaled_input)
    metadata = dict(diverged=bool(np.isnan(trajectories).any()))
    print(metadata)
    with writeable_file(file_prefix / "trajectories.npy") as f:
        np.save(file=f, arr=trajectories)
    with writeable_file(file_prefix / "input_seq.npy") as f:
        np.save(file=f, arr=input_seq)
    with writeable_file(file_prefix / "parameters.yaml", mode="w") as f:
        yaml.dump(simparams, f)
    with writeable_file(file_prefix / "metadata.yaml", mode="w") as f:
        yaml.dump(metadata, f)


def run_multi_input_experiment(file_prefix, **simparams):

    if simparams["discrete"]:
        scaled_input = np.reshape(
            np.random.randint(
                0,
                simparams["uniques"],
                size=simparams["in_dim"] * simparams["nbr_batches"],
            ),
            (simparams["nbr_batches"], simparams["in_dim"]),
        )
        input_seq = equidistant_fix_mean_and_var(
            seq=scaled_input, mean=0, var=simparams["in_variance"]
        )
        print("scaled shape", scaled_input.shape)
        print("input shape", input_seq.shape)
        input_seq = stretch_and_buffer(
            seq=input_seq,
            axis=1,
            stretch_length=simparams["in_width"],
            total_length=simparams["osc"],
        )
    else:
        try:
            scaled_input = np.reshape(
                np.random.uniform(
                    -1, 1, size=simparams["nbr_batches"] * simparams["in_dim"]
                ),
                (simparams["nbr_batches"], simparams["in_dim"]),
            )
            input_seq = stretch_and_buffer(
                seq=scaled_input,
                axis=1,
                stretch_length=simparams["in_width"],
                total_length=simparams["osc"],
            )
        except KeyError:
            scaled_input = np.reshape(
                np.random.uniform(-1, 1, size=64 * simparams["nbr_batches"]),
                (simparmas["osc"], simparams["nbr_batches"]),
            )
            input_seq = scaled_input
    input_seq, trajectories = used_input_seq_and_trajectories(
        input_seq=input_seq, **simparams
    )
    scaled_input = scaled_input[: len(input_seq)]
    with writeable_file(file_prefix / "scaled_input_seq.npy") as f:
        np.save(file=f, arr=scaled_input)
    metadata = dict(diverged=bool(np.isnan(trajectories).any()))

    metadata = dict(diverged=bool(np.isnan(trajectories).any()))
    print(metadata)
    with writeable_file(file_prefix / "trajectories.npy") as f:
        np.save(file=f, arr=trajectories)
    with writeable_file(file_prefix / "input_seq.npy") as f:
        np.save(file=f, arr=input_seq)
    with writeable_file(file_prefix / "parameters.yaml", mode="w") as f:
        yaml.dump(simparams, f)
    with writeable_file(file_prefix / "metadata.yaml", mode="w") as f:
        yaml.dump(metadata, f)


def filename(prefix, postfix, **kwargs):
    return Path(
        prefix
        + "_".join(f"{key}={kwargs[key]}" for key in sorted(kwargs.keys()))
        + postfix
    )


@contextmanager
def writeable_file(file_path, mode="wb"):
    try:
        file = open(file_path, mode=mode)
        yield file
    except FileNotFoundError:
        Path.mkdir(Path(file_path).parent.resolve(), parents=True)
        file = open(file_path, mode=mode)
        yield file
    finally:
        file.close()


def check_convergence(metadata_file):
    with open(metadata_file, "r") as f:
        metadata = yaml.safe_load(f)
    if metadata["diverged"]:
        return False
    else:
        return True


if __name__ == "__main__":

    default_simparams = dict(
        alpha=0.25,
        tau_relax=10,
        nbr_batches=100000,
        warmup_batches=10,
        init_epsilon=0.05,
        trial=0,
        in_dim=1,
        in_width=64,
        osc=64,
    )
    for in_dim, in_width, input_variance, uniques in product(
        [1], [1, 5, 10, 64], [1 / 12, 1 / np.sqrt(12), 3], [2, 3, 6]
    ):
        simparams = {
            **default_simparams,
            "in_dim": in_dim,
            "in_width": in_width,
            "uniques": uniques,
            "discrete": False,
        }
        name = filename(prefix="./../Data/multi_input", postfix="/", **simparams)
        run_multi_input_experiment(file_prefix=name, **simparams)
