from FPUT_input import (
    run_input_experiment,
    filename,
    check_convergence,
    lyapunov_calculation,
)
from pathlib import Path
import subprocess
from itertools import product
import pickle
import matplotlib.pyplot as plt
import numpy as np

RUN_CAPACITY = "/projects/capacity/capacity_v2/run_capacity.py"


def capacity(output_file, fput_file_prefix, **simparams):

    fput_file_prefix = Path(fput_file_prefix)
    metadata_file = fput_file_prefix / "metadata.yaml"
    trajectory_file = fput_file_prefix / "trajectories.npy"
    input_file = fput_file_prefix / "input_seq.npy"
    try:
        if not check_convergence(metadata_file):
            print("these parameters:")
            for k, v in simparams.items():
                print(f"{k}: {v}")
            print("lead to diverging trajectories. Please try something else")
            return
    except FileNotFoundError:
        print(f"File {trajectory_file} not Found, will run simulation for")
        for k, v in simparams.items():
            print(f"{k}: {v}")
        run_input_experiment(file_prefix=fput_file_prefix, **simparams)
        if not check_convergence(metadata_file):
            print("these parameters:")
            for k, v in simparams.items():
                print(f"{k}: {v}")
            print("lead to diverging trajectories. Please try something else")
            return

    subprocess.run(
        [
            "python3",
            RUN_CAPACITY,
            "--name",
            "micks_FPUT",
            "--input",
            input_file,
            "--states_path",
            trajectory_file,
            "--capacity_results",
            output_file,
        ]
    )


def plot_capacity_over_x(ax, x, x_values, trials, default_sim_params):
    readout = {}
    for v, trial in product(
        x_values,
        range(trials),
    ):
        simparams = {
            **default_simparams,
            x: v,
            "trial": trial,
        }
        name = filename(
            prefix="/projects/capacity/FPUT/Data/", postfix="/", **simparams
        )
        output_file = filename(
            prefix="/projects/capacity/micks_capacities/", postfix=".pkl", **simparams
        )

        try:
            with open(output_file, "rb") as f:
                d = pickle.load(f)
        except FileNotFoundError:
            capacity(output_file=output_file, fput_file_prefix=name, **simparams)
            try:
                with open(output_file, "rb") as f:
                    d = pickle.load(f)
            except FileNotFoundError:
                continue
        try:
            readout[v].append(d["total_capacity"])
        except KeyError:
            readout[v] = [d["total_capacity"]]

    abscissas = []
    mean = []
    std = []
    for k, v in readout.items():
        abscissas.append(k)
        mean.append(np.mean(v))
        std.append(np.std(v))
    mean = np.array(mean)
    std = np.array(std)
    ax.plot(abscissas, mean, zorder=2)
    ax.fill_between(abscissas, mean - std, mean + std, zorder=1, alpha=0.8)
    print(abscissas)
    print(mean)
    print(std)


def plot_lyapunov_over_x(ax, x, x_values, trials, default_sim_params):
    readout = {}
    for v, trial in product(
        x_values,
        range(trials),
    ):
        simparams = {
            **default_simparams,
            x: v,
            "trial": trial,
        }
        name = filename(
            prefix="/projects/capacity/FPUT/Data/", postfix="/", **simparams
        )
        output_file = filename(
            prefix="/projects/capacity/FPUT/Data/", postfix="/LS.npy", **simparams
        )

        try:
            with open(output_file, "rb") as f:
                print("try loading")
                d = np.load(f)
        except FileNotFoundError:
            print("couldnt load, will calculate")
            lyapunov_calculation(**simparams, LS_save_file=output_file)
            try:
                with open(output_file, "rb") as f:
                    print("try loading again")
                    d = np.load(f)
            except FileNotFoundError:
                print("couldnt calculate apparently")
                continue
        try:
            print("d", d)
            readout[v].append(d[0])
        except KeyError:
            readout[v] = [d[0]]

    abscissas = []
    mean = []
    std = []
    for k, v in readout.items():
        abscissas.append(k)
        mean.append(np.mean(v))
        std.append(np.std(v))
    mean = np.array(mean)
    std = np.array(std)
    ax.plot(abscissas, mean, zorder=2)
    ax.fill_between(abscissas, mean - std, mean + std, zorder=1, alpha=0.8)
    print(abscissas)
    print(mean)
    print(std)


if __name__ == "__main__":
    fig = plt.figure(constrained_layout=True, figsize=[13, 10])
    ax_dict = fig.subplot_mosaic(
        [
            ["c1", "c2", "c3"],
            ["l1", "l2", "l3"],
        ],
    )
    trials = 15

    default_simparams = dict(
        alpha=0.25,
        tau_relax=3,
        input_duration=1,
        input_amplitude=0.2,
        nbr_batches=100000,
        warmup_batches=10,
        init_epsilon=0.02,
    )

    print("starting panel 1", 40 * "-")
    plot_capacity_over_x(
        ax=ax_dict["c1"],
        x="input_amplitude",
        x_values=[
            0.01,
            0.1,
            0.2,
            0.3,
            0.4,
            0.5,
            0.55,
            0.6,
            0.61,
            0.62,
            0.621,
            0.622,
            0.623,
            0.624,
            0.625,
            0.626,
            0.627,
            0.628,
            0.629,
            0.63,
        ],
        trials=trials,
        default_sim_params=default_simparams,
    )
    ax_dict["c1"].set_xlabel("amplitude")
    ax_dict["c1"].set_ylabel("capacity")

    print("starting panel 2", 40 * "-")
    plot_capacity_over_x(
        ax=ax_dict["c2"],
        x="input_duration",
        x_values=[1, 2, 3, 4, 5, 6, 7],
        trials=trials,
        default_sim_params=default_simparams,
    )
    ax_dict["c2"].set_xlabel("duration")
    ax_dict["c2"].set_ylabel("capacity")

    print("starting panel 3", 40 * "-")
    plot_capacity_over_x(
        ax=ax_dict["c3"],
        x="tau_relax",
        x_values=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
        trials=trials,
        default_sim_params=default_simparams,
    )
    ax_dict["c3"].set_xlabel("duration")
    ax_dict["c3"].set_ylabel("capacity")

    default_simparams["nbr_batches"] = 100200
    default_simparams["warump_batches"] = 100
    print("starting panel 4", 40 * "-")
    plot_lyapunov_over_x(
        ax=ax_dict["l1"],
        x="input_amplitude",
        x_values=[
            0.01,
            0.1,
            0.2,
            0.3,
            0.4,
            0.5,
            0.55,
            0.6,
            0.61,
            0.62,
            0.621,
            0.622,
            0.623,
            0.624,
            0.625,
            0.626,
            0.627,
            0.628,
            0.629,
            0.63,
        ],
        trials=trials,
        default_sim_params=default_simparams,
    )
    ax_dict["l1"].set_xlabel("amplitude")
    ax_dict["l1"].set_ylabel("lyapunov exp.")

    print("starting panel 5", 40 * "-")
    plot_lyapunov_over_x(
        ax=ax_dict["l2"],
        x="input_duration",
        x_values=[1, 2, 3, 4, 5, 6, 7],
        trials=trials,
        default_sim_params=default_simparams,
    )
    ax_dict["l2"].set_xlabel("duration")
    ax_dict["l2"].set_ylabel("lyapunov exp.")

    print("starting panel 6", 40 * "-")
    plot_lyapunov_over_x(
        ax=ax_dict["l3"],
        x="tau_relax",
        x_values=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
        trials=trials,
        default_sim_params=default_simparams,
    )
    ax_dict["l3"].set_xlabel("duration")
    ax_dict["l3"].set_ylabel("lyapunov exp.")

    plt.savefig("test1.pdf")
    print("done")
