from FPUT_input import (
    run_input_experiment,
    run_multi_input_experiment,
    filename,
    check_convergence,
    lyapunov_calculation,
)
import pandas as pd
from pathlib import Path
import subprocess
from itertools import product
import pickle
import matplotlib.pyplot as plt
import numpy as np
import yaml
import os
from functools import partial
import multiprocessing as mp
import seaborn as sns
import matplotlib as mpl

# sns.set_style('dark', {'xtick.bottom': True, 'ytick.left': True})
SMALL_SIZE = 8
MEDIUM_SIZE = 10
BIGGER_SIZE = 12
plt.rc("font", size=SMALL_SIZE)  # controls default text sizes
plt.rc("axes", titlesize=BIGGER_SIZE)  # fontsize of the axes title
plt.rc("axes", labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
plt.rc("xtick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc("ytick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc("legend", fontsize=SMALL_SIZE)  # legend fontsize
plt.rc("figure", titlesize=BIGGER_SIZE)  # fontsize of the figure title
plt.rc("font", family="serif")
mpl.rcParams["figure.dpi"] = 600
# mpl.rcParams["text.usetex"] = True

RUN_CAPACITY = "/projects/capacity/capacity_v2/run_capacity.py"
RUN_TASK = "/users/dick/Projects/capacity/ESN/evaluate_task_separately.py"


def get_full_runname(runname, task, test_ratio, steps, trial, **kwargs):
    full_runname = (
        f"{runname}_task={task}_testratio={test_ratio}_steps={steps}_trial={trial}"
    )
    return full_runname


def capacity(output_file, fput_file_prefix, **simparams):
    try:
        with open(output_file, "rb") as f:
            d = pickle.load(f)
        return d
    except FileNotFoundError:
        print(f"couldn't find {output_file}, will calculate capacity")
        pass

    fput_file_prefix = Path(fput_file_prefix)
    metadata_file = fput_file_prefix / "metadata.yaml"
    trajectory_file = fput_file_prefix / "trajectories.npy"
    input_file = fput_file_prefix / "scaled_input_seq.npy"
    try:
        if not check_convergence(metadata_file):
            print("these parameters:")
            for k, v in simparams.items():
                print(f"{k}: {v}")
            print("lead to diverging trajectories. Please try something else")
            raise OverflowError
    except FileNotFoundError:
        print(f"File {trajectory_file} not Found, will run simulation for")
        for k, v in simparams.items():
            print(f"{k}: {v}")
        run_multi_input_experiment(file_prefix=fput_file_prefix, **simparams)
        if not check_convergence(metadata_file):
            print("these parameters:")
            for k, v in simparams.items():
                print(f"{k}: {v}")
            print("lead to diverging trajectories. Please try something else")
            raise OverflowError

    print(
        " ".join(
            [
                "python3",
                str(RUN_CAPACITY),
                "--name",
                "micks_FPUT",
                "--input",
                str(input_file),
                "--states_path",
                str(trajectory_file),
                "--capacity_results",
                str(output_file),
                "--orth_factor",
                "10",
                "--m_variables",
                "--m_powerlist",
                "--m_windowpos",
                "-v",
                "1",
            ]
        ),
        flush=True,
    )
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
            "--orth_factor",
            "10",
            "--m_variables",
            "--m_powerlist",
            "--m_windowpos",
            "-v",
            "1",
        ],
    )
    with open(output_file, "rb") as f:
        d = pickle.load(f)
    return d


def task_performance(output_file, fput_file_prefix, task_params, **simparams):
    full_runname = get_full_runname(**task_params)
    runpath = os.path.join(
        task_params["data_path"], task_params["groupname"], full_runname
    )
    output_file = os.path.join(runpath, output_file)

    try:
        with open(output_file, "r") as f:
            d = yaml.safe_load(f)
        return d
    except FileNotFoundError:
        print(output_file, "not found")
        pass

    fput_file_prefix = Path(fput_file_prefix)
    metadata_file = fput_file_prefix / "metadata.yaml"
    trajectory_file = fput_file_prefix / "trajectories.npy"
    if task_params["task"] == "narma":
        input_file = fput_file_prefix / "narma_input_seq.npy"
    else:
        input_file = fput_file_prefix / "scaled_input_seq.npy"
    try:
        if not check_convergence(metadata_file):
            print("these parameters:")
            for k, v in simparams.items():
                print(f"{k}: {v}")
            print("lead to diverging trajectories. Please try something else")
            raise OverflowError
    except FileNotFoundError:
        print(f"File {trajectory_file} not Found, will run simulation for")
        for k, v in simparams.items():
            print(f"{k}: {v}")
        run_multi_input_experiment(file_prefix=fput_file_prefix, **simparams)
        if not check_convergence(metadata_file):
            print("these parameters:")
            for k, v in simparams.items():
                print(f"{k}: {v}")
            print("lead to diverging trajectories. Please try something else")
            raise OverflowError
    if task_params["task"] == "narma":
        try:
            a = np.load(input_file)
        except FileNotFoundError:
            a = np.load(fput_file_prefix / "input_seq.npy")
            a = (a + 1) / 4
            assert np.max(a) == 0.5
            assert np.min(a) == 0
            np.save(arr=a, file=input_file)

    subprocess.run(
        [
            "python3",
            RUN_TASK,
            "--input",
            str(input_file),
            "--states",
            str(trajectory_file),
            "--steps",
            str(task_params["steps"]),
            "--task",
            task_params["task"],
            "--groupname",
            task_params["groupname"],
            "--runname",
            str(task_params["runname"]),
            "--test_ratio",
            str(task_params["test_ratio"]),
            "--data_path",
            str(task_params["data_path"]),
        ]
    )
    with open(output_file, "r") as f:
        d = yaml.safe_load(f)
    return d


def get_kappa(input_duration, input_amplitude, task_params, simparams):
    simparams["input_duration"] = input_duration
    simparams["input_amplitude"] = input_amplitude
    task_params["runname"] = filename(prefix="", postfix="", **simparams)
    prefix = filename(prefix="/projects/capacity/FPUT/Data/", postfix="/", **simparams)
    try:
        d = task_performance(
            output_file="test_results.yml",
            fput_file_prefix=prefix,
            task_params=task_params,
            **simparams,
        )
    except Exception as f:
        print(f)
        return np.nan
    return d["kappa"]


def capacity_by_degree(capacity_dict):

    degree_cap = {}
    for cap in capacity_dict["all_capacities"]:
        try:
            degree_cap[cap["degree"]] += cap["score"]
        except KeyError:
            degree_cap[cap["degree"]] = cap["score"]
    return degree_cap


def get_capacity(input_duration, input_amplitude, degrees, simparams):
    simparams["input_duration"] = input_duration
    simparams["input_amplitude"] = input_amplitude
    output_file = filename(
        prefix="/projects/capacity/micks_capacities/", postfix=".pkl", **simparams
    )
    prefix = filename(prefix="/projects/capacity/FPUT/Data/", postfix="/", **simparams)
    try:
        d = capacity(output_file=output_file, fput_file_prefix=prefix, **simparams)
    except Exception as f:
        print(f)
        return [np.nan for _ in range(degrees + 1)]
    cap_d = capacity_by_degree(d)
    return_list = [d["total_capacity"]] + [
        cap_d[i] if i in cap_d else 0 for i in range(1, degrees + 1)
    ]
    print(return_list, input_duration, input_amplitude, flush=True)
    return return_list


def main(
    discretization_factor=1 / 3,
    alpha=0.25,
    init_epsilon=0.05,
    in_dim=1,
    uniques=4,
    in_width=64,
    multiprocess=True,
):

    default_simparams = dict(
        alpha=alpha,
        tau_relax=10,
        nbr_batches=100000,
        warmup_batches=10,
        init_epsilon=init_epsilon,
        trial=0,
        in_dim=in_dim,
        uniques=uniques,
        in_width=in_width,
        osc=64,
    )
    degrees = 3
    input_durations = range(1, 70)
    input_amplitudes = [
        0.001,
        0.002,
        0.003,
        0.004,
        0.005,
        0.006,
        0.007,
        0.008,
        0.009,
        0.010,
        0.011,
        0.012,
        0.013,
        0.014,
        0.015,
        0.016,
        0.017,
        0.018,
        0.019,
        0.020,
        0.021,
        0.022,
        0.023,
        0.024,
        0.025,
        0.026,
        0.027,
        0.028,
        0.029,
        0.030,
        0.031,
        0.032,
        0.033,
        0.034,
        0.035,
        0.036,
        0.037,
        0.038,
        0.039,
        0.040,
        0.041,
        0.042,
        0.043,
        0.044,
        0.045,
        0.046,
        0.047,
        0.048,
        0.049,
        0.050,
    ]

    xor_task_params = dict(
        task="xor",
        test_ratio=0.3,
        steps=default_simparams["nbr_batches"],
        trial=1,
        groupname="g",
        data_path="data/xor",
    )
    get_xor_kappa_default = partial(
        get_kappa,
        task_params=xor_task_params,
        simparams={
            **default_simparams,
            "discrete": True,
            "in_variance": discretization_factor,
        },
    )

    narma_task_params = dict(
        task="narma",
        test_ratio=0.3,
        steps=default_simparams["nbr_batches"],
        trial=1,
        groupname="g",
        data_path="data/narma",
    )
    get_narma_kappa_default = partial(
        get_kappa,
        task_params=narma_task_params,
        simparams={
            **default_simparams,
            "discrete": False,
            "in_variance": discretization_factor,
        },
    )

    get_capacity_default = partial(
        get_capacity,
        degrees=degrees,
        simparams={**default_simparams, "discrete": False},
    )

    pool = mp.Pool(mp.cpu_count())

    print("start capacity calculations", flush=True)

    if multiprocess:
        capacities = pool.starmap(
            get_capacity_default, product(input_durations, input_amplitudes)
        )
    else:
        capacities = []
        for dur, amp in product(input_durations, input_amplitudes):
            capacities.append(get_capacity_default(dur, amp))
    capacities = np.array(capacities).reshape(
        len(input_durations), len(input_amplitudes), degrees + 1
    )

    print("start xor calculation", flush=True)
    if multiprocess:
        xor_kappas = pool.starmap(
            get_xor_kappa_default, product(input_durations, input_amplitudes)
        )
    else:
        xor_kappas = []
        for d, a in product(input_durations, input_amplitudes):
            xor_kappas.append(get_xor_kappa_default(d, a))
    xor_kappas = np.array(xor_kappas).reshape(
        len(input_durations), len(input_amplitudes)
    )

    xor_corrcoef = {}
    for i in range(degrees + 1):
        C = np.corrcoef(
            capacities[:, :, i].reshape((-1, 1)),
            xor_kappas.reshape((-1, 1)),
            rowvar=False,
        )
        assert C.shape == (2, 2)
        xor_corrcoef[i] = C[0, 1]

    print("start narma calculation", flush=True)
    if multiprocess:
        narma_kappas = pool.starmap(
            get_narma_kappa_default, product(input_durations, input_amplitudes)
        )
    else:
        narma_kappas = []
        for d, a in product(input_durations, input_amplitudes):
            narma_kappas.append(get_narma_kappa_default(d, a))
    narma_kappas = np.array(narma_kappas).reshape(
        len(input_durations), len(input_amplitudes)
    )

    narma_corrcoef = {}
    for i in range(degrees + 1):
        C = np.corrcoef(
            capacities[:, :, i].reshape((-1, 1)),
            narma_kappas.reshape((-1, 1)),
            rowvar=False,
        )
        assert C.shape == (2, 2)
        narma_corrcoef[i] = C[0, 1]

    bar_width = 0.25
    lightness_fac = 0.7
    saturation_fac = 1.5
    base_colors = {
        "capacity": "#493657",
        "degree": "#73AB84",
        "delay": "#337ca0",
        "accent": "#DA4167",
        "XOR": "#F4E04D",  # '#F77F00',
        "XORXOR": "#042A2B",  # '#A30000',
    }

    fig = plt.figure(constrained_layout=True)
    axs = fig.subplot_mosaic(
        [
            ["total capacity", "correlation"],
            ["amplitude", "duration"],
        ],
    )
    ax = axs["total capacity"]
    capacity_mesh = ax.pcolormesh(
        input_amplitudes,
        input_durations,
        capacities[:, :, 0],
        vmin=0,
        vmax=64,
        cmap=sns.light_palette(base_colors["capacity"], as_cmap=True),
    )
    ax.set_xlabel("amplitude")
    ax.set_ylabel("duration")
    ax.spines["left"].set_color("none")
    capacity_cbar = fig.colorbar(capacity_mesh, ax=ax)
    ax.set_title("total capacity")

    ax = axs["correlation"]
    ax.set_ylim((-1, 1))
    shift_factor = 0
    w = bar_width - (bar_width / 2) * min(abs(shift_factor), 1)
    ax.bar(
        x=np.arange(degrees + 1) + ((w + bar_width) / 2 * shift_factor),
        height=xor_corrcoef.values(),
        width=w,
        color=base_colors["capacity"],
    )
    ax.set_xticks([0, 1, 2, 3])
    ax.set_xticklabels(["total", 1, 2, 3])
    ax.tick_params(axis="x", direction="in", pad=0, length=0.0)
    ax.set_title("correlation")

    ax = axs["amplitude"]
    print(capacities.shape, flush=True)
    slice_duration_index = 40
    ax.plot(
        input_amplitudes,
        capacities[slice_duration_index, :, 0],
        label="total",
        linestyle="-",
        color="k",
    )
    ax.plot(
        input_amplitudes,
        capacities[slice_duration_index, :, 1],
        label="deg=1",
        linestyle="--",
        color="k",
    )
    ax.plot(
        input_amplitudes,
        capacities[slice_duration_index, :, 2],
        label="deg=2",
        linestyle="-.",
        color="k",
    )
    ax.plot(
        input_amplitudes,
        capacities[slice_duration_index, :, 3],
        label="deg=3",
        linestyle=":",
        color="k",
    )
    ax2 = ax.twinx()
    ax2.plot(
        input_amplitudes,
        xor_kappas[slice_duration_index, :],
        label=r"$\kappa$",
        linestyle="-",
        color="r",
    )
    ax2.tick_params(axis="y", color="r", labelcolor="r")
    ax2.set_ylim(0, 1)
    ax2.spines["top"].set_color("none")
    ax2.spines["bottom"].set_color("none")
    ax2.spines["left"].set_color("none")
    ax2.spines["right"].set_color("r")
    ax2.set_yticklabels([])
    ax.set_ylabel("capacity")
    ax.set_ylim(-1, 64)

    ax = axs["duration"]
    slice_amplitude_index = 20
    ax.plot(
        input_durations,
        capacities[:, slice_amplitude_index, 0],
        linestyle="-",
        color="k",
    )
    ax.plot(
        input_durations,
        capacities[:, slice_amplitude_index, 1],
        linestyle="--",
        color="k",
    )
    ax.plot(
        input_durations,
        capacities[:, slice_amplitude_index, 2],
        linestyle="-.",
        color="k",
    )
    ax.plot(
        input_durations,
        capacities[:, slice_amplitude_index, 3],
        linestyle=":",
        color="k",
    )
    ax2 = ax.twinx()
    ax2.plot(
        input_durations, xor_kappas[:, slice_amplitude_index], linestyle="-", color="r"
    )
    ax2.tick_params(axis="y", color="r", labelcolor="r")
    ax2.set_ylim(0, 1)
    ax2.spines["top"].set_color("none")
    ax2.spines["bottom"].set_color("none")
    ax2.spines["left"].set_color("none")
    ax2.spines["right"].set_color("r")
    ax2.set_ylabel(r"$\kappa$", rotation=-90)
    ax.set_yticklabels([])
    ax.set_ylim(-1, 64)

    fig.legend(prop={"size": 6}, ncol=1, fancybox=True, loc="lower center")
    for name, ax in axs.items():
        ax.spines["top"].set_color("none")
        ax.spines["bottom"].set_color("none")
        ax.spines["right"].set_color("none")

    plt.savefig(
        f"/users/dick/corrcoef_input_dimension={in_dim}_input_width={in_width}_uniques={uniques}_alpha={alpha}_epsiolon={init_epsilon}.pdf"
    )


if __name__ == "__main__":
    for alpha, init_epsilon, in_width in product([0.25], [0.05], [64]):
        main(
            alpha=alpha,
            init_epsilon=init_epsilon,
            multiprocess=True,
            uniques=4,
            in_dim=1,
            in_width=in_width,
        )
