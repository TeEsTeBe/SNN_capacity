import os
import os.path as osp

import json

import numpy as np


import itertools as it


from .random import resetAllSeeds, resetNumpySeed, resetNestSeeds, getRandomSeed


LYAPUNOV_PATH = "lyapunov_spectrum.npy"
PERFORMANCE_PATH = "performance.json"
NMEM_PERFORMANCE_PATH = "nmem_performances.json"
ALL_PERFORMANCES_PATH = "all_performances.json"
FACTOR_LOCAL_LS_PATH = "factor_local_LS.npy"
DELTA_T_LOCAL_LS_PATH = "delta_t_local_LS.npy"
RESERVOIR_READOUT_PATH = "reservoir_readout.npy"
READOUT_TIMES_PATH = "readout_times.npy"
INPUT_SEQUENCE_PATH = "input_sequence.npy"
MEAN_ENERGY_PATH = "mean_energy.json"
RESCOMP_PARAMS_PATH = "rescomp_params.json"
NETWORK_STATISTICS_PATH = "network_statistics.json"


def get_mean_energy(runDatapath, energy_path=MEAN_ENERGY_PATH):
    try:
        energy_dict = load(runDatapath, energy_path)
        return energy_dict["mean_energy"]
    except FileNotFoundError:
        print("WARNING:", osp.join(runDatapath, energy_path), "not found")
        return np.nan


def periodic_actions(total_steps, i, threshold, step_func, action_func):
    while i + total_steps >= threshold:
        cur_steps = threshold - i
        step_func(cur_steps)
        action_func()
        total_steps -= cur_steps
        i = 0
    if total_steps > 0:
        step_func(total_steps)
        i += total_steps
    return i


def load(datapath, load_path=None):
    if load_path is None:
        datapath, load_path = osp.split(datapath)
    _, file_extension = osp.splitext(load_path)
    if file_extension == ".npy":
        return load_npy(datapath, load_path)
    elif file_extension == ".json":
        return load_json(datapath, load_path)
    else:
        raise NotImplementedError(
            "file extension {} not supported".format(file_extension)
        )
    return


def load_npy(datapath, numpy_path=""):
    return np.load(osp.join(datapath, numpy_path))


def load_json(datapath, json_path=""):
    with open(osp.join(datapath, json_path), "r") as f:
        return json.load(f)


def dump_json(fname, json_dict):
    with open(fname, "w") as f:

        json.dump(json_dict, f, indent=2)
    return


def get_run_datapath(experiment_path, param_label, run_number="next", exist_ok=False):
    assert osp.exists(experiment_path)
    curDatapath = osp.join(experiment_path, param_label)
    os.makedirs(curDatapath, exist_ok=True)
    if run_number == "next":
        run_number = 0
        while osp.exists(osp.join(curDatapath, "Run{}".format(run_number))):
            run_number += 1
    runDatapath = osp.join(curDatapath, "Run{}".format(run_number))
    os.makedirs(runDatapath, exist_ok=exist_ok)
    return runDatapath


def get_run_dir(path, run_number="next", exist_ok=False):
    if run_number == "next":
        run_number = 0
        while osp.exists(osp.join(path, "Run{}".format(run_number))):
            run_number += 1
    runDatapath = osp.join(path, "Run{}".format(run_number))
    os.makedirs(runDatapath, exist_ok=exist_ok)
    return runDatapath
