import os
import os.path as osp

import argparse
import json
import numpy as np
import numpy.random as rnd
import numpy.linalg as la
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


from FPUT.Simulation import Simulation, DampedSimulation, FPUSkipTransientAndPlot


import FPUT.utils


def _rescomp_initconds_analysis(
    init_seed=200, jitter=0.01, init_epsilon=0.05, **add_simparams
):
    print("Starting ResComp analysis...")

    jitter_seed = utils.getRandomSeed()
    simparams = dict(
        init_epsilon=init_epsilon,
        jitter=jitter,
        init_seed=init_seed,
        jitter_seed=jitter_seed,
        **add_simparams
    )

    cur_datapath = DATAPATH
    runDatapath = utils.get_run_datapath(
        cur_datapath,
        "{}epsilon{}InitSeed{}Jitter".format(
            np.round(init_epsilon, 5), init_seed, np.round(jitter, 3)
        ),
    )

    utils.dump_json(osp.join(runDatapath, "simparams.json"), simparams)

    cur_sim = Simulation(**simparams)
    cur_sim.startReadout()
    cur_sim.simulate_steps(10 ** 6)

    cur_sim.save_readout(runDatapath)
    return


def analyse(init_epsilon, input_duration=5.0, **add_simparams):
    input_seed = 88111111
    init_seed = utils.getRandomSeed()
    init_Q_seed = utils.getRandomSeed()
    simparams = dict(
        init_epsilon=init_epsilon,
        input_seed=input_seed,
        input_duration=input_duration,
        init_seed=init_seed,
        **add_simparams
    )

    runDatapath = utils.get_run_datapath(DATAPATH, "{}InitEpsilon".format(init_epsilon))

    utils.dump_json(
        osp.join(runDatapath, "seeding.json"),
        {"input_seed": input_seed, "init_seed": init_seed, "init_Q_seed": init_Q_seed},
    )

    print("Starting perturbation analysis...")

    print("Lyapunov_exponents:", LS)

    return


def _skip_transient_and_plot(datapath, **simparams):

    transient = FPUSkipTransientAndPlot(**simparams)
    try:
        transient.get_to_equilibrium(save=datapath)
    except ValueError:
        print(
            "Well, better luck next time I guess. (This is likely caused by runaway trajectories.)"
        )

    try:
        E_dict, comp_E_dict = transient.make_plots(save=datapath)
    except ValueError:
        print(
            "Well, better luck next time I guess. (This is likely caused by runaway trajectories.)"
        )

    return


def analyse_damped(
    alpha=0.25,
    input_duration=1000,
    tau_relax=100,
    input_amplitude=0.005,
    runDatapath=None,
    **add_simparams
):
    input_seed = utils.getRandomSeed()
    init_seed = utils.getRandomSeed()
    init_Q_seed = utils.getRandomSeed()
    simparams = dict(
        init_epsilon=init_epsilon,
        input_seed=input_seed,
        alpha=alpha,
        tau_relax=tau_relax,
        input_amplitude=input_amplitude,
        input_duration=input_duration,
        init_seed=init_seed,
        **add_simparams
    )

    if runDatapath is None:
        runDatapath = utils.get_run_datapath(
            DATAPATH,
            "{}tauRelax{}amplitude{}Duration".format(
                np.round(tau_relax, 1),
                np.round(input_amplitude, 4),
                int(input_duration),
            ),
        )

    utils.dump_json(
        osp.join(runDatapath, "seeding.json"),
        {"input_seed": input_seed, "init_seed": init_seed, "init_Q_seed": init_Q_seed},
    )
    utils.dump_json(osp.join(runDatapath, "simparams.json"), simparams)

    _rescomp_analysis(runDatapath, **simparams)

    return


def initial_transient_damped(
    alpha=0.25,
    input_duration=1000,
    tau_relax=100,
    input_amplitude=0.005,
    **add_simparams
):
    input_seed = utils.getRandomSeed()
    init_seed = utils.getRandomSeed()
    init_Q_seed = utils.getRandomSeed()
    simparams = dict(
        init_epsilon=init_epsilon,
        input_seed=input_seed,
        alpha=alpha,
        tau_relax=tau_relax,
        input_amplitude=input_amplitude,
        input_duration=input_duration,
        init_seed=init_seed,
        **add_simparams
    )

    runDatapath = utils.get_run_datapath(
        DATAPATH,
        "{}tauRelax{}amplitude{}Duration".format(
            np.round(tau_relax, 1), np.round(input_amplitude, 4), int(input_duration)
        ),
    )

    utils.dump_json(
        osp.join(runDatapath, "seeding.json"),
        {"input_seed": input_seed, "init_seed": init_seed, "init_Q_seed": init_Q_seed},
    )
    utils.dump_json(osp.join(runDatapath, "simparams.json"), simparams)

    _skip_transient_and_plot(runDatapath, **simparams)

    return


def analyse_damped_after_transient(runDatapath, n_readouts_per_batch=1):
    simparams = utils.load(runDatapath, "simparams.json")
    seeds = utils.load(runDatapath, "seeding.json")
    init_Q_seed = seeds["init_Q_seed"]
    init_state = utils.load(runDatapath, "equilibrium_state.npy")

    simparams["init_state"] = init_state

    utils.dump_json(
        osp.join(runDatapath, "rescomp_params.json"),
        {"n_readouts_per_batch": n_readouts_per_batch},
    )

    _rescomp_analysis(runDatapath, n_readouts_per_batch, **simparams)

    return


def param_scan(
    init_eps_list=[0.1 / 16, 0.1 / 8, 0.1 / 4, 0.1 / 2, 0.1],
    input_dur_list=[2.5, 5.0, 10.0, 20.0, 40.0],
):
    for init_epsilon in init_eps_list:
        for input_duration in input_dur_list:
            analyse(init_epsilon=init_epsilon, input_duration=input_duration)
    return


def param_scan_initconds(
    init_eps_list=[0.1 / 8, 0.1 / 4, 0.1 / 2, 0.1], jitter_list=[0.5, 0.3, 0.1]
):
    for init_epsilon in init_eps_list:
        for jitter in jitter_list:
            for i in range(nbr_runs):
                _rescomp_initconds_analysis(200, jitter, init_epsilon)
                _rescomp_initconds_analysis(454, jitter, init_epsilon)
    return


def param_scan_rescomp_damped(
    alpha_list=[0.0, 0.125, 0.25, 0.375, 0.5],
    input_dur_list=[100, 1000, 10000],
    tau_relax_list=[100],
    input_amplitude_list=[0.005],
):
    for alpha in alpha_list:
        for input_duration in input_dur_list:
            for tau_relax in tau_relax_list:
                for input_amplitude in input_amplitude_list:
                    for i in range(nbr_runs):
                        try:
                            analyse_damped(
                                alpha,
                                input_duration,
                                tau_relax=tau_relax,
                                input_amplitude=input_amplitude,
                            )
                        except ValueError:
                            print(
                                "Well, better luck next time I guess. (This is likely caused by runaway trajectories.)"
                            )
    return


def param_scan_transient_damped(
    alpha_list=[0.0, 0.125, 0.25, 0.375, 0.5],
    input_dur_list=[100, 1000, 10000],
    tau_relax_list=[100],
    input_amplitude_list=[0.005],
    N=32,
):
    for alpha in alpha_list:
        for input_duration in input_dur_list:
            for tau_relax in tau_relax_list:
                for input_amplitude in input_amplitude_list:
                    for i in range(nbr_runs):
                        try:
                            initial_transient_damped(
                                alpha,
                                input_duration,
                                tau_relax=tau_relax,
                                input_amplitude=input_amplitude,
                                N=N,
                            )
                        except ValueError:
                            print(
                                "Well, better luck next time I guess. (This is likely caused by runaway trajectories.)"
                            )
    return


if __name__ == "__main__":
    pass
