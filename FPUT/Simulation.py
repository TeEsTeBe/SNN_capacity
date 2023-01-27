import os.path as osp

import numpy as np
import numpy.random as rnd
import matplotlib.pyplot as plt

from FPUT.Sim_util import simulate_steps as simulate_steps_cython
from FPUT.Sim_util import simulate_steps_damped as simulate_steps_damped_cython
from FPUT.Sim_util import (
    simulate_steps_damped_multi_input as simulate_steps_damped_multi_input_cython,
)
import FPUT.utils as utils

# OBSOLETE
# DEBUG = False # can be set upon import to activate DEBUG option

# ATTENTION! This file includes three classes:
# - FPUSkipTransientAndPlot: a basic helper class to discard transient behaviour
# - Simulation: simulation class for the original, non-modified FPU alpha model
# - DampedSimulation: simulation class for the damped, input-driven FPU alpha model


"""
A helper class to skip the initial transients and save the equilibrium state and to facilitate fast support for testing within jupyter notebooks.
"""


class FPUSkipTransientAndPlot:
    def __init__(self, **simparams):
        simparams["init_epsilon"] = 0.05
        simparams["input_seed"] = utils.getRandomSeed()
        self.simulation = DampedSimulation(**simparams)
        simparams["init_epsilon"] = 0.0
        simparams["input_seed"] = utils.getRandomSeed()
        self.comp_simulation = DampedSimulation(**simparams)
        self.step_size = 1000
        self.s = 0
        self.histogram_vals = 1000
        self.readout_every = 7
        return

    def _for_all_sims(self, sim_func, *args, **kwargs):
        sim_func(self.simulation, *args, **kwargs)
        sim_func(self.comp_simulation, *args, **kwargs)
        return

    def _simulate_batches(self, nbr_batches):
        self._for_all_sims(DampedSimulation.simulate_batches, nbr_batches)
        return

    def _simulate_steps(self, nbr_steps):
        self._for_all_sims(DampedSimulation.simulate_steps, nbr_steps)
        return

    def get_to_equilibrium(self, save=None):
        while (
            self.simulation.getCurTotalEnergy()
            > self.comp_simulation.getCurTotalEnergy()
        ):
            self._simulate_steps(self.step_size)
            self.s += 1
            print("{} steps done".format(self.s))
        self._simulate_steps(
            2 * self.s * self.step_size
        )  # simulate double time after energies first cross
        if save:
            np.save(
                osp.join(save, "equilibrium_state.npy"), self.simulation.getCurState()
            )
        return self.simulation.getCurState()

    def _get_energy_readouts(self, cur):
        t_vec = np.array(cur.ts)
        # x_vec, v_vec = np.array(cur.readout)[:, :cur.N], np.array(cur.readout)[:, cur.N:]
        E_vec = np.array(cur.E_n_readout)
        E_anh_vec = np.array(cur.E_anharmonic_readout)
        E_ext_vec = np.array(cur.E_external_readout)
        E_tot_vec = np.sum(E_vec, axis=1) + E_anh_vec
        return dict(
            t_vec=t_vec,
            E_vec=E_vec,
            E_anh_vec=E_anh_vec,
            E_ext_vec=E_ext_vec,
            E_tot_vec=E_tot_vec,
        )

    def make_plots(self, save=None):
        self._for_all_sims(
            DampedSimulation.startReadout,
            readout_every=self.readout_every,
            readout_energies=True,
        )
        self._simulate_steps(self.readout_every * self.histogram_vals)
        if save is not None:
            self.simulation.save_energies(save)
        self.simulation.energies_plot(save)
        self.simulation.plot_energy_distribution(save)
        print("lets gather readouts")
        E_dict = self._get_energy_readouts(self.simulation)
        comp_E_dict = self._get_energy_readouts(self.comp_simulation)

        plt.figure()
        plt.hist(E_dict["E_tot_vec"], density=True)
        E_mean = np.mean(E_dict["E_tot_vec"])
        plt.axvline(
            E_mean,
            c="black",
            linestyle="dashed",
            label="$\\bar{{E}} = {}$".format(np.round(E_mean, 6)),
        )
        plt.hist(comp_E_dict["E_tot_vec"], density=True, alpha=0.3)
        plt.legend()

        if save:
            plt.savefig(osp.join(save, "energy_distribution_both.pdf"))
        else:
            plt.show()
        return E_dict, comp_E_dict


"""
Handles the simulation of the original FPUT model. No Damping, no Input.
"""


class Simulation:
    def __init__(
        self, init_state=None, init_epsilon=0.05, init_t=0, jitter=0, **params
    ):
        """
        Initializes all parameters of the simulation and sets its initial state.
        :param init_state: A specific initial state vector.
        :param init_epsilon: The energy per oscillator in the initial state. Draws one configuration uniformly from the constant kinetic energy surface (positions are fixed at 0). This argument is ignored, if init_state is provided instead.
        :param init_t: t_0, the internal clock value at the initial configuration.
        :param jitter: Specifies the amount of jitter to add to the initial state vector (keeps the state vector on the same energy). Between 0 and 1.
        :param params: Additional parameters to be set upon initialization. See "_setParams".
        """
        self._setParams(**params)
        if init_state is None:
            init_state = self._create_random_momentum_state(
                init_epsilon, seed=self.init_seed
            )
        self.setCurState(init_state)
        self.setCurTime(init_t)
        if jitter > 0:
            self._add_momentum_jitter(jitter)
        self._init_readout_utils()
        return

    def _init_readout_utils(self):
        """
        Initializes the attributes that are needed for readout.
        :return: None
        """
        self.readout_started = False
        self.readout_every = None
        self.s = None
        self.ts = None
        self.readout = None
        self.E_n_readout = None
        self.E_anharmonic_readout = None
        return

    def _create_random_momentum_state(self, epsilon, seed):
        """
        Creates a state vector with all positions of the oscillators set to zero, and a momentum vector drawn from the constant energy surface defined by epsilon.
        :param epsilon: The energy per oscillator in the chain.
        :param seed: The random seed to be used for initialization.
        :return: Configuration: position vector is zeros, momentum vector from constant energy surface.
        """
        rnd.seed(seed)
        cur_state = rnd.normal(size=self.N * 2)  # initial positions and velocities
        cur_state[: self.N] = 0  # initialize positions with zero
        cur_state = self._scale_v_to_kin_energy(
            cur_state, epsilon * self.N
        )  # satisfy average modal energy E/N=epsilon
        # cur_state *= np.sqrt(2*epsilon*self.N)/np.linalg.norm(cur_state)  # satisfy average modal energy E/N=epsilon
        return cur_state

    # TODO: as of now, only puts on hypercut of sphere within kinetic energy sphere, not of total energy
    def _add_momentum_jitter(self, magnitude=0.1):
        """
        Adds Jitter to the momentum vector.
        :param magnitude: The magnitude of the jitter. Between 0 and 1.
        :return: None. [In-place operation]
        """
        before_energy = self.getCurKinEnergy()
        print(before_energy)
        # self.v *= np.sqrt(1 - magnitude)
        self.v = self._scale_v_to_kin_energy(self.v, before_energy * (1 - magnitude))
        # v_energy = self.getCurKinEnergy()
        energy_jitter = before_energy * magnitude
        jitter_v = self._create_random_momentum_state(
            energy_jitter / self.N, seed=self.jitter_seed
        )[self.N :]
        # DONE: orthogonalize jitter_v to self.v, then normalize to have energy_jitter again
        jitter_v -= np.dot(self.v, jitter_v) / (np.linalg.norm(self.v) ** 2) * self.v
        jitter_v = self._scale_v_to_kin_energy(jitter_v, energy_jitter)
        self.v += jitter_v
        assert np.isclose(self.getCurKinEnergy(), before_energy)
        return

    def _scale_v_to_kin_energy(self, v, kin_energy):
        """
        Scales the momentum vector to match the specified kinetic energy of the system. Does not change its direction.
        :param v: Momentum vector.
        :param kin_energy: Kinetic energy.
        :return: Scaled momentum vector.
        """
        v *= np.sqrt(2 * kin_energy) / np.linalg.norm(v)
        return v

    def _init_readout(self, readout_every):
        """
        Sets up the readout variables, such that readout can begin. See "_readout_now".
        :param readout_every: The time between two readout events in units of h (the simulation step size).
        :return: None
        """
        self.readout_started = True
        self.readout_every = readout_every
        self.ts = []
        self.readout = []
        self.E_n_readout = []
        self.E_anharmonic_readout = []
        self.s = 0
        return

    def startReadout(self, readout_every=100, readout_energies=False, readout_now=True):
        """
        Starts the readout. Reads out the current state every 'readout_every' simulation steps.
        :param readout_every: The time between two readout events in units of h (the simulation step size).
        :param readout_energies: Boolean, whether energies should be logged as well.
        :param readout_now: Whether the state should be read out immediately, or first after 'readout_every' steps.
        :return: None
        """
        self._init_readout(readout_every)
        self.readout_energies = readout_energies
        if readout_now:
            self._readout_now()
        return

    def set_readout_energies(self, readout_energies):
        self.readout_energies = readout_energies
        return

    def setCurState(self, state):
        """
        Set the current configuration.
        :param state: State vector.
        :return: None
        """
        self.x = np.copy(state[: self.N])
        self.v = np.copy(state[self.N :])
        return

    def getCurState(self):
        """
        Get the current configuration.
        :return: Current state vector.
        """
        return np.concatenate((self.x, self.v))

    def getCurModalEnergies(self):
        """
        Computes and returns the energies per eigenmode of the harmonic system.
        :return: Modal energies vector.
        """
        cur_energies = np.zeros(self.N)
        for i in range(self.N):
            cur_energies[i] = self.E_k(i, self.x, self.v)
        return cur_energies

    def getCurKinEnergy(self):
        """
        Get the current kinetic energy.
        :return: Current kinetic energy.
        """
        return np.sum(self.v ** 2) / 2

    def _get_diff_x(self):
        bound_x = np.insert(self.x, (0, self.N), 0)
        return bound_x[1:] - bound_x[:-1]

    # DONE: Implement this with hamiltonian formula
    def getCurPotEnergy(self):
        """
        Get the current potential energy.
        :return: Current potential energy.
        """
        diff_x = self._get_diff_x()
        return self.getCurHarmonicPotEnergy(diff_x) + self.getCurAnharmonicPotEnergy(
            diff_x
        )

    def getCurHarmonicPotEnergy(self, diff_x=None):
        """
        Get the current potential energy stored in the harmonic interactions.
        :param diff_x: Unused. [Only used to speed up internal computations.]
        :return: Harmonic part of the current potential energy.
        """
        if diff_x is None:
            diff_x = self._get_diff_x()
        return np.sum(diff_x ** 2) / 2

    def getCurAnharmonicPotEnergy(self, diff_x=None):
        """
        Get the current potential energy stored in the anharmonic interactions.
        :param diff_x: Unused. [Only used to speed up internal computations.]
        :return: Anharmonic part of the current potential energy.
        """
        if diff_x is None:
            diff_x = self._get_diff_x()
        return self.alpha / 3 * np.sum(diff_x ** 3)

    def getCurTotalEnergy(self):
        """
        Get the current total energy of the system.
        :return: Current total energy.
        """
        return self.getCurKinEnergy() + self.getCurPotEnergy()

    def setCurTime(self, t):
        """
        Set the internal clock of the system.
        :param t: Current time.
        :return: None
        """
        self.t = t
        return

    def getCurTime(self):
        """
        Get the current value of the internal clock.
        :return: Current time.
        """
        return self.t

    def _setParams(
        self, N=32, h=0.1, alpha=0.25, init_seed=200, jitter_seed=9999444, **kwargs
    ):
        """
        Parameters to be used for the simulation.
        :param N: The number of oscillators in the chain.
        :param h: The step size of one numerical integration step in units of time.
        :param alpha: The strength of the cubic potential.
        :param init_seed: The random seed to be used for the creation of the initial state from a constant energy surface.
        :param jitter_seed: The random seed to be used for the jitter that is added to the initial state.
        :return: None
        """
        self.init_seed = init_seed
        self.jitter_seed = jitter_seed
        self.N = N
        self.alpha = alpha
        self.h = h
        # define coefficients for symplectic integration
        self.K = 4
        self.c_k = np.zeros(self.K)
        self.d_k = np.zeros(self.K)
        self.c_k[0] = 1.0 / (2.0 * (2.0 - 2 ** (1 / 3)))
        self.c_k[3] = self.c_k[0]
        self.c_k[1] = (1.0 - 2 ** (1 / 3)) / (2.0 * (2.0 - 2 ** (1 / 3)))
        self.c_k[2] = self.c_k[1]
        self.d_k[0] = 1.0 / (2.0 - 2 ** (1 / 3))
        self.d_k[2] = self.d_k[0]
        self.d_k[1] = -(2 ** (1 / 3)) / (2.0 - 2 ** (1 / 3))
        self.d_k[3] = 0
        return

    def _simulate_steps_no_readout(self, nbr_steps):
        for _ in range(nbr_steps):
            for j in range(self.K):
                for i in range(self.N):
                    self.x[i] += self.c_k[j] * self.h * self.v[i]
                for i in range(self.N):
                    self.v[i] += self.d_k[j] * self.h * self._a(self.x, i)
            self.t += self.h
        # self.s += nbr_steps
        return

    def _simulate_steps_cython(self, nbr_steps):
        self.t, self.x, self.v = simulate_steps_cython(
            self.t, nbr_steps, self.x, self.v, self.h, self.N, self.alpha
        )
        self.x = np.array(self.x)
        self.v = np.array(self.v)
        # self.s += nbr_steps
        return

    def numpify_readout(self):
        """
        Put the energy measurements in numpy arrays and return them.
        :return: Dictionary, consisting of: measurement times, modal energies, anharmonic potential energies.
        """
        t_vec = np.array(self.ts)
        E_vec = np.array(self.E_n_readout)
        E_anh_vec = np.array(self.E_anharmonic_readout)
        return dict(t_vec=t_vec, E_vec=E_vec, E_anh_vec=E_anh_vec)

    def save_energies(self, datapath):
        """
        Save energy measurements to the given path.
        :param datapath: Path to store the energy measurements in.
        :return: None
        """
        E_dict = self.numpify_readout()
        t_vec = E_dict["t_vec"]
        E_vec = E_dict["E_vec"]
        E_anh_vec = E_dict["E_anh_vec"]
        np.save(osp.join(datapath, "energies_ts.npy"), t_vec)
        np.save(osp.join(datapath, "energies_E_n.npy"), E_vec)
        np.save(osp.join(datapath, "energies_E_anh.npy"), E_anh_vec)
        return

    def energies_plot(self, datapath=None, n_modes=4):
        """
        Make a plot of the total energy, modal energies and anharmonic potential over time. Only works if energies were included in the readout.
        :param datapath: Optional path to store the figure to.
        :param n_modes: Amount of eigenmodes to plot.
        :return: None
        """
        E_dict = self.numpify_readout()
        t_vec = E_dict["t_vec"]
        E_vec = E_dict["E_vec"]
        E_anh_vec = E_dict["E_anh_vec"]

        plt.figure(figsize=(15, 10))
        plt.plot(t_vec, np.sum(E_vec, axis=1) + E_anh_vec, c="red", label="energy")
        plt.plot(t_vec, E_anh_vec, c="black", label="anharmonic potential")
        # if self.N == 32 and self.alpha == 0.25:
        #    plt.axhline(10 ** (-2) * self.N, c="black", label="stochasticity threshold")
        for i in range(n_modes):
            # plt.plot(moving_average(t_vec), moving_average(E_vec[:, i]), label="mode {}".format(i))
            plt.plot(t_vec, E_vec[:, i], label="mode {}".format(i + 1))
        plt.xlabel("t")
        plt.ylabel("E")
        plt.legend()
        if datapath:
            plt.savefig(osp.join(datapath, "energies_plot.pdf"))
        else:
            plt.show()
        return

    def plot_energy_distribution(self, datapath=None):
        """
        Plot the distribution of total energy over time. Only works if energies were included in the readout.
        :param datapath: Optional path to store the figure to.
        :return: None
        """
        E_dict = self.numpify_readout()
        t_vec = E_dict["t_vec"]
        E_vec = E_dict["E_vec"]
        E_anh_vec = E_dict["E_anh_vec"]
        E_tot_vec = np.sum(E_vec, axis=1) + E_anh_vec

        plt.figure()
        plt.hist(E_tot_vec, density=True)
        E_mean = np.mean(E_tot_vec)
        plt.axvline(
            E_mean,
            c="black",
            linestyle="dashed",
            label="$\\bar{{E}} = {}$".format(np.round(E_mean, 6)),
        )
        plt.legend()
        if datapath:
            plt.savefig(osp.join(datapath, "energy_histogram.pdf"))
        else:
            plt.show()
        return

    def _readout_now(self):
        """
        Readout the current state of the system and append it to the readout vectors.
        :return: None
        """
        self.ts.append(self.getCurTime())
        self.readout.append(self.getCurState())
        if self.readout_energies:
            self.E_n_readout.append(self.getCurModalEnergies())
            self.E_anharmonic_readout.append(self.getCurAnharmonicPotEnergy())
            # self.getCurTotalEnergy()
        self.s = 0
        return

    def save_readout(self, datapath):
        """
        Save the state readouts to the given path.
        :param datapath: Path to store the readouts in.
        :return: None
        """
        assert self.readout is not None
        np.save(osp.join(datapath, "readout_times.npy"), np.array(self.ts))
        np.save(osp.join(datapath, "reservoir_readout.npy"), np.array(self.readout))
        return

    def checkpoint_state(self, datapath, label_prefix=""):
        """
        Checkpoint the current state of the system. Overwrites existing files.
        CAUTION: does not store previous readouts.
        :param datapath: Path to store the current state in.
        :param label_prefix: Optional label to prepend to the filename.
        :return: None
        """
        np.save(
            osp.join(datapath, label_prefix + "current_state.npy"), self.getCurState()
        )
        return

    def _simulate_steps_with_func(self, sim_func, nbr_steps):
        if not self.readout_started:
            sim_func(nbr_steps)
        else:
            self.s = utils.periodic_actions(
                nbr_steps,
                self.s,
                self.readout_every,
                step_func=sim_func,
                action_func=self._readout_now,
            )
            # assert self.s == test_correct_cnt
        return

    def simulate_steps_no_cython(self, nbr_steps):
        """
        Use the python implementation to simulate the given number of numerical integration steps.
        :param nbr_steps: Number of numerical integration steps to simulate.
        :return: None
        """
        return self._simulate_steps_with_func(
            self._simulate_steps_no_readout, nbr_steps
        )

    def simulate_steps(self, nbr_steps):
        """
        Use the cython implementation to simulate the given number of numerical integration steps. Requires the compiled cython script.
        :param nbr_steps: Number of numerical integration steps to simulate.
        :return: None
        """
        return self._simulate_steps_with_func(self._simulate_steps_cython, nbr_steps)

    def _a(self, x, i):
        """
        Python implementation of the acceleration acting on particle 'i'.
        :param x: Current position vector.
        :param i: Particle for which to calculate the current acceleration.
        :return: Current acceleration of particle 'i'.
        """
        if i == 0:
            return (
                x[i + 1] - 2 * x[i] + self.alpha * ((x[i + 1] - x[i]) ** 2 - x[i] ** 2)
            )
        elif i == self.N - 1:
            return (
                x[i - 1] - 2 * x[i] + self.alpha * (x[i] ** 2 - (x[i] - x[i - 1]) ** 2)
            )
        else:
            return (
                x[i + 1]
                + x[i - 1]
                - 2 * x[i]
                + self.alpha * ((x[i + 1] - x[i]) ** 2 - (x[i] - x[i - 1]) ** 2)
            )
        return

    # define normal modes
    def Q_k(self, k, x):
        Q_temp = 0
        for i in range(1, self.N + 1):
            Q_temp += (
                np.sqrt(2 / (self.N + 1))
                * x[i - 1]
                * np.sin(np.pi * (k + 1) * i / (self.N + 1))
            )
            # Q_temp += x[i-1]*np.sin(np.pi*(k+1)*i/(self.N+1))
        return Q_temp

    def w_k(self, k):
        # return 2*np.sin(np.pi*(k+1)/(2*self.N))
        return 2 * np.sin(np.pi * (k + 1) / (2 * (self.N + 1)))

    def E_k(self, k, x, p):
        return (self.Q_k(k, p) ** 2 + self.w_k(k) ** 2 * self.Q_k(k, x) ** 2) / 2

    # fourier back transform
    # same function as Q_k
    def xb_i(self, i, Q):
        x_tmp = 0
        for k in range(1, self.N + 1):
            x_tmp += (
                np.sqrt(2 / (self.N + 1))
                * Q[k - 1]
                * np.sin(np.pi * k * (i + 1) / (self.N + 1))
            )
        return x_tmp


"""
Handles the simulation of the damped, input-driven FPUT system. Yes, with input_amplitude=0.0 and gamma=0.0 or tau_relax=np.inf,
the simulation will be equivalent to the original FPUT system (but computationally more costly and return values might be different).
"""


class DampedSimulation(Simulation):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.input_seq = self._init_input_seq()
        return

    def _init_readout_utils(self):
        self.readout_started = False
        self.readout_every = None
        self.s = None
        self.ts = None
        self.readout = None
        self.E_n_readout = None
        self.E_anharmonic_readout = None
        self.E_external_readout = None
        return

    def _init_input_seq(self):
        rnd.seed(self.input_seed)
        input_seq = rnd.randint(
            2, size=10000000, dtype=np.dtype("i")
        )  # just make enough
        self.input_0_steps_done = 0
        return input_seq

    def _setParams(
        self,
        *args,
        gamma=None,
        tau_relax=None,
        input_seed=4444777,
        input_duration=100,
        input_amplitude=0.005,
        **kwargs
    ):
        """
        Parameters to be used for the damped, input-driven simulation.
        :param args: Passed to base class Simulation.
        :param gamma: Obsolete. Use tau_relax instead.
        :param tau_relax: The damping time scale. p' = - 1/tau_relax * p + ...
        :param input_seed: Random seed used to create the input sequence.
        :param input_duration: The duration of one input signal.
        :param input_amplitude: The amplitude of the input signal.
        :param kwargs: Passed to base class Simulation.
        :return: None
        """
        super()._setParams(*args, **kwargs)
        if tau_relax is not None:
            assert gamma is None
            gamma = 1 / tau_relax
        elif gamma is not None:
            assert tau_relax is None
            tau_relax = 1 / gamma
        elif gamma is None and tau_relax is None:
            gamma = 0.01  # default gamma
            tau_relax = 100  # default tau_relax
        self.gamma = gamma  # gamma still here for backwards compatibility
        self.tau_relax = tau_relax
        self.input_seed = input_seed
        self.input_duration = input_duration  # in units of self.h
        self.input_amplitude = input_amplitude
        return

    def _init_readout(self, readout_every):
        self.readout_started = True
        self.readout_every = readout_every
        self.ts = []
        self.readout = []
        self.E_n_readout = []
        self.E_anharmonic_readout = []
        self.E_external_readout = []
        self.s = 0
        return

    def startReadout(
        self,
        readout_every="stim_offset",
        n_readouts=1,
        readout_energies=False,
        readout_now=False,
    ):
        """
        Starts the readout. Reads out the current state every 'readout_every' simulation steps.
        :param readout_every: The time between two readout events in units of h (the simulation step size).
        If set to "stim_offset", it is set to sample each time one symbol of the input sequence was processed.
        Starts after the next batch if one input symbol was just processed.
        :param n_readouts: Only used if "stim_offset" mode selected. The number of readouts to be done during one stimulus.
        :param readout_energies: Boolean, whether energies should be logged as well.
        :param readout_now: Whether the state should be read out immediately, or first after 'readout_every' steps.
        If "stim_offset" mode selected, it is usually advised to sample first AFTER the next batch started being processed
        :return: None
        """
        self._init_readout(readout_every)
        self.readout_energies = readout_energies
        if self.readout_every == "stim_offset":
            self.readout_every = self.input_duration // n_readouts
            assert (
                self.readout_every * n_readouts == self.input_duration
            )  # make sure that input_duration is divisible by readout_every, so that no offsets will be produced
            self.s = self.input_0_steps_done
            assert self.s == 0  # usually this should hold True in almost all use cases
        if readout_now:
            self._readout_now()
        return

    def _readout_now(self):
        self.ts.append(self.getCurTime())
        self.readout.append(self.getCurState())
        if self.readout_energies:
            self.E_n_readout.append(self.getCurModalEnergies())
            self.E_anharmonic_readout.append(self.getCurAnharmonicPotEnergy())
            self.E_external_readout.append(self.getCurExternalEnergy())
            # self.getCurTotalEnergy()
        self.s = 0
        return

    def numpify_readout(self):
        """
        Put the energy measurements in numpy arrays and return them.
        :return: Dictionary, consisting of: measurement times, modal energies, anharmonic potential energies and external potential energies.
        """
        t_vec = np.array(self.ts)
        E_vec = np.array(self.E_n_readout)
        E_anh_vec = np.array(self.E_anharmonic_readout)
        E_ext_vec = np.array(self.E_external_readout)
        return dict(t_vec=t_vec, E_vec=E_vec, E_anh_vec=E_anh_vec, E_ext_vec=E_ext_vec)

    def save_energies(self, datapath):
        """
        Save energy measurements to the given path.
        :param datapath: Path to store the energy measurements in.
        :return: None
        """
        E_dict = self.numpify_readout()
        t_vec = E_dict["t_vec"]
        E_vec = E_dict["E_vec"]
        E_anh_vec = E_dict["E_anh_vec"]
        E_ext_vec = E_dict["E_ext_vec"]
        np.save(osp.join(datapath, "energies_ts.npy"), t_vec)
        np.save(osp.join(datapath, "energies_E_n.npy"), E_vec)
        np.save(osp.join(datapath, "energies_E_anh.npy"), E_anh_vec)
        np.save(osp.join(datapath, "energies_E_ext.npy"), E_ext_vec)
        return

    def energies_plot(self, datapath=None, n_modes=4):
        """
        Make a plot of the total energy, modal energies, anharmonic potential and external potential over time.
        Only works if energies were included in the readout.
        :param datapath: Optional path to store the figure to.
        :param n_modes: Amount of eigenmodes to plot.
        :return: None
        """
        E_dict = self.numpify_readout()
        t_vec = E_dict["t_vec"]
        E_vec = E_dict["E_vec"]
        E_anh_vec = E_dict["E_anh_vec"]
        E_ext_vec = E_dict["E_ext_vec"]

        plt.figure(figsize=(15, 10))
        plt.plot(t_vec, np.sum(E_vec, axis=1) + E_anh_vec, c="red", label="energy")
        plt.plot(
            t_vec,
            np.sum(E_vec, axis=1) + E_anh_vec + E_ext_vec,
            c="red",
            linestyle="none",
            marker=".",
            label="energy + external",
        )
        plt.plot(t_vec, E_anh_vec, c="black", label="anharmonic potential")
        plt.plot(
            t_vec,
            E_ext_vec,
            c="green",
            linestyle="none",
            marker=".",
            label="external potential",
        )
        # if self.N == 32 and self.alpha == 0.25:
        #    plt.axhline(10 ** (-2) * self.N, c="black", label="stochasticity threshold")
        for i in range(n_modes):
            # plt.plot(moving_average(t_vec), moving_average(E_vec[:, i]), label="mode {}".format(i))
            plt.plot(t_vec, E_vec[:, i], label="mode {}".format(i + 1))
        plt.xlabel("t")
        plt.ylabel("E")
        plt.legend()
        if datapath:
            plt.savefig(osp.join(datapath, "energies_plot.pdf"))
        else:
            plt.show()
        return

    def getBatchDuration(self):
        """
        Get the input duration in units of internal time.
        :return: Input duration in units of time.
        """
        return self.h * self.input_duration  # unit of time

    def setInputSeq(self, input_seq, input_0_steps_done=0):
        """
        Set the input sequence manually.
        :param input_seq: The input sequence to use for the subsequent simulation.
        :param input_0_steps_done: The number of steps the first element of "input_seq" is believed to be processed already.
        The first element of "input_seq" will be processed for a number of simulation steps given by "self.input_duration" - input_0_steps_done.
        All subsequent elements will be processed for "self.input_duration" number of simulation steps.
        :return: None
        """
        self.input_seq = np.copy(input_seq)
        self.input_0_steps_done = input_0_steps_done
        return

    def _a(self, x, v, i, cur_input):
        return (
            -v[i] / self.tau_relax + self._give_input(i, cur_input) + super()._a(x, i)
        )

    def _next_input(self):
        self.input_seq = self.input_seq[1:]
        assert len(self.input_seq) > 0
        self.input_0_steps_done = 0
        return

    def _simulate_steps_no_readout(self, nbr_steps):
        test_correct_cnt = utils.periodic_actions(
            nbr_steps,
            self.input_0_steps_done,
            self.input_duration,
            step_func=self._simulate_steps_no_readout_one_input,
            action_func=self._next_input,
        )
        assert self.input_0_steps_done == test_correct_cnt
        return

    def _simulate_steps_no_readout_one_input(self, nbr_steps):
        cur_input = self.input_seq[0]
        for _ in range(nbr_steps):
            for j in range(self.K):
                for i in range(self.N):
                    self.x[i] += self.c_k[j] * self.h * self.v[i]
                for i in range(self.N):
                    self.v[i] += (
                        self.d_k[j] * self.h * self._a(self.x, self.v, i, cur_input)
                    )
            self.t += self.h
        self.input_0_steps_done += nbr_steps
        return

    def _simulate_steps_cython(self, nbr_steps):
        if len(self.input_seq.shape) == 1:
            update = simulate_steps_damped_cython
        elif len(self.input_seq.shape) == 2:
            update = simulate_steps_damped_multi_input_cython
        else:
            raise ValueError
        self.t, self.x, self.v, self.input_0_steps_done, input_used = update(
            self.t,
            nbr_steps,
            self.x,
            self.v,
            self.h,
            self.N,
            self.alpha,
            self.gamma,
            self.input_duration,
            self.input_amplitude,
            self.input_seq[
                : (nbr_steps + self.input_0_steps_done) // self.input_duration
            ],
            self.input_0_steps_done,
        )
        assert len(self.input_seq) >= input_used
        self.input_seq = self.input_seq[input_used:]
        self.x = np.array(self.x)
        self.v = np.array(self.v)
        return

    def _simulate_batches_with_func(self, sim_func, nbr_batches):
        """
        Use the specified function to simulate the given number of batches.
        :param sim_func: The method to be used for the simulation of the system.
        :param nbr_batches: Number of input batches to simulate.
        :return: The input sequence used during the simulation.
        """
        assert self.input_0_steps_done == 0
        return_seq = self.input_seq[:nbr_batches]
        self._simulate_steps_with_func(sim_func, nbr_batches * self.input_duration)
        assert self.input_0_steps_done == 0
        return return_seq

    def simulate_batches_no_cython(self, nbr_batches):
        """
        Use the python implementation to simulate the given number of batches.
        :param nbr_batches: Number of input batches to simulate.
        :return: The input sequence used during the simulation.
        """
        return self._simulate_batches_with_func(
            self._simulate_steps_no_readout, nbr_batches
        )

    def simulate_batches(self, nbr_batches):
        """
        Use the cython implementation to simulate the given number of batches. Requires the compiled cython script.
        :param nbr_batches: Number of input batches to simulate.
        :return: The input sequence used during the simulation.
        """
        return self._simulate_batches_with_func(
            self._simulate_steps_cython, nbr_batches
        )

    def getCurExternalEnergy(self):
        """
        Get the current external potential energy generated by the input.
        :return: Current external potential energy.
        """
        input_state = 1 if self.input_seq[0] == 1 else -1
        return -self.input_amplitude * input_state * np.sum(self.x)

    def _give_input(self, i, input_state):
        if input_state == 0:
            input_state = -1
        return self.input_amplitude * input_state


def prepare_initcond_for_unit_test():
    N = 32
    x0 = np.zeros(N)
    v0 = np.zeros(N)
    t_start = 0

    def w_k(k):
        return 2 * np.sin(np.pi * (k + 1) / (2 * (N + 1)))

    def Q_k(k, x):
        Q_temp = 0
        for i in range(1, N + 1):
            Q_temp += (
                np.sqrt(2 / (N + 1)) * x[i - 1] * np.sin(np.pi * (k + 1) * i / (N + 1))
            )
        return Q_temp

    modes = [0]
    # setup initial state with modes to excite
    Q_init = np.zeros(N)
    P_init = np.zeros(N)
    eps = 0.0217  # 1.0/N
    E_in = eps * N / len(modes)
    for j in range(len(modes)):
        np.random.seed(1234)
        phase = np.random.uniform(low=0.0, high=2 * np.pi)
        print("initial phase:", phase)
        Q_init[modes[j]] = np.sqrt(2 * E_in) / w_k(modes[j]) * np.sin(phase)
        P_init[modes[j]] = np.sqrt(2 * E_in) * np.cos(phase)
        for i in range(N):
            x0[i] = Q_k(i, Q_init)
            v0[i] = Q_k(i, P_init)
    return x0, v0


def unit_testing_cython():
    x0, v0 = prepare_initcond_for_unit_test()

    cur = Simulation(init_state=np.concatenate((x0, v0)))
    cur.startReadout(readout_every=10)
    # cur.simulate_batches(550)
    cur.simulate_steps(5500)

    cur.energies_plot()

    cur2 = Simulation(init_state=np.concatenate((x0, v0)))
    cur2.startReadout(readout_every=10)
    # cur2.simulate_batches(550)
    cur2.simulate_steps_no_cython(5500)

    cur2.energies_plot()

    t_vec = np.array(cur.ts)
    x_vec, v_vec = np.array(cur.readout)[:, : cur.N], np.array(cur.readout)[:, cur.N :]
    E_vec = np.array(cur.E_n_readout)
    E_anh_vec = np.array(cur.E_anharmonic_readout)

    assert np.allclose(t_vec, np.array(cur2.ts))
    assert np.allclose(x_vec, np.array(cur2.readout)[:, : cur2.N])
    assert np.allclose(v_vec, np.array(cur2.readout)[:, cur2.N :])
    assert np.allclose(E_vec, np.array(cur2.E_n_readout))
    assert np.allclose(E_anh_vec, np.array(cur2.E_anharmonic_readout))

    return t_vec, x_vec, v_vec, E_vec, E_anh_vec


def unit_testing():
    x0, v0 = prepare_initcond_for_unit_test()

    cur = Simulation(init_state=np.concatenate((x0, v0)))
    cur.startReadout(readout_every=100)
    # cur.simulate_batches(550)
    cur.simulate_steps(60000)

    cur.energies_plot()

    t_vec = np.array(cur.ts)
    x_vec, v_vec = np.array(cur.readout)[:, : cur.N], np.array(cur.readout)[:, cur.N :]
    E_vec = np.array(cur.E_n_readout)
    E_anh_vec = np.array(cur.E_anharmonic_readout)

    return t_vec, x_vec, v_vec, E_vec, E_anh_vec


def unit_testing_initial_conditions():
    cur = Simulation()
    cur.startReadout(readout_every=10)
    cur.simulate_steps(5500)

    cur.energies_plot()

    t_vec = np.array(cur.ts)
    x_vec, v_vec = np.array(cur.readout)[:, : cur.N], np.array(cur.readout)[:, cur.N :]
    E_vec = np.array(cur.E_n_readout)
    E_anh_vec = np.array(cur.E_anharmonic_readout)

    return t_vec, x_vec, v_vec, E_vec, E_anh_vec


def unit_testing_initial_jitter():
    cur = Simulation(jitter=0.1)
    cur.startReadout(readout_every=10)
    cur.simulate_steps(5500)

    cur.energies_plot()

    t_vec = np.array(cur.ts)
    x_vec, v_vec = np.array(cur.readout)[:, : cur.N], np.array(cur.readout)[:, cur.N :]
    E_vec = np.array(cur.E_n_readout)
    E_anh_vec = np.array(cur.E_anharmonic_readout)

    return t_vec, x_vec, v_vec, E_vec, E_anh_vec


def unit_testing_damped():
    x0, v0 = prepare_initcond_for_unit_test()

    # cur = DampedSimulation(init_state=np.concatenate((x0, v0)), input_duration=10, input_amplitude=0.005, alpha=0.0)
    cur = DampedSimulation(
        init_epsilon=0.01,
        input_duration=300,
        input_amplitude=0.005,
        alpha=0.25,
        tau_relax=2.5,
    )
    cur.simulate_batches(50000)
    cur.startReadout(readout_every=50)
    cur.simulate_batches(100)
    # cur.simulate_steps(550000)

    cur.energies_plot()

    t_vec = np.array(cur.ts)
    x_vec, v_vec = np.array(cur.readout)[:, : cur.N], np.array(cur.readout)[:, cur.N :]
    E_vec = np.array(cur.E_n_readout)
    E_anh_vec = np.array(cur.E_anharmonic_readout)
    E_ext_vec = np.array(cur.E_external_readout)

    return t_vec, x_vec, v_vec, E_vec, E_anh_vec, E_ext_vec


if __name__ == "__main__":
    DEBUG = True
    E_ext_vec = None
    #% time t_vec, x_vec, v_vec, E_vec, E_anh_vec = unit_testing()
    # %time t_vec, x_vec, v_vec, E_vec, E_anh_vec, E_ext_vec = unit_testing_damped()
