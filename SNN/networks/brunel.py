from copy import deepcopy
import numpy as np
import nest

from SNN.networks.base_network import BaseNetwork
from SNN.utils import input_utils


class BrunelNetwork(BaseNetwork):

    default_neuron_params = {
        'C_m': 1.0,  # membrane capacity (pF)
        'E_L': 0.,  # resting membrane potential (mV)
        'I_e': 0.,  # external input current (pA)
        'V_m': 0.,  # membrane potential (mV)
        'V_reset': 10.,  # reset membrane potential after a spike (mV)
        'V_th': 20.,  # spike threshold (mV)
        't_ref': 2.0,  # refractory period (ms)
        'tau_m': 20.,  # membrane time constant (ms)
    }

    def __init__(self, neuron_model='iaf_psc_delta', neuron_params=None, N=1250, g=5., J=0.2, density=0.1, delay=1.5,
                 CEE=None):
        super().__init__()
        self.neuron_model = neuron_model
        if neuron_params is None:
            neuron_params = self.default_neuron_params.copy()
        self.neuron_params = neuron_params
        self.N = N
        self.NE = int(self.N * 0.8)
        self.NI = N - self.NE
        self.g = g
        self.J = J
        self.input_weight = J
        self.density = density
        self.CE = int(self.NE * self.density)
        self.CI = int(self.NI * self.density)
        if CEE is None:
            self.CEE = self.CE
        else:
            self.CEE = CEE
        self.delay = delay
        # if neuron_model == 'iaf_psc_delta':
        #     self.nu_th = 1000. * self.neuron_params['V_th'] / (self.CE * self.J * self.neuron_params['tau_m'])
        self.populations = self._create_populations()
        self.connect_net()

    def _create_populations(self):
        params_with_distribution = {}
        neuron_params_copy = deepcopy(self.neuron_params)
        for parname, parvalue in self.neuron_params.items():
            if isinstance(parvalue, dict):
                params_with_distribution[parname] = parvalue
                neuron_params_copy[parname] = self.default_neuron_params[parname]
                # del neuron_params_copy[parname]

        pop_dict = {
            'E': nest.Create(self.neuron_model, n=self.NE, params=neuron_params_copy),
            'I': nest.Create(self.neuron_model, n=self.NI, params=neuron_params_copy)
        }

        for parname, parvalue in params_with_distribution.items():
            fct = parvalue['function']
            fct_pars = parvalue['parameters']
            nest.SetStatus(pop_dict['E'], parname, eval(f"{fct}(size=self.NE, **fct_pars)"))
            nest.SetStatus(pop_dict['I'], parname, eval(f"{fct}(size=self.NI, **fct_pars)"))

        vreset = self.neuron_params['V_reset']
        if isinstance(self.neuron_params['V_th'], dict):
            try:
                vth = self.neuron_params['V_th']['parameters']['low']
            except:
                vth = self.default_neuron_params['V_th']
        else:
            vth = self.neuron_params['V_th']
        for pop in pop_dict.values():
            nest.SetStatus(pop, 'V_m', np.random.uniform(vreset, vth, size=len(pop)))

        return pop_dict

    def connect_net(self):
        Jinh = -self.g * self.J
        # conn_dict = {'rule': 'pairwise_bernoulli', 'p': self.density}
        conn_dict_exc = {'rule': 'fixed_indegree', 'indegree': self.CE}
        conn_dict_inh = {'rule': 'fixed_indegree', 'indegree': self.CI}
        conn_dict_ee = {'rule': 'fixed_indegree', 'indegree': self.CEE}
        syn_dict_exc = {'weight': self.J, 'delay': self.delay}
        # syn_dict_exc = {'weight': np.sqrt(10)*0.1, 'delay': self.delay}
        # syn_dict_exc = {'weight': 0., 'delay': self.delay}
        syn_dict_inh = {'weight': Jinh, 'delay': self.delay}

        # nest.Connect(self.populations['E'], self.populations['E'], syn_spec=syn_dict_exc, conn_spec=conn_dict_exc)
        nest.Connect(self.populations['E'], self.populations['E'], syn_spec=syn_dict_exc, conn_spec=conn_dict_ee)
        nest.Connect(self.populations['E'], self.populations['I'], syn_spec=syn_dict_exc, conn_spec=conn_dict_exc)
        nest.Connect(self.populations['I'], self.populations['I'], syn_spec=syn_dict_inh, conn_spec=conn_dict_inh)
        nest.Connect(self.populations['I'], self.populations['E'], syn_spec=syn_dict_inh, conn_spec=conn_dict_inh)

    def add_spiking_noise(self, rate=None, weight=None, loop_duration=None):
        if rate is None:
            rate = 40. * self.CE

        if weight is None:
            weight = self.J

        if loop_duration is None:
            noise_generator = input_utils.add_poisson_noise(self.populations.values(), rate=rate, weight=weight)
            parrots = None
        else:
            noise_generator, parrots = input_utils.add_repeating_noise(self.populations.values(), rate, weight, loop_duration)

        return noise_generator, parrots

    def add_ac_current_noise(self, loop_duration, n_sources=1000, indegree=100, mean_input=0.54):

        p = indegree / n_sources
        mean_offset = mean_input / indegree
        # offset_radius = mean_offset * 0.05
        # offsets = np.random.uniform(mean_offset-offset_radius, mean_offset+offset_radius, n_sources)
        offsets = np.ones(n_sources) * mean_offset
        frequencies = np.random.randint(1, 20, n_sources)*(1000. / loop_duration)
        amp_mean = mean_input * 0.75
        amp_radius = amp_mean * (1/3)
        amplitudes = np.random.uniform(amp_mean-amp_radius, amp_mean+amp_radius, n_sources)

        noise_generators = nest.Create('ac_generator', n=n_sources)
        noise_generators.set({
            'phase': np.random.uniform(0., 360., n_sources),
            'offset':offsets,
            'amplitude': amplitudes,
            'frequency': frequencies
        })
        for pop in self.populations.values():
            nest.Connect(noise_generators, pop, {'rule': 'pairwise_bernoulli', 'p': p})

        return noise_generators

    def get_state_populations(self):
        return {'E': self.populations['E']}

    def get_input_populations(self):
        return {'E': self.populations['E']}


if __name__ == "__main__":
    network = BrunelNetwork()
