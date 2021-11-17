import nest

from networks.base_network import BaseNetwork
from utils import input_utils, state_utils


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

    def __init__(self, neuron_model='iaf_psc_delta', neuron_params=None, N=1250, g=5., J=0.2, density=0.1, delay=1.5, CEE=None):
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
        self.density = density
        self.CE = int(self.NE * self.density)
        self.CI = int(self.NI * self.density)
        if CEE is None:
            self.CEE = self.CE
        else:
            self.CEE = CEE
        self.delay = delay
        if neuron_model == 'iaf_psc_delta':
            self.nu_th = 1000. * self.neuron_params['V_th'] / (self.CE * self.J * self.neuron_params['tau_m'])
        self.populations = self._create_populations()
        self.connect_net()

    def _create_populations(self):
        pop_dict = {
            'E': nest.Create(self.neuron_model, n=self.NE, params=self.neuron_params),
            'I': nest.Create(self.neuron_model, n=self.NI, params=self.neuron_params)
        }

        return pop_dict

    def connect_net(self):
        Jinh = -self.g * self.J
        # conn_dict = {'rule': 'pairwise_bernoulli', 'p': self.density}
        conn_dict_exc = {'rule': 'fixed_indegree', 'indegree': self.CE}
        conn_dict_inh = {'rule': 'fixed_indegree', 'indegree': self.CI}
        conn_dict_ee = {'rule': 'fixed_indegree', 'indegree': self.CEE}
        syn_dict_exc = {'weight': self.J, 'delay': self.delay}
        syn_dict_inh = {'weight': Jinh, 'delay': self.delay}

        # nest.Connect(self.populations['E'], self.populations['E'], syn_spec=syn_dict_exc, conn_spec=conn_dict_exc)
        nest.Connect(self.populations['E'], self.populations['E'], syn_spec=syn_dict_exc, conn_spec=conn_dict_ee)
        nest.Connect(self.populations['E'], self.populations['I'], syn_spec=syn_dict_exc, conn_spec=conn_dict_exc)
        nest.Connect(self.populations['I'], self.populations['I'], syn_spec=syn_dict_inh, conn_spec=conn_dict_inh)
        nest.Connect(self.populations['I'], self.populations['E'], syn_spec=syn_dict_inh, conn_spec=conn_dict_inh)

    def add_default_noise(self, rate=None, weight=None):
        if rate is None:
            rate = 40. * self.CE

        if weight is None:
            weight = self.J

        noise_generator = input_utils.add_poisson_noise(self.populations.values(), rate=rate, weight=weight)

        return noise_generator

    def get_state_populations(self):
        return {'E': self.populations['E']}

    def get_input_populations(self):
        return {'E': self.populations['E']}


if __name__ == "__main__":
    network = BrunelNetwork()
