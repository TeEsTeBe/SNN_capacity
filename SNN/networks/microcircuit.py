import numpy as np
import nest

from SNN.networks.base_network import BaseNetwork
from SNN.utils import connection_utils, input_utils


class Microcircuit(BaseNetwork):
    """ Data based microcircuit network model

    based on  T. Schulte to Brinke, R. Duarte, and A. Morrison,
    “Characteristic columnar connectivity caters to cortical computation: Replication, simulation, and evaluation of a
    microcircuit model,” Frontiers in Integrative Neuroscience, vol. 16, 2022, Accessed: Oct. 24, 2022. [Online].
    Available: https://www.frontiersin.org/articles/10.3389/fnint.2022.923468

    """

    # Connection weights in mV
    psp_amp_from_to = {
        'l23_exc': {
            'l23_exc': 1.7,
            'l23_inh': 1.9,
            'l4_exc': 0.,
            'l4_inh': 1.6,
            'l5_exc': 1.4,
            'l5_inh': 0.,
        },
        'l23_inh': {
            'l23_exc': -0.65,
            'l23_inh': -1.35,
            'l4_exc': 0.,
            'l4_inh': 0.,
            'l5_exc': -5.2,
            'l5_inh': 0.,
        },
        'l4_exc': {
            'l23_exc': 4.,
            'l23_inh': 0.15,
            'l4_exc': 1.1,
            'l4_inh': 3.7,
            'l5_exc': 0.,
            'l5_inh': 0.,
        },
        'l4_inh': {
            'l23_exc': -1.75,
            'l23_inh': -1.5,
            'l4_exc': -0.85,
            'l4_inh': -1.55,
            'l5_exc': 0.,
            'l5_inh': 0.,
        },
        'l5_exc': {
            'l23_exc': 0.3,
            'l23_inh': 0.,
            'l4_exc': 0.,
            'l4_inh': 0.,
            'l5_exc': 1.7,
            'l5_inh': 0.9,
        },
        'l5_inh': {
            'l23_exc': 0.,
            'l23_inh': 0.,
            'l4_exc': 0.,
            'l4_inh': 0.,
            'l5_exc': -1.2,
            'l5_inh': -1.2,
        },
    }

    probabilities_from_input1 = {
        'l23_exc': 0.2,
        'l23_inh': 0.,
        'l4_exc': 0.8,
        'l4_inh': 0.5,
        'l5_exc': 0.1,
        'l5_inh': 0.,
    }

    probabilities_from_input2 = {
        'l23_exc': 0.2,
        'l23_inh': 0.,
        'l4_exc': 0.,
        'l4_inh': 0.,
        'l5_exc': 0.,
        'l5_inh': 0.,
    }

    # Connection probabilities
    probabilities_from_to = {
        'l23_exc': {
            'l23_exc': 0.26,
            'l23_inh': 0.21,
            'l4_exc': 0.,
            'l4_inh': 0.08,
            'l5_exc': 0.55,
            'l5_inh': 0.,
        },
        'l23_inh': {
            'l23_exc': 0.16,
            'l23_inh': 0.25,
            'l4_exc': 0.,
            'l4_inh': 0.,
            'l5_exc': 0.2,
            'l5_inh': 0.,
        },
        'l4_exc': {
            'l23_exc': 0.28,
            'l23_inh': 0.1,
            'l4_exc': 0.17,
            'l4_inh': 0.19,
            'l5_exc': 0.,
            'l5_inh': 0.,
        },
        'l4_inh': {
            'l23_exc': 0.5,
            'l23_inh': 0.2,
            'l4_exc': 0.1,
            'l4_inh': 0.5,
            'l5_exc': 0.,
            'l5_inh': 0.,
        },
        'l5_exc': {
            'l23_exc': 0.03,
            'l23_inh': 0.,
            'l4_exc': 0.,
            'l4_inh': 0.,
            'l5_exc': 0.09,
            'l5_inh': 0.1,
        },
        'l5_inh': {
            'l23_exc': 0.,
            'l23_inh': 0.,
            'l4_exc': 0.,
            'l4_inh': 0.,
            'l5_exc': 0.12,
            'l5_inh': 0.6,
        },
    }

    def __init__(self, neuron_model, neuron_params_exc, neuron_params_inh, N=560, S_rw=119.3304,
                 static_synapses=False, random_synaptic_dynamics=False, vt_l23exc=None, vt_l23inh=None, vt_l4exc=None,
                 vt_l4inh=None, vt_l5exc=None, vt_l5inh=None):
        super().__init__()
        self.network_type = 'microcircuit'
        self.neuron_model = neuron_model
        self.install_neuron_model_if_necessary()
        self.v_ths = {
            'l23_exc': vt_l23exc,
            'l23_inh': vt_l23inh,
            'l4_exc': vt_l4exc,
            'l4_inh': vt_l4inh,
            'l5_exc': vt_l5exc,
            'l5_inh': vt_l5inh,
        }
        self.neuron_params_exc = neuron_params_exc
        self.neuron_params_inh = neuron_params_inh
        # input_weight = 1.9  # mV
        self.input_weight = 30. / self.neuron_params_exc['g_L']  # 1.924779612734342
        self.N = N
        self.S_rw = S_rw
        self.neuron_pars = self.create_neuron_pars()
        self.populations, self.pop_counts = self.create_populations()
        self._input_populations = None
        self.random_synaptic_dynamics = random_synaptic_dynamics
        self.static_synapses = static_synapses
        if static_synapses:
            self.syn_params_from_to = connection_utils.create_synapse_parameters('static_synapse')
        else:
            self.syn_params_from_to = connection_utils.create_synapse_parameters()
        self.conn_ids_from_to = self.connect_net()

        self.statemat23exc = None
        self.statemat5exc = None

    def install_neuron_model_if_necessary(self):
        if self.neuron_model not in nest.Models():
            # nest.Install('destexhemodule')
            nest.Install('destexheascppvthreshmodule')
            # nest.Install('destexheascppscaling2module')
            # nest.Install('destexheascppscalingunitsmodule')
            # nest.Install('scalingsimplifiedimnoscaledmodule')
            # nest.Install('scaleonlyvmodule')
            # nest.Install('variablescalingmodule')
            # nest.Install('variablescalingVmIdiffmodule')
            # nest.Install('variablescalingecorrectedmodule')
            # nest.Install('variablescalingvmdiffiecorrectedmodule')
            print('Destexhe NEST module installed')
        else:
            print('Destexhe NEST module has been installed before')

    def create_neuron_pars(self):
        """ Sets up the neuron parameters for each population """

        neuron_pars = {
            'l23_exc': self.neuron_params_exc.copy(),
            'l23_inh': self.neuron_params_inh.copy(),
            'l4_exc': self.neuron_params_exc.copy(),
            'l4_inh': self.neuron_params_inh.copy(),
            'l5_exc': self.neuron_params_exc.copy(),
            'l5_inh': self.neuron_params_inh.copy(),
        }

        for pop_name, vth in self.v_ths.items():
            if vth is not None:
                neuron_pars[pop_name]['V_th'] = vth

        return neuron_pars

    def calculate_connection_ids_from_to(self):
        connection_ids_from_to = {}
        for src_name, src_neurons in self.populations.items():
            if src_name not in connection_ids_from_to.keys():
                connection_ids_from_to[src_name] = {}
            for trg_name, trg_neurons in self.populations.items():
                connection_ids_from_to[src_name][trg_name] = nest.GetConnections(source=src_neurons, target=trg_neurons)

        return connection_ids_from_to

    def get_all_neuron_pop_dicts(self):
        neuron_pop_dicts = []
        for pop_name, pop_neurons in self.populations.items():
            for neuron in pop_neurons:
                neuron_pop_dicts.append({'neuron': neuron, 'pop': pop_name})

        return neuron_pop_dicts

    def get_neurons_separated_by_exc_inh(self):
        """ Creates a dict which contains the neurons separated by excitatory and inhibitory

        Returns
        -------
        neurons: dict
            dictionary with neurons separated by excitatory and inhibitory

        """

        exc_neurons = None
        inh_neurons = None
        for pop_name, nodes in self.populations.items():
            pop_type = pop_name[-3:]
            if pop_type == 'exc':
                if exc_neurons is None:
                    exc_neurons = nodes
                else:
                    exc_neurons += nodes
            elif pop_type == 'inh':
                if inh_neurons is None:
                    inh_neurons = nodes
                else:
                    inh_neurons += nodes

        neurons = {
            'exc': exc_neurons,
            'inh': inh_neurons,
        }

        return neurons

    def create_populations(self):
        """ Creates the populations of neurons that are the basis of the network

        Returns
        -------
        populations: dict
            dictionary with nest neuron collections for each population
        pop_counts: dict
            neuron counts for the different layers

        """

        # Layer 2/3
        N23 = int(self.N * 0.3)
        N23e = int(N23 * 0.8)
        N23i = N23 - N23e

        # Layer 4
        N4 = int(self.N * 0.2)
        N4e = int(N4 * 0.8)
        N4i = N4 - N4e

        # Layer 5
        N5 = int(self.N * 0.5)
        N5e = int(N5 * 0.8)
        N5i = N5 - N5e

        populations = {
            'l5_exc': nest.Create(self.neuron_model, n=N5e, params=self.neuron_pars['l5_exc']),
            'l5_inh': nest.Create(self.neuron_model, n=N5i, params=self.neuron_pars['l5_inh']),
            'l4_exc': nest.Create(self.neuron_model, n=N4e, params=self.neuron_pars['l4_exc']),
            'l4_inh': nest.Create(self.neuron_model, n=N4i, params=self.neuron_pars['l4_inh']),
            'l23_exc': nest.Create(self.neuron_model, n=N23e, params=self.neuron_pars['l23_exc']),
            'l23_inh': nest.Create(self.neuron_model, n=N23i, params=self.neuron_pars['l23_inh'])
        }

        for _, pop in populations.items():
            nest.SetStatus(pop, 'V_m', np.random.uniform(-70, -60, size=len(pop)))

        pop_counts = {
            'all': self.N,
            'l23': N23,
            'l4': N4,
            'l5': N5
        }
        for pop, neurons in populations.items():
            pop_counts[pop] = len(neurons)

        return populations, pop_counts

    def connect_net(self, print_connections=False):
        """ Connects the neuron populations to create the networks connectivity structure

        Parameters
        ----------
        print_connections: bool
            whether to print the generated connections between populations

        Returns
        -------
        connection_ids_from_to: dict
            dictionary that contains all nest Connections in the network, organized by dict[src_pop][trg_pop]

        """

        connection_ids_from_to = {}
        for (src_pop, trg_w_dict), (_, trg_probs) in zip(self.psp_amp_from_to.items(), self.probabilities_from_to.items()):
            connection_ids_from_to[src_pop] = {}
            if print_connections:
                print(f'\nsource population = {src_pop}')
            for (trg_pop, weight), (_, probability) in zip(trg_w_dict.items(), trg_probs.items()):

                if self.random_synaptic_dynamics:
                    from_type = np.random.choice(list(self.syn_params_from_to.keys()))
                    to_type = np.random.choice(list(self.syn_params_from_to[from_type].keys()))
                    syn_dict = self.syn_params_from_to[from_type][to_type].copy()
                else:
                    syn_dict = self.syn_params_from_to[src_pop[-3:]][trg_pop[-3:]].copy()
                weight = connection_utils.calc_synaptic_weight(
                    psp_amp=weight,
                    scaling_factor=self.S_rw,
                    exc_or_inh_src=src_pop[-3:],
                    g_L=self.neuron_pars[src_pop]['g_L']
                )
                syn_dict['weight'] = weight
                conn_dict = {'rule': 'pairwise_bernoulli', 'p': probability}

                if print_connections:
                    print(f'{src_pop}\t-----{weight} ({probability})-->\t{trg_pop}\t{self.syn_params_from_to[src_pop[-1]][trg_pop[-1]]}')

                if abs(weight) > 0 and self.S_rw > 0:
                    connection_ids_from_to[src_pop][trg_pop] = connection_utils.connect_population_pair(
                        source_ids=self.populations[src_pop],
                        target_ids=self.populations[trg_pop],
                        syn_dict=syn_dict,
                        conn_dict=conn_dict,
                        static_synapses=self.static_synapses
                    )

        return connection_ids_from_to

    def connect_input_stream1(self, spike_generators, scaling_factor, weight=None):
        """ Connects the first input stream to the network"""

        self.connect_input_stream(spike_generators, self.probabilities_from_input1, scaling_factor, weight=weight)

    def connect_input_stream2(self, spike_generators, scaling_factor, weight=None):
        """ Connects the second input stream to the network"""

        self.connect_input_stream(spike_generators, self.probabilities_from_input2, scaling_factor, weight=weight)

    def connect_input_stream(self, spike_generators, connection_probabilities, scaling_factor, weight=None):
        """ Connects one of the input streams to the network

        Parameters
        ----------
        spike_generators
            NEST spike generators that will be connected to the network
        connection_probabilities: dict
            dictionary with the connection probabilities for all populations
        scaling_factor: float
            scaling factor of the connection weight
        weight: float
            weight of the input connections

        Returns
        -------
        None

        """

        if weight is None:
            scaled_input_weight = connection_utils.calc_synaptic_weight(
                self.input_weight,
                scaling_factor,
                'exc',
                self.neuron_params_exc['g_L']
            )
        else:
            print('Scaling factor is not used, because weight is set directly.')
            scaled_input_weight = weight

        for pop_name, pop_neurons in self.populations.items():
            if connection_probabilities[pop_name] > 0.:
                nest.Connect(spike_generators, pop_neurons,
                             syn_spec={'weight': scaled_input_weight},
                             conn_spec={'rule': 'pairwise_bernoulli', 'p': connection_probabilities[pop_name]})
                connection_utils.randomize_conn_parameter(spike_generators, pop_neurons, 'weight', scaled_input_weight, 0.7)

    def add_spiking_noise(self, rate=None, weight=None, loop_duration=None):
        """ Creates a spike generator and connects it to all populations

        Parameters
        ----------
        rate: float
            firing rate of the spike trains that should be generated
        weight: float
            weight for the connections from the generator to the network
        loop_duration: float
            if this is not None the generator produces frozen noise, that repeats every loop_duration ms

        Returns
        -------
        noise_generator
            nest poisson generator object
        parrots
            nest parrot neurons that are used to generate the frozen noise

        """

        if rate is None:
            rate = 20.

        n_generators_s1 = 40
        n_generators_s2 = 40

        s1 = 14.85
        s2 = 36.498

        if loop_duration is None:
            noise_generators_s1 = nest.Create('poisson_generator', params={'rate': rate}, n=n_generators_s1)
            noise_generators_s2 = nest.Create('poisson_generator', params={'rate': rate}, n=n_generators_s2)
            self.connect_input_stream1(noise_generators_s1, scaling_factor=s1, weight=weight)
            self.connect_input_stream2(noise_generators_s2, scaling_factor=s2, weight=weight)

            # noise_generator = input_utils.add_poisson_noise(self.populations.values(), rate=rate, weight=weight)
            noise_generator = [noise_generators_s1, noise_generators_s2]
            parrots = None
        else:
            # noise_generator, parrots = input_utils.add_repeating_noise(self.populations.values(), rate, weight, loop_duration)
            noise_generator_S1, parrots_S1 = input_utils.add_repeating_noise([list(range(n_generators_s1))], rate=rate, weight=None, loop_duration=loop_duration, connect_to_populations=False)
            noise_generator_S2, parrots_S2 = input_utils.add_repeating_noise([list(range(n_generators_s2))], rate=rate, weight=None, loop_duration=loop_duration, connect_to_populations=False)
            self.connect_input_stream1(parrots_S1, scaling_factor=s1, weight=weight)
            self.connect_input_stream2(parrots_S2, scaling_factor=s2, weight=weight)

            noise_generator = [noise_generator_S1, noise_generator_S2]
            parrots = [parrots_S1, parrots_S2]

        return noise_generator, parrots

    def add_ac_current_noise(self, loop_duration, n_sources=1000, indegree=100, mean_input=0.54):
        raise NotImplementedError('The function add_ac_current_noise is not yet fully implmented for the microcircuit!')

        # p = indegree / n_sources
        mean_offset = mean_input / indegree
        # offset_radius = mean_offset * 0.05
        # offsets = np.random.uniform(mean_offset-offset_radius, mean_offset+offset_radius, n_sources)
        offsets = np.ones(n_sources) * mean_offset
        frequencies = np.random.randint(1, 20, n_sources)*(1000. / loop_duration)
        amp_mean = mean_input * 0.75
        amp_radius = amp_mean * (1/3)
        amplitudes = np.random.uniform(amp_mean-amp_radius, amp_mean+amp_radius, n_sources)

        noise_generators_s1 = nest.Create('ac_generator', n=n_sources)
        noise_generators_s1.set({
            'phase': np.random.uniform(0., 360., n_sources),
            'offset':offsets,
            'amplitude': amplitudes,
            'frequency': frequencies
        })
        self.connect_input_stream1(noise_generators_s1, scaling_factor=None, weight=1.)

        noise_generators_s2 = nest.Create('ac_generator', n=n_sources)
        noise_generators_s2.set({
            'phase': np.random.uniform(0., 360., n_sources),
            'offset':offsets,
            'amplitude': amplitudes,
            'frequency': frequencies
        })
        self.connect_input_stream2(noise_generators_s2, scaling_factor=None, weight=1.)
        # for pop in self.populations.values():
        #     nest.Connect(noise_generators, pop, {'rule': 'pairwise_bernoulli', 'p': p})

        return [noise_generators_s1, noise_generators_s2]

    def get_state_populations(self):
        """ Returns the populations for the readout"""

        state_populations = {}
        for pop_name, pop_neurons in self.populations.items():
            if 'exc' in pop_name:
                state_populations[pop_name] = pop_neurons

        return state_populations

    def get_input_populations(self):
        """ Returns the populations that get an input signal """

        if self._input_populations is None:
            self._input_populations = {}

            # input stream 1
            for pop_name, pop_neurons in self.populations.items():
                connection_probability = self.probabilities_from_input1[pop_name]
                if connection_probability > 0:
                    # n_neurons = int(connection_probability*len(pop_neurons))
                    # chosen_ids = np.random.choice(pop_neurons.global_id, n_neurons, replace=False)
                    # self._input_populations[f'{pop_name}_S1'] = nest.NodeCollection(sorted(chosen_ids))
                    self._input_populations[f'{pop_name}_S1'] = pop_neurons

            # for now we only use the first input stream
            # # input stream 2
            # for pop_name, pop_neurons in self.populations.items():
            #     connection_probability = self.probabilities_from_input2[pop_name]
            #     if connection_probability > 0:
            #         n_neurons = int(connection_probability*len(pop_neurons))
            #         chosen_ids = np.random.choice(pop_neurons.global_id, n_neurons, replace=False)
            #         self._input_populations[f'{pop_name}_S2'] = nest.NodeCollection(chosen_ids)

        return self._input_populations
