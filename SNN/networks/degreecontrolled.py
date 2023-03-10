import numpy as np
import nest

from SNN.networks.microcircuit import Microcircuit
from SNN.utils import connection_utils


class DegreeControlledCircuit(Microcircuit):
    """ Data-based Microcircuit with scrambled connections, but kept degree distributions """

    def __init__(self, N=560, S_rw=119.3304, neuron_params_exc=None, neuron_params_inh=None,
                 remove_io_specificity=False, neuron_model=None, vt_l23exc=None,
                 vt_l23inh=None, vt_l4exc=None, vt_l4inh=None, vt_l5exc=None, vt_l5inh=None):
        super().__init__(N=N, S_rw=S_rw, neuron_model=neuron_model, neuron_params_exc=neuron_params_exc,
                         neuron_params_inh=neuron_params_inh,
                         vt_l23exc=vt_l23exc, vt_l23inh=vt_l23inh, vt_l4exc=vt_l4exc,
                         vt_l4inh=vt_l4inh, vt_l5exc=vt_l5exc, vt_l5inh=vt_l5inh)
        self.network_type = 'degreecontrolled'
        self.remove_io_specificity = remove_io_specificity
        if remove_io_specificity:
            self.scramble_neurons()

    def scramble_neurons(self):
        """ Scrambles the data-based connectivity structure, but keeps the degree distributions """

        neurons_exc_inh = self.get_neurons_separated_by_exc_inh()
        neuron_ids_exc = list(neurons_exc_inh['exc'].global_id)
        neuron_ids_inh = list(neurons_exc_inh['inh'].global_id)
        np.random.shuffle(neuron_ids_exc)
        np.random.shuffle(neuron_ids_inh)

        idx_start_exc = 0
        idx_start_inh = 0
        new_populations = {}
        for pop_name, pop_neurons in self.populations.items():
            pop_count = self.pop_counts[pop_name]
            pop_type = pop_name[-3:]
            if pop_type == 'exc':
                chosen_ids = sorted(neuron_ids_exc[idx_start_exc:idx_start_exc + pop_count])
                new_populations[pop_name] = nest.NodeCollection(chosen_ids)
                idx_start_exc += pop_count
            elif pop_type == 'inh':
                chosen_ids = sorted(neuron_ids_inh[idx_start_inh:idx_start_inh + pop_count])
                new_populations[pop_name] = nest.NodeCollection(chosen_ids)
                idx_start_inh += pop_count
            else:
                raise ValueError('Naming of populations is not consistent. Every population name should end with exc or'
                                 f' inh. Used population name: {pop_name}')

        self.populations = new_populations.copy()

    def connect_net(self, print_connections=False):

        if print_connections:
            print(f'Print out of connections is not implemented for amorhous circuit!')

        source_neurons = []
        target_neurons = []
        synaptic_parameters = {
            'synapse_model': [],
            'weight': [],
            'U': [],
            'u': [],
            'tau_rec': [],
            'tau_fac': [],
            'delay': [],
        }

        for (src_pop, trg_w_dict), (_, trg_probs) in zip(self.psp_amp_from_to.items(),
                                                         self.probabilities_from_to.items()):
            for (trg_pop, weight), (_, probability) in zip(trg_w_dict.items(), trg_probs.items()):

                if probability > 0.:

                    syn_dict = self.syn_params_from_to[src_pop[-3:]][trg_pop[-3:]].copy()
                    weight = connection_utils.calc_synaptic_weight(
                        psp_amp=weight,
                        scaling_factor=self.S_rw,
                        exc_or_inh_src=src_pop[-3:],
                        g_L=self.neuron_pars[src_pop]['g_L']
                    )
                    syn_dict['weight'] = weight

                    n_connections = 0
                    for src_neuron in self.populations[src_pop]:
                        for trg_neuron in self.populations[trg_pop]:

                            if np.random.uniform(low=0., high=1., size=1) <= probability:
                                source_neurons.append(src_neuron)
                                target_neurons.append(trg_neuron)
                                n_connections += 1

                    synaptic_parameters['synapse_model'].extend(
                        ['tsodyks2_synapse' for _ in range(n_connections)])
                    synaptic_parameters['weight'].extend(
                        connection_utils.get_conn_parameter_distribution(mean=weight, part_std=0.7, size=n_connections)
                    )
                    u_values = connection_utils.get_conn_parameter_distribution(mean=syn_dict['U'], part_std=0.5,
                                                                                size=n_connections, maximum=1.)
                    synaptic_parameters['U'].extend(u_values)
                    synaptic_parameters['u'].extend(u_values)
                    synaptic_parameters['delay'].extend(
                        connection_utils.get_conn_parameter_distribution(mean=syn_dict['delay'], part_std=0.1,
                                                                         size=n_connections)
                    )
                    synaptic_parameters['tau_rec'].extend(
                        connection_utils.get_conn_parameter_distribution(mean=syn_dict['tau_rec'], part_std=0.5,
                                                                         size=n_connections)
                    )
                    synaptic_parameters['tau_fac'].extend(
                        connection_utils.get_conn_parameter_distribution(mean=syn_dict['tau_fac'], part_std=0.5,
                                                                         size=n_connections)
                    )

        np.random.shuffle(source_neurons)
        np.random.shuffle(target_neurons)
        for i, (src_neuron, trg_neuron) in enumerate(zip(source_neurons, target_neurons)):
            syn_dict = dict((param_name, param_values[i]) for param_name, param_values in synaptic_parameters.items())
            nest.Connect(src_neuron, trg_neuron, 'one_to_one', syn_spec=syn_dict)

        connection_ids_from_to = {}
        for src_name, src_neurons in self.populations.items():
            if src_name not in connection_ids_from_to.keys():
                connection_ids_from_to[src_name] = {}
            for trg_name, trg_neurons in self.populations.items():
                connection_ids_from_to[src_name][trg_name] = nest.GetConnections(source=src_neurons, target=trg_neurons)

        print(f'{len(source_neurons)} connections have been created.')

        return connection_ids_from_to
