import sys

import nest
import numpy as np
from tqdm import tqdm

from SNN.networks.microcircuit import Microcircuit
from SNN.utils import connection_utils, general_utils


class SmallWorldCircuit(Microcircuit):
    """ Network whose undirected graph has a small-world structure """

    def __init__(self, N=560, S_rw=119.3304, alpha=4., beta=1.32,
                 random_weight_when_unconnected=True,
                 neuron_model=None, neuron_params_exc=None, neuron_params_inh=None, vt_l23exc=None,
                 vt_l23inh=None, vt_l4exc=None, vt_l4inh=None, vt_l5exc=None, vt_l5inh=None):
        self.network_type = 'smallworld'
        self.alpha = alpha
        self.beta = beta
        self.random_weight_when_unconnected = random_weight_when_unconnected
        super().__init__(N=N, S_rw=S_rw, neuron_model=neuron_model, neuron_params_exc=neuron_params_exc,
                         neuron_params_inh=neuron_params_inh,
                         vt_l23exc=vt_l23exc, vt_l23inh=vt_l23inh, vt_l4exc=vt_l4exc,
                         vt_l4inh=vt_l4inh, vt_l5exc=vt_l5exc, vt_l5inh=vt_l5inh)

    def connect_net(self, print_connections=False):
        """ Connects the previously created neurons to create the network

        for a description of the algorithm see Paper by Kaiser and Hilgetag 2004 (DOI: 10.1103/PhysRevE.69.036103)

        Parameters
        ----------
        print_connections: bool
            whether to print out the connections (mainly for debugging)

        Returns
        -------
        dict
            dictionary[source_population][target_population] = all connections from source_population to target_population

        """

        if print_connections:
            print('printing out the connections is not implemented for SmallWorldCircuit!')
        neuron_pop_dicts = self.get_all_neuron_pop_dicts()
        np.random.shuffle(neuron_pop_dicts)

        neuron_positions = {}
        for neuron_dict in neuron_pop_dicts:
            neuron_positions[neuron_dict['neuron'].get('global_id')] = np.random.uniform(low=0., high=1., size=2)

        def get_distance(neuron1, neuron2):
            pos1 = neuron_positions[neuron1.get('global_id')]
            pos2 = neuron_positions[neuron2.get('global_id')]
            distance = np.abs(np.sqrt((pos1[0] - pos2[0]) ** 2 + (pos1[1] - pos2[1]) ** 2))

            return distance

        def get_conn_prob(distance):
            connection_probability = self.beta * np.exp(-self.alpha * distance)

            return connection_probability

        connection_tuples = []
        already_added_neuron_dicts = [neuron_pop_dicts[0]]

        for neuron_to_add_dict in tqdm(neuron_pop_dicts[1:], desc="Creating undirected Small World Network"):
            distances = [get_distance(neuron_to_add_dict['neuron'], added_neuron_dict['neuron']) for added_neuron_dict
                         in already_added_neuron_dicts]
            probabilities = [get_conn_prob(dist) for dist in distances]

            rnd_values = np.random.uniform(low=0., high=1., size=len(already_added_neuron_dicts))
            connect_bools = [rnd_val <= prob for (rnd_val, prob) in zip(rnd_values, probabilities)]
            while not True in connect_bools:
                rnd_values = np.random.uniform(low=0., high=1., size=len(already_added_neuron_dicts))
                connect_bools = [rnd_val <= prob for (rnd_val, prob) in zip(rnd_values, probabilities)]

            for target_neuron_dict, connect in zip(already_added_neuron_dicts, connect_bools):
                if connect:
                    connection_tuples.append((neuron_to_add_dict, target_neuron_dict))

            already_added_neuron_dicts.append(neuron_to_add_dict)

        print(f'{len(connection_tuples)} connections generated', flush=True)

        for i, conn_tupel in tqdm(enumerate(connection_tuples), desc="Transforming to directed connections"):
            switch = np.random.uniform(low=0., high=1., size=1) < 0.5
            if switch:
                connection_tuples[i] = (conn_tupel[1], conn_tupel[0])

        nonzero_weights = {
            'exc': [],
            'inh': [],
        }
        for src_pop, target_dict in self.psp_amp_from_to.items():
            src_type = src_pop[-3:]
            for psp_amp in target_dict.values():
                if psp_amp != 0.:
                    nonzero_weights[src_type].append(psp_amp)

        conns_from_to = {}
        for conn_tuple in connection_tuples:
            source = conn_tuple[0]
            source_id = source['neuron'].get('global_id')
            target = conn_tuple[1]
            if source_id not in conns_from_to.keys():
                conns_from_to[source_id] = {
                    'source_neuron': source['neuron'],
                    'source_pop': source['pop'],
                    'target_neurons': target['neuron'],
                    'target_pops': [target['pop']]
                }
            else:
                conns_from_to[source_id]['target_neurons'] += target['neuron']
                conns_from_to[source_id]['target_pops'].append(target['pop'])

        for source_id, connection_map in conns_from_to.items():
            source_neuron = connection_map['source_neuron']
            source_pop = connection_map['source_pop']
            source_type = source_pop[-3:]
            for target_neuron, target_pop in zip(connection_map['target_neurons'], connection_map['target_pops']):
                target_type = target_pop[-3:]
                syn_dict = self.syn_params_from_to[source_type][target_type]
                syn_weight = self.psp_amp_from_to[source_pop][target_pop]
                if self.random_weight_when_unconnected and syn_weight == 0.:
                    syn_weight = np.random.choice(nonzero_weights[source_type])
                syn_dict['weight'] = connection_utils.calc_synaptic_weight(syn_weight, self.S_rw,
                                                                           source_pop[-3:],
                                                                           self.neuron_pars[
                                                                               source_pop]['g_L'])
                connection_utils.connect_population_pair(source_neuron, target_neuron, syn_dict=syn_dict,
                                                         conn_dict='one_to_one')

        connection_ids_from_to = self.calculate_connection_ids_from_to()

        return connection_ids_from_to
