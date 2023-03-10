import numpy as np
import nest
import time
from tqdm import tqdm

from SNN.networks.microcircuit import Microcircuit
from SNN.utils import connection_utils


class AmorphousCircuit(Microcircuit):
    """ Microcircuit model with scrambled connections """

    def __init__(self, N=560, S_rw=119.3304, neuron_model=None, neuron_params_exc=None, neuron_params_inh=None, vt_l23exc=None,
                 vt_l23inh=None, vt_l4exc=None, vt_l4inh=None, vt_l5exc=None, vt_l5inh=None):
        super().__init__(N=N, S_rw=S_rw,
                         neuron_model=neuron_model, neuron_params_exc=neuron_params_exc, neuron_params_inh=neuron_params_inh,
                         vt_l23exc=vt_l23exc, vt_l23inh=vt_l23inh, vt_l4exc=vt_l4exc,
                         vt_l4inh=vt_l4inh, vt_l5exc=vt_l5exc, vt_l5inh=vt_l5inh)
        self.network_type = 'amorphous'

    def connect_net(self, print_connections=False):
        if print_connections:
            print(f'Print out of connections is not implemented for amorhous circuit!')

        syn_dicts = {
            'exc_exc': [],
            'exc_inh': [],
            'inh_exc': [],
            'inh_inh': [],
        }

        n_total_connections = 0

        print('caculating syn_dicts ...')
        syn_calc_start = time.time()
        for (src_pop, trg_w_dict), (_, trg_probs) in zip(self.psp_amp_from_to.items(),
                                                         self.probabilities_from_to.items()):
            src_pop_type = src_pop[-3:]

            for (trg_pop, weight), (_, probability) in tqdm(zip(trg_w_dict.items(), trg_probs.items()),
                                                            desc=f'create connectivity for {src_pop} as source',
                                                            total=len(trg_probs.values()), disable=True):
                if probability > 0.:
                    trg_pop_type = trg_pop[-3:]

                    syn_dict_template = self.syn_params_from_to[src_pop[-3:]][trg_pop[-3:]].copy()
                    mean_weight = connection_utils.calc_synaptic_weight(
                        psp_amp=weight,
                        scaling_factor=self.S_rw,
                        exc_or_inh_src=src_pop[-3:],
                        g_L=self.neuron_pars[src_pop]['g_L']
                    )

                    n_possible_connections = len(list(self.populations[src_pop])) * len(list(self.populations[trg_pop]))
                    random_numbers = np.random.uniform(low=0., high=1., size=n_possible_connections)
                    n_connections = random_numbers[random_numbers <= probability].size
                    n_total_connections += n_connections

                    syn_models = ['tsodyks2_synapse' for _ in range(n_connections)]
                    weights = connection_utils.get_conn_parameter_distribution(mean=mean_weight, part_std=0.7,
                                                                               size=n_connections)
                    u_values = connection_utils.get_conn_parameter_distribution(mean=syn_dict_template['U'],
                                                                                part_std=0.5, size=n_connections,
                                                                                maximum=1.)
                    delays = connection_utils.get_conn_parameter_distribution(mean=syn_dict_template['delay'],
                                                                              part_std=0.1, size=n_connections)
                    tau_recs = connection_utils.get_conn_parameter_distribution(mean=syn_dict_template['tau_rec'],
                                                                                part_std=0.5, size=n_connections)
                    tau_facs = connection_utils.get_conn_parameter_distribution(mean=syn_dict_template['tau_fac'],
                                                                                part_std=0.5, size=n_connections)

                    for i in range(n_connections):
                        syn_dict = syn_dict_template.copy()
                        syn_dict['synapse_model'] = syn_models[i]
                        syn_dict['weight'] = weights[i]
                        syn_dict['U'] = u_values[i]
                        syn_dict['u'] = u_values[i]
                        syn_dict['delay'] = delays[i]
                        syn_dict['tau_rec'] = tau_recs[i]
                        syn_dict['tau_fac'] = tau_facs[i]

                        syn_dicts[f'{src_pop_type}_{trg_pop_type}'].append(syn_dict)
        syn_calc_time = time.time() - syn_calc_start
        print(
            f'Syn dict calculation time: {syn_calc_time} seconds ({round(syn_calc_time / 60, 2)} minutes; {round(syn_calc_time / 60 / 60, 4)} hours)')

        print('Connecting neurons ...')
        conn_start = time.time()
        self.connect_neurons_randomly_new(syn_dicts=syn_dicts)
        conn_time = time.time() - conn_start
        print(
            f'Connection time: {conn_time} seconds ({round(conn_time / 60, 2)} minutes; {round(conn_time / 60 / 60, 4)} hours)')

        print(f'{n_total_connections} connections have been created.')

        print('calculating connection ids ...')
        connection_ids_from_to = self.calculate_connection_ids_from_to()

        return connection_ids_from_to

    def connect_neurons_randomly_new(self, syn_dicts):
        """ Scrambles the data-based connectivity structure

        Parameters
        ----------
        syn_dicts: dict
            dictionary with all connections

        """

        exc_inh_neuron_dict = self.get_neurons_separated_by_exc_inh()
        exc_inh_ids = {
            'exc': exc_inh_neuron_dict['exc'].global_id,
            'inh': exc_inh_neuron_dict['inh'].global_id,
        }
        for syn_type, syn_parameter_list in tqdm(syn_dicts.items(), desc=f'connect populations',
                                                 total=len(syn_dicts.keys()), disable=True):
            syn_parameter_array = np.array(syn_parameter_list)
            n_connections = len(syn_parameter_list)
            src_type = syn_type[:3]
            trg_type = syn_type[-3:]
            new_src_ids = np.sort(np.random.choice(exc_inh_ids[src_type], size=n_connections, replace=True))

            # can't just connect all new_src_ids with all new_trg_ids, because the ids in a node collection have to be unique ...
            unique_src_ids, src_id_counts = np.unique(new_src_ids, return_counts=True)
            for src_id, syn_count in zip(unique_src_ids, src_id_counts):
                trg_ids = np.sort(np.random.choice(exc_inh_ids[trg_type], size=syn_count, replace=False))

                for trg_id, syn_dict in zip(trg_ids, syn_parameter_array):
                    nest.Connect(nest.NodeCollection([src_id]), nest.NodeCollection([trg_id]), 'one_to_one', syn_dict)
