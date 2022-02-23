import numpy as np
import networkx as nx
from scipy.stats import truncnorm
import nest


def create_synapse_parameters(synapse_model='tsodyks2_synapse'):
    # syn_name = 'tsodyks2_synapse'
    # syn_name = 'tsodyks_synapse'

    if synapse_model == 'static_synapse':
        syn_params_from_to = {
            'exc': {
                'exc': {'synapse_model': synapse_model, 'delay': 1.5},
                'inh': {'synapse_model': synapse_model, 'delay': 0.8},
            },
            'inh': {
                'exc': {'synapse_model': synapse_model, 'delay': 0.8},
                'inh': {'synapse_model': synapse_model, 'delay': 0.8},
            },
        }
    else:
        syn_params_from_to = {
            'exc': {
                'exc': {'synapse_model': synapse_model, 'U': 0.5, 'tau_rec': 1.1e3, 'tau_fac': 0.05e3, 'delay': 1.5},
                'inh': {'synapse_model': synapse_model, 'U': 0.05, 'tau_rec': 0.125e3, 'tau_fac': 1.200e3, 'delay': 0.8},
            },
            'inh': {
                'exc': {'synapse_model': synapse_model, 'U': 0.25, 'tau_rec': 0.7e3, 'tau_fac': 0.02e3, 'delay': 0.8},
                'inh': {'synapse_model': synapse_model, 'U': 0.32, 'tau_rec': 0.144e3, 'tau_fac': 0.06e3, 'delay': 0.8},
            },
        }

        # add missing tau_psc parameters
        if synapse_model == 'tsodyks_synapse':
            for source, syn_params in syn_params_from_to.items():
                for target, param_dict in syn_params.items():
                    if source == 'exc':
                        syn_params_from_to[source][target]['tau_psc'] = 3.
                    elif source == 'inh':
                        syn_params_from_to[source][target]['tau_psc'] = 6.
        # add missing u parameters
        elif synapse_model == 'tsodyks2_synapse':
            for source, syn_params in syn_params_from_to.items():
                for target, param_dict in syn_params.items():
                    syn_params_from_to[source][target]['u'] = param_dict['U']

    return syn_params_from_to


def calc_synaptic_weight(psp_amp, scaling_factor, exc_or_inh_src, g_L):
    exc_codes = ['e', 'exc', 'ex', 'excitatory']
    inh_codes = ['i', 'inh', 'in', 'inhibitory']
    if exc_or_inh_src in exc_codes:
        src_type = 'exc'
    elif exc_or_inh_src in inh_codes:
        src_type = 'inh'
    else:
        raise ValueError(
            f"Can't determine whether the source population is excitatory or inhibitory based on string '{exc_or_inh_src}'."
            f"Possible for excitatory sources: {exc_codes}. "
            f"Possible for inhibitory sources: {inh_codes}. ")

    # same as division by Rm (see matlab code)
    weight = psp_amp * g_L

    # see matlab code make_network_V1_HH, line 247
    E_syn = 0. if src_type == 'exc' else -75.
    V = -65.
    weight /= abs(E_syn - V)  # (mV * nS) / mV -> nS
    weight *= scaling_factor

    return weight


def get_conn_parameter_distribution(mean, part_std, size, minimum=0., maximum=2**32, use_truncnorm=False):
    if use_truncnorm:
        a = (minimum - mean) / abs(0.7 * mean)
        b = maximum
        distribution = truncnorm.rvs(a, b, size=size, loc=mean, scale=part_std*abs(mean))
    else:
        distribution = np.random.normal(mean, part_std, size=size)
        mask = (distribution < minimum) | (distribution > maximum)
        n_uniform = len(distribution[mask])
        # print(f'{param}, mean: {mean}, min: {minimum}, len(uniform): {n_uniform}')
        distribution[mask] = np.random.uniform(minimum, min(maximum, minimum+2*mean), size=n_uniform)

    return distribution


def randomize_conn_parameter(source, target, param, mean, part_std, minimum=0., maximum=2**32, use_truncnorm=False):
    conn = nest.GetConnections(source=source, target=target)
    distribution = get_conn_parameter_distribution(mean=mean, part_std=part_std, size=len(conn), minimum=minimum,
                                                   maximum=maximum, use_truncnorm=use_truncnorm)

    nest.SetStatus(conn, param, distribution)


def connect_population_pair(source_ids, target_ids, syn_dict, conn_dict, static_synapses=False):
    # TODO: rename to connect_population_pair_with_randomization
    # TODO: docstring

    connections = nest.Connect(source_ids, target_ids, conn_spec=conn_dict, syn_spec=syn_dict)
    randomize_conn_parameter(source_ids, target_ids, 'weight', syn_dict['weight'], 0.7)
    randomize_conn_parameter(source_ids, target_ids, 'delay', syn_dict['delay'],
                                              0.1)  # from SH_delay in matlab conn.parameters

    if not static_synapses:
        randomize_conn_parameter(source_ids, target_ids, 'U', syn_dict['U'], 0.5, maximum=1.)

        # set u1 to U for every connection
        connections = nest.GetConnections(source=source_ids, target=target_ids)
        conn_states = nest.GetStatus(connections)
        new_u_values = []
        for cs in conn_states:
            new_u_values.append({'u': cs['U']})
        nest.SetStatus(connections, new_u_values)

        randomize_conn_parameter(source_ids, target_ids, 'tau_fac', syn_dict['tau_fac'], 0.5)
        randomize_conn_parameter(source_ids, target_ids, 'tau_rec', syn_dict['tau_rec'], 0.5)

    return connections


def get_in_and_outdegrees(neuron_nodecollection):
    outdegrees = []
    indegrees = []
    for neuron in neuron_nodecollection:

        # I have no idea, why this doesn't work for the degree controlled circuit w/o input and output specificity
        # after scrambling the populations the following line gives back the connections for all neuron_ids and not only
        # the ones of the single neuron.
        # Somehow the line after this works fine.
        # out_neurons = nest.GetConnections(source=neuron)

        out_neurons = nest.GetConnections(source=nest.NodeCollection([neuron.global_id]))
        in_neurons = nest.GetConnections(target=neuron)
        outdegrees.append(len(out_neurons))
        indegrees.append(len(in_neurons))

    return indegrees, outdegrees


def get_total_degrees_per_pop(network):
    degrees_per_pop = {}
    for pop_name, pop_neurons in network.populations.items():
        indegrees, outdegrees = get_in_and_outdegrees(pop_neurons)
        degrees_combined = [indeg + outdeg for (indeg, outdeg) in zip(indegrees, outdegrees)]
        if pop_name not in degrees_per_pop.keys():
            degrees_per_pop[pop_name] = degrees_combined
        else:
            degrees_per_pop[pop_name] += degrees_combined

    return degrees_per_pop


def get_network_graph(network):
    network_neurons = nest.NodeCollection([])
    for neurons in network.populations.values():
        network_neurons += neurons
    network_connections = nest.GetConnections(source=network_neurons, target=network_neurons)

    edges = [(connection.source, connection.target) for connection in network_connections]
    network_graph = nx.Graph()
    network_graph.add_edges_from(edges)

    return network_graph


def conductances_to_psp(weights, g_L):

    E_syn_exc = 0.
    E_syn_inh = -75
    V = -65.

    weights[weights>0] *= abs(E_syn_exc - V)
    weights[weights<0] *= abs(E_syn_inh - V)
    weights /= g_L

    return weights


def get_input_data_per_neuron(network):
    input_data = {}
    for popname, neurons in network.populations.items():
        input_data[popname] = {}
        for neuron in neurons:
            incoming_connections = nest.GetConnections(target=neuron)
            conn_status = nest.GetStatus(incoming_connections)

            sources = [conn['source'] for conn in conn_status]
            weights = [conn['weight'] for conn in conn_status]
            u_values = [conn['u'] if 'u' in conn.keys() else 1. for conn in conn_status]
            x_values = [conn['x'] if 'x' in conn.keys() else 1. for conn in conn_status]
            u_values = [u if u is not None else 1. for u in u_values]
            x_values = [x if x is not None else 1. for x in x_values]
            A_values = [w * u * x for w, u, x in zip(weights, u_values, x_values)]
            psp_values = conductances_to_psp(np.array(A_values), network.neuron_pars[popname]['g_L'])

            input_data[popname][neuron.global_id] = {
                'source_ids': sources,
                'weights': weights,
                'psp': psp_values,
                'u': u_values,
                'x': x_values,
                'A': A_values
            }

    return input_data
