import sys
import gc
import numpy as np
import nest

# from utils import general_utils


def create_multimeter(population_list, interval):
    multimeter = nest.Create('multimeter')
    nest.SetStatus(multimeter, {'interval': interval, 'record_from': ['V_m']})
    for pop in population_list:
        nest.Connect(multimeter, pop, syn_spec={'weight': 1., 'delay': 0.1})

    return multimeter


def create_spike_filtering_multimeter(population_list, interval, filter_tau):
    filter_neuron_list = []

    for pop in population_list:
        filter_neurons = nest.Create('iaf_psc_delta', len(pop),
                    params={'C_m': 1., 'E_L': 0., 'V_th': sys.float_info.max, 'V_m': 0., 'V_reset': 0., 'V_min': 0.,
                            'tau_m': filter_tau, 'refractory_input': True})
        conn_dict = {"rule": "one_to_one"}
        syn_dict = {"synapse_model": "static_synapse", "delay": 0.1}
        nest.Connect(pop, filter_neurons, conn_dict, syn_dict)
        filter_neuron_list.append(filter_neurons)

    return create_multimeter(filter_neuron_list, interval=interval)


def create_spike_recorder(population_list, start=None, stop=None):
    """ Creates spike recorders that record from the given population list

    Parameters
    ----------
    population_list
        populations to record from
    start: float
        recording start time in ms
    stop
        recording stop time in ms

    Returns
    -------
    the generated nest spike recorders

    """
    spike_recorder = nest.Create('spike_recorder')
    if start is not None:
        nest.SetStatus(spike_recorder, {'start': start})
    if stop is not None:
        nest.SetStatus(spike_recorder, {'stop': stop})

    for pop in population_list:
        nest.Connect(pop, spike_recorder)

    return spike_recorder


def get_statematrix(multimeter):
    multimeter_status = nest.GetStatus(multimeter)[0]['events']

    senders = multimeter_status['senders']
    unique_senders = np.unique(senders)
    n_senders = unique_senders.size
    n_steps = int(senders.size / n_senders)

    vms = multimeter_status['V_m']

    statemat = np.empty((n_senders, n_steps))
    for column_id, sender_id in enumerate(unique_senders):
        statemat[column_id, :] = vms[senders == sender_id]

    del multimeter_status
    del senders
    del vms
    gc.collect()
    # statematx = general_utils.order_array_by_ids(array_to_order=vms, n_possible_ids=n_senders, ids=senders)

    return statemat
