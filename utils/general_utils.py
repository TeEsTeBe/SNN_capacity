import os
import numpy as np
import nest
from pathlib import Path

from fna.tools.signals import SpikeList


def get_data_dir():
    return os.path.join(Path(__file__).parent.parent.resolve(), 'data')


def get_paramfiles_dir():
    return os.path.join(Path(__file__).parent.parent.resolve(), 'parameter_files')


def spikelist_from_recorder(spikedetector):
    detector_status = nest.GetStatus(spikedetector)[0]['events']
    senders = detector_status['senders']
    times = detector_status['times']
    spikes = [(neuron_id, spike_time) for neuron_id, spike_time in zip(senders, times)]
    spikelist = SpikeList(spikes, np.unique(senders))

    return spikelist


def order_array_by_ids(array_to_order, n_possible_ids, ids):
    """
    Orders an array (for example spike trains of neurons) by the given ids (of the neurons).
    Needs the number of possible (neuron) ids, because some ids could be missing (neurons may not have
    fired), but they should be in the resulting list as well.

    Parameters
    ----------
    array_to_order: ndarray of floats
        ndarray with spike times
    n_possible_ids: int
        number of possible ids
    ids: ndarray of int
        ids of the objects to which the elements in the array_to_order belong

    Returns
    -------
    ndarray
        spike trains (ndarrays) for each neuron

    Examples
    --------
    >>> spike_times = np.array([10.2, 20.1, 30.1])
    >>> ids = np.array([2, 1, 1])
    >>> order_array_by_ids(spike_times, 3, ids)
    [array([20.1, 30.1]), array([10.2]), array([], dtype=float64)]
    """

    if len(array_to_order) == 0:
        print('Array to order is empty!')
        return None
    else:
        spk_times_list = [np.array([]) for _ in range(n_possible_ids)]
        neurons = np.unique(ids)
        new_ids = ids - min(ids)

        for i, n in enumerate(neurons):
            idx = np.where(ids == n)[0]
            spk_times_list[new_ids[idx[0]]] = array_to_order[idx]

        spk_times_array = np.array(spk_times_list)
        # spk_times_array = np.hstack((spk_times_array, spk_times_array[:, -1]))

        return spk_times_array


def combine_nodelists(list_of_nodelists):
    combined_nodelist = list_of_nodelists[0]
    for nodelist in list_of_nodelists[1:]:
        combined_nodelist += nodelist

    return combined_nodelist
