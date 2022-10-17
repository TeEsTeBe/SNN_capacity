import os
import re
import psutil
import pickle
from pathlib import Path
from hashlib import sha1
from copy import deepcopy

import numpy as np
import nest

from fna.tools.signals import SpikeList


def get_default_data_dir():
    return os.path.join(Path(__file__).parent.parent.resolve(), 'data')


def get_paramfiles_dir():
    return os.path.join(Path(__file__).parent.parent.resolve(), 'parameter_files')


def print_memory_consumption(message, logger):
    memory_mb = psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2
    logger.info(f"{message}: {memory_mb} MB")


def spikelist_from_recorder(spikedetector, stop=None, start=None):
    detector_status = nest.GetStatus(spikedetector)[0]['events']
    senders = detector_status['senders']
    times = detector_status['times']
    if stop is not None:
        mask = times <= stop
        times = times[mask]
        senders = senders[mask]
    if start is not None:
        mask = times >= start
        times = times[mask]
        senders = senders[mask]
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


def filter_paths(paths, params_to_filter, other_filter_keys=None):
    filtered_paths = paths

    for paramname, paramvalue in params_to_filter.items():
        filtered_paths = [p for p in paths if f'{paramname}={paramvalue}_' in p]

    if other_filter_keys is not None:
        for filter_key in other_filter_keys:
            filtered_paths = [p for p in filtered_paths if filter_key in p]

    return filtered_paths


def translate(param_name):
    translation_dict = {
        'inpscaling': r'$\iota$',
        'specrad': r'$\rho$',
        'dur': r'$\Delta s$',
        'max': r'$\mathrm{a_{max}}$'
    }

    if param_name in translation_dict.keys():
        translation = translation_dict[param_name]
    else:
        translation = param_name

    return translation


def default_cast_function(value):
    if value.isnumeric():
        casted_value = int(value)
    else:
        try:
            casted_value = float(value)
        except:
            casted_value = str(value)

    return casted_value


def get_param_from_path(pathstr, param_name, cast_function=default_cast_function):
    re_result = re.search(f'_{param_name}=([^_^\/^-]+)_', pathstr)
    if not re_result:
        re_result = re.search(f'_{param_name}=([^_^\/^-]+).pkl', pathstr)
    if not re_result:
        raise ValueError(f'parameter {param_name} was not found in path "{pathstr}"')

    return cast_function(re_result.group(1))


def cached_data_path():
    base_path = os.path.dirname(os.path.abspath(__file__))
    cached_folder = os.path.join(base_path, 'cached_data')
    os.makedirs(cached_folder, exist_ok=True)

    return cached_folder


def _remove_memory_address(string):
    return re.sub('at 0x.+>', '', string)


def get_cached_filepath(locals_):
    parameters = deepcopy(locals_)
    for key in parameters.keys():
        if key.startswith('_'):
            del parameters[key]
    param_keys = sorted(list(parameters.keys()))
    param_string = ''
    for pkey in param_keys:
        value_str = _remove_memory_address(f'{parameters[pkey]}')
        param_string += f'{pkey}-{value_str}'
    filename = f"{sha1(param_string.encode('utf-8')).hexdigest()}.pkl"
    filepath = os.path.join(cached_data_path(), filename)

    return filepath


def get_cached_file(locals_):
    filepath = get_cached_filepath(locals_)
    if os.path.exists(filepath):
        with open(filepath, 'rb') as f:
            cached_data = pickle.load(f)
    else:
        cached_data = None

    return cached_data


def store_cached_file(data, locals_):
    filepath = get_cached_filepath(locals_)
    with open(filepath, 'wb') as f:
        pickle.dump(data, f)
