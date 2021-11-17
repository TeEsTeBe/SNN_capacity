import os
import itertools
import numpy as np
import yaml
from parameters import params, derive_parameters
from utils import general_utils


def get_dict_combinations(dictionary):
    """
    Transforms a dictionary with lists as dict values into a list of all combinations of the list items.
    Be aware that all values of the dicts will be transformed to str.

    Parameters
    ----------
    dictionary: dict
        dictionary with list as values

    Returns
    -------
    list
        list of dictionaries with all combinations

    Examples
    --------
    >>> mydict = {'a': [0, 1], 'b': [2, 3], 'c': ['xyz']}
    >>> get_dict_combinations(mydict)
    [{'a': '0', 'b': '2', 'c': 'xyz'}, {'a': '0', 'b': '3', 'c': 'xyz'}, {'a': '1', 'b': '2', 'c': 'xyz'}, {'a': '1', 'b': '3', 'c': 'xyz'}]


    """

    par_keys = dictionary.keys()
    par_vals = [v if isinstance(v, list) else [v] for v in list(dictionary.values())]
    value_combinations = list(itertools.product(*par_vals))
    dict_combinations = []
    for values in value_combinations:
        d = {}
        for idx, key in enumerate(par_keys):
            if key == 'seed' and values[idx] == 'random':
                d[key] = np.random.randint(999999)
            else:
                d[key] = values[idx]
        dict_combinations.append(d)

    return dict_combinations


def create_dict_list_recursively(dict_to_search):
    for key, value in dict_to_search.items():
        if isinstance(value, dict):
            dict_to_search[key] = create_dict_list_recursively(value)

    return get_dict_combinations(dict_to_search)


if __name__ == '__main__':
    parameter_list = create_dict_list_recursively(params)
    parameter_list = [derive_parameters(param_dict) for param_dict in parameter_list]

    params_dir = general_utils.get_paramfiles_dir()
    for param_dict in parameter_list:
        group_params_dir = os.path.join(params_dir, param_dict['group_name'])
        os.makedirs(group_params_dir, exist_ok=True)
        param_file_path = os.path.join(group_params_dir, f"{param_dict['run_title']}.yaml")
        with open(param_file_path, 'w') as param_file:
            yaml.safe_dump(param_dict, param_file, default_flow_style=False)
