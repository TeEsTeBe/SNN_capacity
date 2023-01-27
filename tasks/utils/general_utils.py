import os

import numpy as np
import yaml


def setup_seeding(seed):
    if seed is None:
        seed = np.random.randint(2 ** 32 - 1)
    else:
        seed = seed
    np.random.seed(seed)

    return seed


def store_parameter_file(args, runpath):
    with open(os.path.join(runpath, 'parameters.yaml'), 'w') as params_file:
        params = vars(args)
        # params['seed'] = seed
        params['runpath'] = runpath
        yaml.dump(params, params_file)
