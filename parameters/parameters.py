
params = {
    'group_name': 'test',
    'run_title': None,
    'trial': list(range(1, 11)),
    'network_type': ['brunel', 'alzheimers'],
    'step_duration': 20.,
    'num_steps': 100,
    'input_type': ['step_rates', 'step_DC', 'spatial_rates', 'spatial_DC'],
    'input_min': 0.,
    'input_max': 1.,
    'n_spatial_encoder': 100,
    'spatial_std_factor': 20,
    'input_connection_probability': 0.25,
    'network_params': {
        'N': 1250,
        'g': [5., 6.],
        'J': [0.2, 0.3, 0.4]
    }
}

add_to_group_name = ['network_type', 'input_type', ['network_params', 'N']]
add_to_runtitle = [['network_params', 'g'], ['network_params', 'J']]

shortened_params_dict = {
    'network_type': 'net',
    'step_duration': 'dur',
    'num_steps': 'steps',
    'input_type': 'inp',
    'input_min': 'min',
    'input_max': 'max',
    'n_spatial_encoder': 'nenc',
    'spatial_std_factor': 'std',
    'input_connection_probability': 'p',
}


def append_param_string(current_str, params_dict, params_to_add):
    for param in params_to_add:
        if isinstance(param, list):
            current_str = append_param_string(current_str, params_dict[param[0]], param[1:])
        else:
            if param in shortened_params_dict.keys():
                param_name = shortened_params_dict[param]
            else:
                param_name = param
            current_str += f"_{param_name}={params_dict[param]}"

    return current_str


def derive_parameters(params_dict):
    if add_to_group_name is not None:
        params_dict['group_name'] = append_param_string(params_dict['group_name'], params_dict, add_to_group_name)

    if params_dict['run_title'] is None:
        params_dict['run_title'] = params_dict['group_name']
    if add_to_runtitle is not None:
        params_dict['run_title'] = append_param_string(params_dict['run_title'], params_dict, add_to_runtitle)
    params_dict['run_title'] += f"_{params_dict['trial']}"

    return params_dict