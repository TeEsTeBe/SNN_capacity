import copy

default_alzheimer_neuron_params = {
    'C_m': 250.,
    'E_L': 0.0,
    'V_reset': 0.0,
    'V_th': 15.,
    't_ref': 2.,
    'tau_syn_ex': 2.,
    'tau_syn_in': 2.,
    'tau_m': 20.,
}
default_brunel_neuron_params = {
    'C_m': 1.0,  # membrane capacity (pF)
    'E_L': 0.,  # resting membrane potential (mV)
    'I_e': 0.,  # external input current (pA)
    'V_m': 0.,  # membrane potential (mV)
    'V_reset': 10.,  # reset membrane potential after a spike (mV)
    'V_th': 20.,  # spike threshold (mV)
    't_ref': 2.0,  # refractory period (ms)
    'tau_m': 20.,  # membrane time constant (ms)
}

params = {
    'group_name': 'test',
    'run_title': None,
    'trial': list(range(1, 11)),
    'network_type': ['brunel', 'alzheimers'],
    'step_duration': 20.,
    'num_steps': 500,
    'input_type': ['step_DC', 'spatial_DC'],
    'input_min_value': 0.,
    'input_max_value': 1.,
    'n_spatial_encoder': 1000,
    'spatial_std_factor': 20,
    'input_connection_probability': 0.25,
    'data_dir': None,
    'dt': 0.1,
    'spike_recorder_duration': 10000.,
    'raster_plot_duration': 1000.,
    'network_params': {
        'N': 1250,
        'neuron_params': {
            'C_m': 11.
        }
        # 'g': [5.],
        # 'J': [0.2]
    }
}

add_to_group_name = ['network_type', 'input_type', ['network_params', 'N']]
add_to_runtitle = []  # [['network_params', 'g'], ['network_params', 'J']]

shortened_params_dict = {
    'network_type': 'net',
    'step_duration': 'dur',
    'num_steps': 'steps',
    'input_type': 'inp',
    'input_min_value': 'min',
    'input_max_value': 'max',
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

    if params_dict['network_type'] == 'brunel':
        if 'neuron_params' not in params_dict['network_params'].keys():
            params_dict['network_params']['neuron_params'] = copy.deepcopy(default_brunel_neuron_params)
        else:
            for key, value in default_brunel_neuron_params.items():
                if key not in params_dict['network_params']['neuron_params'].keys():
                    params_dict['network_params']['neuron_params'][key] = value
        params_dict['network_params']['neuron_model'] = 'iaf_psc_delta'
    elif params_dict['network_type'] == 'alzheimers':
        if 'neuron_params' not in params_dict['network_params'].keys():
            params_dict['network_params']['neuron_params'] = copy.deepcopy(default_alzheimer_neuron_params)
        else:
            for key, value in default_alzheimer_neuron_params.items():
                if key not in params_dict['network_params']['neuron_params'].keys():
                    params_dict['network_params']['neuron_params'][key] = value
        params_dict['network_params']['neuron_model'] = 'iaf_psc_exp_ps'

    return params_dict