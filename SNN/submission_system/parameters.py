import copy
import numpy as np

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

default_microcircuit_hh_neuron_params_exc = {
    'g_L': 15.5862,
    'g_M': 100.,
    'sigma_noise_ex': 3.,
    'sigma_noise_in':6.6,
}

default_microcircuit_hh_neuron_params_inh = {
    'g_L': 15.5862,
    'g_M': 0.,
    'sigma_noise_ex': 3.,
    'sigma_noise_in': 6.6,
}

default_microcircuit_iaf_neuron_params = {
    'C_m': 346.36,  # membrane capacity (pF)
    'E_L': -80.,  # resting membrane potential (mV)
    'I_e': 0.,  # external input current (pA)
    'V_m': -80.,  # membrane potential (mV)
    'V_reset': -80.,  # reset membrane potential after a spike (mV)
    'V_th': -55.,  # spike threshold (mV)
    't_ref': 3.0,  # refractory period (ms)
    'tau_syn_ex': 3.,  # membrane time constant (ms)
    'tau_syn_in': 6.,  # membrane time constant (ms)
    'g_L': 15.5862,
    'E_ex': 0.,
    'E_in': -75.,
}

params = {
    # 'group_name': 'MC-iaf-diff-Vth',
    'group_name': 'uniform-inp_BRN-classification_test',
    'run_title': None,
    # 'trial': list(range(1, 11)),
    # 'trial': [1, 2, 3],
    'trial': [1],
    # 'network_type': ['microcircuit'],
    'network_type': ['brunel'],
    'step_duration': [2., 50.],  # [10., 20.],
    'num_steps': 6000,
    # 'input_type': ['spatial_DC', 'spatial_rate', 'step_DC', 'step_rate', 'spatial_DC_classification', 'spatial_rate_classification'],
    # 'input_type': ['uniform_DC', 'uniform_rate'],
    'input_type': ['uniform_DC_classification', 'uniform_rate_classification'],
    # 'input_type': ['spatial_DC_XOR', 'spatial_DC_XORXOR'],
    # 'input_type': ['spatial_DC', 'spatial_rate'],
    # 'input_min_value': 0.,  # 'input_max_value',
    'input_min_value': '-input_max_value',  # 'input_max_value',
    # 'input_max_value': [0., 1., 10., 100., 200., 500., 1000. ,2000.],
    'input_max_value': [0., 0.04, 0.6, 1000., 3000.],
    # 'input_max_value':[0., 200., 600., 1000., 1600., 2000., 2600., 3000., 4000],
    # 'input_max_value':[ round(x/100,3) for x in [10., 15., 20., 25., 30., 35., 40., 45.]],
    # 'n_spatial_encoder': 1000,
    # 'n_spatial_encoder': 130,
    'n_spatial_encoder': 'n_input_neurons',
    # 'spatial_std_factor': [1, 2, 5, 10, 15, 20],  # [20, 30, 40, 50],
    # 'spatial_std_factor': [1, 20],  # [20, 30, 40, 50],
    'spatial_std_factor': [None],  # [20, 30, 40, 50],
    # 'spatial_std_factor': None,  # [20, 30, 40, 50],
    # 'input_connection_probability': [0.25, 0.5, 0.75, 1.],
    'input_connection_probability': 1.,
    'data_dir': None,
    'dt': 0.1,
    'spike_recorder_duration': 10000.,
    'raster_plot_duration': 1000.,
    # 'background_rate': [None, 3500., 4500.],
    'background_rate': [None],
    'background_weight': None,
    'noise_loop_duration': 'step_duration',
    # 'noise_loop_duration': [None, 'step_duration'],
    'add_ac_current': None,
    'batch_steps': 10000,
    'enable_spike_filtering': False,
    'network_params': {
        # 'N': 560,
        # # 'neuron_model': 'iaf_psc_delta',
        # 'neuron_params_exc': default_microcircuit_iaf_neuron_params.copy(),
        # 'neuron_params_inh': default_microcircuit_iaf_neuron_params.copy(),
        # 'vt_l23exc': -52.,
        # 'vt_l23inh': -55.,
        # 'vt_l4exc': -49.,
        # 'vt_l4inh': -55.,
        # 'vt_l5exc': -57.,
        # 'vt_l5inh': -65.,
        # 'neuron_model': 'hh_cond_exp_destexhe',
        # 'neuron_params_exc': default_microcircuit_hh_neuron_params_exc.copy(),
        # 'neuron_params_inh': default_microcircuit_hh_neuron_params_inh.copy(),
        'neuron_model': 'iaf_cond_exp',
        'N': 1250,
        # 'neuron_params': {
        #     'I_e': {'function': 'np.random.uniform', 'parameters': {'low': 0.95, 'high': 1.05}}
        # }
        # 'g': [5., 12.],
        # 'J': [0.2, float(np.sqrt(1250/12500)*0.1)]
        'g': [5.],
        'J': [0.2]
    },
}

add_to_group_name = ['network_type', 'spatial_std_factor', 'input_max_value', 'step_duration', ['network_params', 'g']]
# add_to_runtitle = ['input_type', 'input_connection_probability', ['network_params', 'J'], 'spatial_std_factor', 'step_duration', 'input_max_value', 'noise_loop_duration']  # [['network_params', 'g'], ['network_params', 'J']]
add_to_runtitle = ['input_type', ['network_params', 'J'], 'noise_loop_duration', 'background_rate']  # [['network_params', 'g'], ['network_params', 'J']]

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
    'noise_loop_duration': 'loop',
    'background_rate': 'bgrate'
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
            current_str += f"__{param_name}={params_dict[param]}"

    return current_str


def derive_parameters(params_dict):
    if add_to_group_name is not None:
        params_dict['group_name'] = append_param_string(params_dict['group_name'], params_dict, add_to_group_name)

    if params_dict['run_title'] is None:
        params_dict['run_title'] = params_dict['group_name']
    if add_to_runtitle is not None:
        params_dict['run_title'] = append_param_string(params_dict['run_title'], params_dict, add_to_runtitle)
    params_dict['run_title'] += f"__{params_dict['trial']}"

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

    if params_dict['noise_loop_duration'] == 'step_duration':
        params_dict['noise_loop_duration'] = params_dict['step_duration']

    if params_dict['input_min_value'] == 'input_max_value':
        params_dict['input_min_value'] = params_dict['input_max_value']
    elif params_dict['input_min_value'] == '-input_max_value':
        params_dict['input_min_value'] = -params_dict['input_max_value']

    return params_dict