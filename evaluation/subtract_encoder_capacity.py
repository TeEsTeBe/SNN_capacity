import pickle
import argparse
import matplotlib.pyplot as plt
from collections import defaultdict

import numpy as np

from plotting.cap_bars_single_run import plot_capacity_bars
from plotting.colors import get_degree_color


def to_alpha(squared_corr_coeff, var_ratio):
    squared_corr_coeff = min(squared_corr_coeff, 1)
    if squared_corr_coeff == 0:
        return 0
    else:
        # print(squared_corr_coeff)
        return 1 / (np.sqrt((1 / squared_corr_coeff - 1) / var_ratio) + 1)
        # return 1 / (np.sqrt(1 / squared_corr_coeff - 1) + 1)


def to_squared_corr(alpha, var_ratio):
    if alpha == 0.:
        return 0
    else:
        return 1 / (1 + var_ratio * (1 - alpha) ** 2 / alpha ** 2)


def plot_capacity_figure(combined_capacities: dict, encoder_capacities: dict, remembered_enc_capacities: dict,
                         subtract_capacities: dict, network_capacities: dict, linear_network_memory: dict):
    cap_dict_list = [combined_capacities, encoder_capacities, remembered_enc_capacities, subtract_capacities,
                     network_capacities]
    titles = ['combined', 'encoder', 'remembered', 'subtract', 'network']
    fig, axes = plt.subplots(nrows=len(titles) + 1, sharex='all', figsize=((5.2, 7.2)))

    delays = [d - 1 for d in linear_network_memory.keys()]
    axes[0].bar(delays, linear_network_memory.values(), color=get_degree_color(1))
    axes[0].set_title('network memory')

    for ax, capacities, title in zip(axes[1:], cap_dict_list, titles):
        plot_capacity_bars(capacity_dict=capacities, ax=ax)
        ax.set_title(title)

    return fig, axes


def delay_window_capdict_to_default_capdictlist_dummy(delay_window_capdict: dict) -> list:
    cap_dict_list_dummy = []
    for delay, window_to_score_dict in delay_window_capdict.items():
        for window_str, score in window_to_score_dict.items():
            # print(f'delay: {delay}, window: {window_str}')
            # window = [float(x) for x in window_str[1:-1].split(',')]
            # print(window_str)
            # window = [float(x) for x in window_str[1:-1].split(' ')]
            window = [float(x) for x in window_str[1:-1].split()]
            degree = np.sum(window)
            powerlist = [x for x in window if x > 0.]
            window_positions = [i for i, deg in enumerate(window) if deg > 0.]

            cap_dict_list_dummy.append({
                'delay': int(delay),
                'score': score,
                'degree': int(degree),
                'window': len(window),
                'powerlist': powerlist,
                'window_positions': window_positions,
            })

    return cap_dict_list_dummy


def get_window_values(single_capacity: dict) -> np.ndarray:
    poly_definition = np.zeros(single_capacity['window'])
    for pos, power in zip(single_capacity['window_positions'], single_capacity['powerlist']):
        poly_definition[pos] = power

    return poly_definition


def subtract_capacities(combined_capacities: list, capacities_to_subtract: dict, alpha_var_ratio: float,
                        subtract_undelayed_degree1=False, transform_modus='identity'):
    result_capacities = []

    f_T = get_transform_function(transform_modus=transform_modus, var_ratio=alpha_var_ratio)
    inverse_f_T = get_inverse_transform_function(transform_modus=transform_modus, var_ratio=alpha_var_ratio)

    for cap in combined_capacities:
        result_cap = cap.copy()
        # if cap['degree'] > 1:  # we don't want to remove the linear memory of the system
        if subtract_undelayed_degree1 or not (cap['degree'] == 1 and cap['delay'] == 1):
            try:
                window_values = get_window_values(cap)
                subtraction_value = capacities_to_subtract[cap['delay']][str(window_values)]
                # subtraction_value = capacities_to_subtract[cap['delay']][str(cap['powerlist'])]
            except:
                subtraction_value = 0.

            result_cap['score'] = inverse_f_T(max(0, f_T(result_cap['score']) - f_T(subtraction_value)))

            # if transform_modus in ['identity', 'remove_full']:
            #     result_cap['score'] = max(0, result_cap['score'] - subtraction_value)
            # elif transform_modus == 'sqrt':
            #     transformed_score = np.sqrt(result_cap['score'])
            #     transformed_subtraction = np.sqrt(subtraction_value)
            #     result_cap['score'] = max(0, (transformed_score - transformed_subtraction) ** 2)
            # # result_cap['score'] = max(0, (np.sqrt(result_cap['score']) - np.sqrt(subtraction_value))**2)
            # elif transform_modus == 'alpha':
            #     transformed_score = to_alpha(result_cap['score'], var_ratio=alpha_var_ratio)
            #     transformed_subtraction = to_alpha(subtraction_value, var_ratio=alpha_var_ratio)
            #     result_cap['score'] = to_squared_corr(transformed_score - transformed_subtraction,
            #                                           var_ratio=alpha_var_ratio)
            #     result_cap['score'] = max(0, result_cap['score'])
            # else:
            #     raise ValueError(f'Unknown transformation modus "{transform_modus}"')

        if result_cap['score'] > 0.:
            result_capacities.append(result_cap)

    return result_capacities


def get_linear_memory(capacities: list, use_all_delays=True) -> dict:
    linear_memory = defaultdict(int)

    for cap in capacities:
        delay = cap['delay']
        if cap['degree'] == 1:
            linear_memory[delay] = cap['score']
        elif delay not in linear_memory.keys() and use_all_delays:
            linear_memory[delay] = 0

    return linear_memory


def get_linear_network_memory(combined_linear_memory: dict, encoder_linear_memory: dict, transform_modus: str,
                              alpha_var_ratio: float) -> dict:
    network_linear_memory = {}
    for delay, combined_mem_val in combined_linear_memory.items():
        if delay in encoder_linear_memory.keys():
            encoder_mem_val = encoder_linear_memory[delay]
        else:
            encoder_mem_val = 0.

        f_T = get_transform_function(transform_modus=transform_modus, var_ratio=alpha_var_ratio)
        inverse_f_T = get_inverse_transform_function(transform_modus=transform_modus, var_ratio=alpha_var_ratio)
        network_linear_memory[delay] = inverse_f_T(max(0, f_T(combined_mem_val) - f_T(encoder_mem_val)))

        # if transform_modus in ['identity', 'remove_full']:
        #     network_linear_memory[delay] = max(0, combined_mem_val - encoder_mem_val)
        # elif transform_modus == 'alpha':
        #     network_linear_memory[delay] = max(0,
        #                                        to_squared_corr(
        #                                            to_alpha(combined_mem_val, var_ratio=alpha_var_ratio) - to_alpha(
        #                                                encoder_mem_val, var_ratio=alpha_var_ratio),
        #                                            var_ratio=alpha_var_ratio))
        # elif transform_modus == 'sqrt':
        #     network_linear_memory[delay] = max(0, (np.sqrt(combined_mem_val) - np.sqrt(encoder_mem_val)) ** 2)
        # else:
        #     raise ValueError(f'Unknown transfom_modus "{transform_modus}"')

    return network_linear_memory


def get_transform_function(transform_modus: str, var_ratio=None):
    if transform_modus == 'alpha':
        return lambda x: to_alpha(x, var_ratio=var_ratio)
    elif transform_modus == 'sqrt':
        return np.sqrt
    elif transform_modus == 'identity':
        return lambda x: x
    else:
        raise NotImplementedError('Transformation function is not implemented')


def get_inverse_transform_function(transform_modus: str, var_ratio=None):
    if transform_modus == 'alpha':
        return lambda x: to_squared_corr(x, var_ratio=var_ratio)
    elif transform_modus == 'sqrt':
        return lambda x: x ** 2
    elif transform_modus == 'identity':
        return lambda x: x
    else:
        raise NotImplementedError('Transformation function is not implemented')


def get_remembered_encoder_capacities(encoder_capacities: list, network_linear_memory: dict, alpha_var_ratio: float,
                                      transform_modus='identity', fixed_memratio=None, max_delay=None) -> dict:

    linear_encoder_delay0_memory = get_linear_memory(encoder_capacities)[1]

    f_T = get_transform_function(transform_modus=transform_modus, var_ratio=alpha_var_ratio)
    inverse_f_T = get_inverse_transform_function(transform_modus=transform_modus, var_ratio=alpha_var_ratio)

    # calc resulting capacities
    # TODO: check whether it is correct how to do it with the linear network memory

    remembered_enc_capacities = defaultdict(lambda: defaultdict(int))
    for enc_cap in encoder_capacities:
        if enc_cap['degree'] > 1:
            enc_cap_delay = enc_cap['delay']
            window_str = str(get_window_values(enc_cap))
            # for delay, linear_network_memory_value in combined_linear_memory.items():
            for delay, linear_network_memory_value in network_linear_memory.items():
                delay_key = enc_cap_delay + delay - 1  # delays start with 1
                old_remembered_score = remembered_enc_capacities[delay_key][
                    window_str]  # 0 if there was no old score, because of defaultdict
                if fixed_memratio is None:
                    linear_memory_ratio = f_T(linear_network_memory_value) / f_T(linear_encoder_delay0_memory)
                else:
                    linear_memory_ratio = fixed_memratio
                new_remembered_score = inverse_f_T(f_T(enc_cap['score']) * linear_memory_ratio)
                # if transform_modus == 'alpha':
                #     if fixed_memratio is None:
                #         linear_memory_ratio = to_alpha(linear_network_memory_value, var_ratio=alpha_var_ratio) / to_alpha(
                #             linear_encoder_delay0_memory, var_ratio=alpha_var_ratio)
                #     else:
                #         linear_memory_ratio = fixed_memratio
                #     new_remembered_score = to_squared_corr(
                #         to_alpha(enc_cap['score'], var_ratio=alpha_var_ratio) * linear_memory_ratio,
                #         var_ratio=alpha_var_ratio)
                # elif transform_modus == 'sqrt':
                #     linear_memory_ratio = np.sqrt(linear_network_memory_value) / np.sqrt(
                #         linear_encoder_delay0_memory)
                #     new_remembered_score = (np.sqrt(enc_cap['score']) * linear_memory_ratio) ** 2
                # elif transform_modus == 'identity':
                #     linear_memory_ratio = linear_network_memory_value / linear_encoder_delay0_memory
                #     new_remembered_score = enc_cap['score'] * linear_memory_ratio
                # else:
                #     raise ValueError(f'transform_modus "{transform_modus}" does not exists.')
                remembered_enc_capacities[delay_key][window_str] = max(old_remembered_score, new_remembered_score)

    return remembered_enc_capacities


def get_effective_capacity_sum(effective_enc_capacities):
    effective_cap_scores = []
    for delay, powerlist_dict in effective_enc_capacities.items():
        for powerlist_str, score in powerlist_dict.items():
            effective_cap_scores.append(score)
    total_effective_enc_capacity = np.sum(effective_cap_scores)

    return total_effective_enc_capacity


def load_capacities(capacity_path):
    with open(capacity_path, 'rb') as cap_file:
        capacity_data = pickle.load(cap_file)

    return capacity_data


def parse_cmd():
    parser = argparse.ArgumentParser()

    parser.add_argument('--encoder_capacity', required=True, help='Path to the pickled capacities of the encoder.')
    parser.add_argument('--total_capacity', required=True,
                        help='Path to the pickled capacities of the network state matrix.')
    parser.add_argument('--output_path', required=True, help='Path to store the resulting capacities to.')
    parser.add_argument('--figure_path', help='Path to store the resulting capacity figure to.')
    parser.add_argument('--show_figure', action='store_true', help='Whether to show the resulting capacity figure.')
    # parser.add_argument('--use_sqrt', action='store_true',
    #                     help='Whether to use the sqrt correction for the memory ratio.')
    # parser.add_argument('--disable_alpha_transformation', action='store_true',
    #                     help='Whether to disable the transformation to alpha values.')
    parser.add_argument('--transformation_modus', default='identity',
                        help='Possible arguments: identity, sqrt, alpha')
    parser.add_argument('--subtract_all', action='store_true', help='Whether to set all memory ratios to 1 and '
                                                                    'therefore fully subtract all delayed encoder '
                                                                    'capacities.')
    parser.add_argument('--variance_ratio', type=float, help="Ratio of variances of noise signal and target signal. "
                                                             "Only used for alpha transformation modus.",
                        default=None)
    parser.add_argument('--subtract_undelayed_degree1', action='store_true',
                        help='Whether to subtract also the undelayed reconstruction of the input signal.')

    args = parser.parse_args()

    if args.transformation_modus == 'alpha' and args.variance_ratio is None:
        args.variance_ratio = 1.
        print('Using default value of 1. for alpha_var_ratio.')
    elif args.transformation_modus != 'alpha' and args.variance_ratio is not None:
        print("alpha_var_ratio won't be used, because transformation_modus is not 'alpha'.")
        args.variance_ratio = None

    return args


def get_capacity_maxes(encoder_capacities: list, remembered_capacities: dict) -> dict:
    capacity_maxes = remembered_capacities.copy()
    if type(capacity_maxes) != defaultdict:
        capacity_maxes = defaultdict(lambda: defaultdict(int), capacity_maxes)

    for enc_cap in encoder_capacities:
        delay = enc_cap['delay']
        window_str = str(get_window_values(enc_cap))
        enc_score = enc_cap['score']
        remembered_score = capacity_maxes[delay][window_str]
        capacity_maxes[delay][window_str] = max(enc_score, remembered_score)

    return capacity_maxes


def main():
    args = parse_cmd()
    # use_sqrt = not args.disable_sqrt
    # args.subtract_undelayed_degree1 = True
    # use_alpha = not args.disable_alpha_transformation

    total_cap_data = load_capacities(args.total_capacity)

    combined_capacities = total_cap_data['all_capacities']
    combined_linear_memory = get_linear_memory(combined_capacities)

    # max_delay = np.max([x['delay']+x['window'] for x in combined_capacities])

    encoder_capacities = load_capacities(args.encoder_capacity)['all_capacities']
    encoder_linear_memory = get_linear_memory(encoder_capacities)
    network_linear_memory = get_linear_network_memory(combined_linear_memory, encoder_linear_memory,
                                                      transform_modus=args.transformation_modus,
                                                      alpha_var_ratio=args.variance_ratio)
    memratio = 1. if args.subtract_all else None
    remembered_enc_capacities = get_remembered_encoder_capacities(encoder_capacities=encoder_capacities,
                                                                  network_linear_memory=network_linear_memory,
                                                                  transform_modus=args.transformation_modus,
                                                                  alpha_var_ratio=args.variance_ratio,
                                                                  fixed_memratio=memratio)
    # remembered_enc_capacities = get_remenbered_encoder_capacities(encoder_capacities, combined_capacities,
    #                                                               use_sqrt=use_sqrt)
    subtraction_capacities = get_capacity_maxes(encoder_capacities, remembered_enc_capacities)
    # network_capacities = subtract_capacities(combined_capacities, remembered_enc_capacities, subtract_undelayed_degree1=args.subtract_undelayed_degree1)
    network_capacities = subtract_capacities(combined_capacities, subtraction_capacities,
                                             subtract_undelayed_degree1=args.subtract_undelayed_degree1,
                                             transform_modus=args.transformation_modus,
                                             alpha_var_ratio=args.variance_ratio)

    result_capacities_dict = {
        'name': f'network_cap_{total_cap_data["name"]}',
        'total_capacity': np.sum([x['score'] for x in network_capacities]),
        'all_capacities': network_capacities,
        'num_capacities': len(network_capacities),  # do I have to remove the new 0 value capacities?
        'nodes': total_cap_data['nodes'],
        'compute_time': total_cap_data['compute_time']
    }

    with open(args.output_path, 'wb') as cap_file:
        pickle.dump(result_capacities_dict, cap_file)

    print(f'Total capacity before removing encoder capacities: {total_cap_data["total_capacity"]}')
    print(f'Total capacity after removing encoder capacities: {np.sum([x["score"] for x in network_capacities])}')
    print(f'Total possible (remembered) encoder capacity: {get_effective_capacity_sum(remembered_enc_capacities)}')
    print(f'Resulting capacities were stored to {args.output_path}')

    if args.figure_path is not None or args.show_figure:
        plot_capacity_figure(
            combined_capacities=total_cap_data,
            encoder_capacities={'all_capacities': encoder_capacities},
            remembered_enc_capacities={
                'all_capacities': delay_window_capdict_to_default_capdictlist_dummy(remembered_enc_capacities)},
            subtract_capacities={
                'all_capacities': delay_window_capdict_to_default_capdictlist_dummy(subtraction_capacities)},
            network_capacities=result_capacities_dict,
            linear_network_memory=network_linear_memory
        )
        plt.tight_layout()
        if args.figure_path is not None:
            plt.savefig(args.figure_path)
        if args.show_figure:
            plt.show()


if __name__ == "__main__":
    main()
