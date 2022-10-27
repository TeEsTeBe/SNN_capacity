import pickle
import argparse
import matplotlib.pyplot as plt
from collections import defaultdict

import numpy as np

from plotting.cap_bars_single_run import plot_capacity_bars


def plot_capacity_figure(combined_capacities: dict, encoder_capacities: dict, remembered_enc_capacities: dict,
                         network_capacities: dict):
    fig, axes = plt.subplots(nrows=4, sharex='all', figsize=((5.2, 5.2)))

    cap_dict_list = [combined_capacities, encoder_capacities, remembered_enc_capacities, network_capacities]
    titles = ['combined', 'encoder', 'remembered', 'network']

    for ax, capacities, title in zip(axes, cap_dict_list, titles):
        plot_capacity_bars(capacity_dict=capacities, ax=ax)
        ax.set_title(title)

    return fig, axes


def delay_window_capdict_to_default_capdictlist_dummy(delay_window_capdict: dict) -> list:
    cap_dict_list_dummy = []
    for delay, window_to_score_dict in delay_window_capdict.items():
        for window_str, score in window_to_score_dict.items():
            # print(f'delay: {delay}, window: {window_str}')
            window = [float(x) for x in window_str[1:-1].split(',')]
            degree = np.sum(window)
            powerlist = [x for x in window if x > 0.]
            window_positions = [i for i, deg in enumerate(window) if deg > 0.]

            cap_dict_list_dummy.append({
                'delay': int(delay),
                'score': score,
                'degree': int(degree),
                'powerlist': powerlist,
                'window_positions': window_positions,
            })

    return cap_dict_list_dummy


def get_window_values(single_capacity):
    poly_definition = np.zeros(single_capacity['window'])
    for pos, power in zip(single_capacity['window_positions'], single_capacity['powerlist']):
        poly_definition[pos] = power

    return poly_definition


def subtract_capacities(combined_capacities: list, capacities_to_subtract: dict):
    result_capacities = []

    for cap in combined_capacities:
        result_cap = cap.copy()
        if cap['degree'] > 1:  # we don't want to remove the linear memory of the system
            try:
                window_values = get_window_values(cap)
                subtraction_value = capacities_to_subtract[cap['delay']][str(window_values)]
                # subtraction_value = capacities_to_subtract[cap['delay']][str(cap['powerlist'])]
            except:
                subtraction_value = 0.

            result_cap['score'] = max(0, result_cap['score'] - subtraction_value)

        if result_cap['score'] > 0.:
            result_capacities.append(result_cap)

    return result_capacities


def get_linear_memory(capacities: list) -> dict:
    linear_memory = {}

    for cap in capacities:
        if cap['degree'] == 1:
            linear_memory[cap['delay']] = cap['score']

    return linear_memory


def get_linear_network_memory(combined_linear_memory: dict, encoder_linear_memory: dict) -> dict:
    network_linear_memory = {}
    for delay, combined_mem_val in combined_linear_memory.items():
        if delay in encoder_linear_memory.keys():
            encoder_mem_val = encoder_linear_memory[delay]
        else:
            encoder_mem_val = 0.
        network_linear_memory[delay] = max(0, combined_mem_val - encoder_mem_val)

    return network_linear_memory


def get_remenbered_encoder_capacities(encoder_capacities: list, total_capacities: list) -> dict:
    # calc linear memory
    combined_linear_memory = get_linear_memory(total_capacities)
    encoder_linear_memory = get_linear_memory(encoder_capacities)
    network_linear_memory = get_linear_network_memory(combined_linear_memory, encoder_linear_memory)

    linear_encoder_delay0_memory = encoder_linear_memory[1]

    # calc resulting capacities
    # TODO: check whether it is correct how to do it with the linear network memory

    remembered_enc_capacities = defaultdict(lambda: defaultdict(int))
    for enc_cap in encoder_capacities:
        enc_cap_delay = enc_cap['delay']
        window_str = str(get_window_values(enc_cap))
        # for delay, linear_network_memory_value in combined_linear_memory.items():
        for delay, linear_network_memory_value in network_linear_memory.items():
            delay_key = enc_cap_delay + delay - 1  # delays start with 1
            old_remembered_score = remembered_enc_capacities[delay_key][
                window_str]  # 0 if there was no old score, because of defaultdict
            linear_memory_ratio = linear_network_memory_value / linear_encoder_delay0_memory
            new_remembered_score = enc_cap['score'] * linear_memory_ratio
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

    return parser.parse_args()


def main():
    args = parse_cmd()

    total_cap_data = load_capacities(args.total_capacity)

    combined_capacities = total_cap_data['all_capacities']
    encoder_capacities = load_capacities(args.encoder_capacity)['all_capacities']
    remembered_enc_capacities = get_remenbered_encoder_capacities(encoder_capacities, combined_capacities)
    network_capacities = subtract_capacities(combined_capacities, remembered_enc_capacities)

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
    print(f'Total possible encoder capacity: {get_effective_capacity_sum(remembered_enc_capacities)}')
    print(f'Resulting capacities were stored to {args.output_path}')

    if args.figure_path is not None or args.show_figure:
        plot_capacity_figure(
            combined_capacities=total_cap_data,
            encoder_capacities={'all_capacities': encoder_capacities},
            remembered_enc_capacities={
                'all_capacities': delay_window_capdict_to_default_capdictlist_dummy(remembered_enc_capacities)},
            network_capacities=result_capacities_dict)
        plt.tight_layout()
    if args.figure_path is not None:
        plt.savefig(args.figure_path)
    if args.show_figure:
        plt.show()


if __name__ == "__main__":
    main()
