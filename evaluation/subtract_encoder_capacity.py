import pickle
import argparse
from collections import defaultdict

import numpy as np


def subtract_capacities(total_capacities, capacities_to_subtract):
    result_capacities = []

    for cap in total_capacities:
        result_cap = cap.copy()
        if cap['degree'] > 1:  # we don't want to remove the linear memory of the system
            try:
                subtraction_value = capacities_to_subtract[cap['delay']][str(cap['powerlist'])]
            except:
                subtraction_value = 0.

            result_cap['score'] = max(0, result_cap['score'] - subtraction_value)

        result_capacities.append(result_cap)

    return result_capacities


def get_linear_memory(capacities):
    linear_memory = {}

    for cap in capacities:
        if cap['degree'] == 1:
            linear_memory[cap['delay'] - 1] = cap['score']

    return linear_memory


def get_effective_encoder_capacities(encoder_capacities, total_capacities):
    # calc linear memory
    linear_memory_total = get_linear_memory(total_capacities)
    enc_deg1_del0 = get_linear_memory(encoder_capacities)[0]

    # calc resulting capacities
    effective_enc_capacities = defaultdict(dict)
    for enc_cap in encoder_capacities:
        asdf = 0
        for delay, memory_value in linear_memory_total.items():
            effective_enc_capacities[delay][str(enc_cap['powerlist'])] = enc_cap['score'] * (
                    memory_value / enc_deg1_del0)
            print(f'del {delay}\tenccap {enc_cap["powerlist"]}\tmemoryval {memory_value}')

    return effective_enc_capacities


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

    return parser.parse_args()


def main():
    args = parse_cmd()

    encoder_capacities = load_capacities(args.encoder_capacity)['all_capacities']

    total_cap_data = load_capacities(args.total_capacity)
    print(f'Total capacity before: {total_cap_data["total_capacity"]}')
    total_capacities = total_cap_data['all_capacities']

    result_capacities = get_network_capacities(encoder_capacities, total_capacities)

    with open(args.output_path, 'wb') as cap_file:
        pickle.dump(result_capacities, cap_file)

    print(f'Resulting capacities were stored to {args.output_path}')


def get_network_capacities(encoder_capacities, total_capacities):
    effective_enc_capacities = get_effective_encoder_capacities(encoder_capacities, total_capacities)
    effective_cap_scores = []
    for delay, powerlist_dict in effective_enc_capacities.items():
        for powerlist_str, score in powerlist_dict.items():
            effective_cap_scores.append(score)
    total_effective_enc_capacity = np.sum(effective_cap_scores)
    result_capacities = subtract_capacities(total_capacities, effective_enc_capacities)

    print(f'Total capacity after: {np.sum([x["score"] for x in result_capacities])}')
    print(f'Total possible encoder capacity: {total_effective_enc_capacity}')

    return result_capacities


if __name__ == "__main__":
    main()
