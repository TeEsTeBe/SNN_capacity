#!/usr/env python

import os
import argparse
from time import time
import pickle

import numpy as np
import pandas as pd

from capacity import capacity_calculation


def main(args):
    print('================= Processing Capacity =================')
    inputs, states = load_inputs_and_states(args)

    # run the processing capacity
    time_before_cap = time()
    cap_iter = capacity_calculation.CapacityIterator(verbosity=args.verbosity, orth_factor=args.orth_factor,
                                                     maxdeg=args.max_degree,
                                                     maxdel=args.max_delay, m_variables=args.m_variables,
                                                     m_powerlist=args.m_powerlist,
                                                     m_windowpos=args.m_windowpos,
                                                     delskip=args.delskip, windowskip=args.windowskip)
    total_capacity, all_capacities, num_capacities, nodes = cap_iter.collect(inputs, states)
    cap_duration = time() - time_before_cap
    print(f'Capacity computation took {cap_duration} sec ({cap_duration / 60.} min).', flush=True)
    full_results = {
        'name': args.name,
        'total_capacity': total_capacity,
        'all_capacities': all_capacities,
        'num_capacities': num_capacities,
        'nodes': nodes,
        'compute_time': cap_duration,
    }
    store_results(args, full_results)


def parse_cmd():
    parser = argparse.ArgumentParser()

    parser.add_argument('--name', type=str, help='name of the current run')
    parser.add_argument('--input', type=str, help='Path to input numpy array.')
    parser.add_argument('--states_path', type=str, help='Path to statematrix numpy array.')
    parser.add_argument('--results_file', type=str, help='Path to a CSV file where the results of multiple runs should '
                                                         'be stored.')
    parser.add_argument('--capacity_results', type=str, help='Path where the capacity values of this run should be '
                                                             'stored')
    parser.add_argument('--max_degree', type=int, default=1000, help='Maximum degree that should be evaluated.')
    parser.add_argument('--max_delay', type=int, default=1000, help='Maximum delay that should be evaluated.')
    parser.add_argument('--m_variables', action='store_true', help="Whether to assume a monotonous decrease of capacity"
                                                                   " with increasing number of variables")
    parser.add_argument('--m_powerlist', action='store_true', help="Whether to assume a monotonous decrease of capacity"
                                                                   " with increasing power list")
    parser.add_argument('--m_windowpos', action='store_true', help="Whether to assume a monotonous decrease of capacity"
                                                                   " with increasing positions in the window")
    parser.add_argument('--orth_factor', type=float, default=6., help="Factor that increases the cutoff value")
    parser.add_argument('--figures_path', type=str, default='figures')
    parser.add_argument('--n_warmup', type=int, default=0, help="Number of warm up simulation steps.")
    parser.add_argument('--sample_ids', type=str, default=None, help="Path to a numpy array with ids of unist that"
                                                                     " should be used for subsampling the state matrix")
    parser.add_argument('--sample_size', type=int, default=None, help="Number of units to use as a random subsample "
                                                                      "of the state matrix")
    parser.add_argument('--sample_step', type=int, default=None, help="Subsample the state matrix by using only every "
                                                                      "sample_step unit.")
    parser.add_argument('--delskip', type=int, default=0, help="Number of delays before a monotonous decrease of "
                                                               "capacity values is assumed.")
    parser.add_argument('--windowskip', type=int, default=0, help="Number of windows before a monotonous decrease of "
                                                                  "capacity values is assumed.")
    parser.add_argument('-v', '--verbosity', type=int, default=0)

    return parser.parse_args()


def cap2vec(capacities):
    maxdeg = np.max([cap['degree'] for cap in capacities])
    maxdel = np.max([cap['delay'] for cap in capacities])
    vec = np.zeros((maxdeg + 1, maxdel))
    for idx in range(len(capacities)):
        delay = capacities[idx]['delay']
        degree = capacities[idx]['degree']
        if (delay <= maxdel) and (degree <= maxdeg):
            vec[degree, delay - 1] += capacities[idx]['score']

    return vec


def store_results(args, full_results):
    with open(args.capacity_results, 'wb') as pklfile:
        pickle.dump(full_results, pklfile)
        print(f'Results saved to {args.capacity_results}', flush=True)
    if args.results_file is not None:
        result_dict = {
            'name': [args.name],
            'N': [full_results['nodes']],
            'input': [args.input],
            'states_path': [args.states_path],
            'max_degree': [args.max_degree],
            'max_delay': [args.max_delay],
            'm_variables': [args.m_variables],
            'm_powerlist': [args.m_powerlist],
            'orth_factor': [args.orth_factor],
            'n_warmup': [args.n_warmup],
            'total_cap': [full_results['total_capacity']],
            'cap_duration': [full_results['compute_time']],
            'sample_size': [args.sample_size],
            'sample_ids': [args.sample_ids],
            'sample_step': [args.sample_step],
            'cap_results_path': [args.capacity_results],
        }

        all_capacities = full_results['all_capacities']
        if len(all_capacities) > 0:
            max_degree = np.max([x['degree'] for x in all_capacities])

            degree_delay_cap_array = cap2vec(all_capacities)

            for degree in range(1, max_degree):
                result_dict[f'cap_deg{degree}'] = [np.sum(degree_delay_cap_array[degree, :])]

        result_df = pd.DataFrame.from_dict(result_dict)
        if os.path.exists(args.results_file):
            result_df.to_csv(args.results_file, mode='a', sep='\t', header=False, index=False)
        else:
            result_df.to_csv(args.results_file, sep='\t', header=True, index=False)


def load_inputs_and_states(args):
    inputs = np.load(args.input)
    print(f'Loading state matrix from {args.states_path}', flush=True)
    states = np.load(args.states_path)
    if states.shape[1] > states.shape[0]:
        print(f'State matrix has old shape (N,T). Transposed matrix is used.', flush=True)
        states = states.T
    if args.n_warmup > 0:
        print(f'Discarding first {args.n_warmup} steps.', flush=True)
        inputs = inputs[args.n_warmup:]
        states = states[args.n_warmup:, :]
        if states.shape[0] > inputs.shape[0]:
            states = states[:-1, :]
        print(f'State shape: {states.shape}, input shape: {inputs.shape}', flush=True)
    if args.sample_size is not None or args.sample_ids is not None:
        if args.sample_ids is not None and os.path.exists(args.sample_ids):
            print(f'Using precomputed sample ids from {args.sample_ids}', flush=True)
            sample_ids = np.load(args.sample_ids)
        else:
            sample_ids = np.random.choice(range(states.shape[1]), size=args.sample_size, replace=False)
            if args.sample_ids is not None:
                print(f'Saved generated subsampling ids to {args.sample_ids}', flush=True)
                np.save(args.sample_ids, sample_ids)
        print(f'Using random subsample of {len(sample_ids)} neurons.', flush=True)
        states = states[:, sample_ids]
    elif args.sample_step is not None:
        print(f'Sampling state matrix with step size {args.sample_step}')
        states = states[:, ::args.sample_step]
    else:
        print('Using full state matrix without subsampling.', flush=True)
    return inputs, states


if __name__ == '__main__':
    main(parse_cmd())
