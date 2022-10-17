import pickle
import argparse

import numpy as np
import matplotlib.pyplot as plt

# import utils
from plotting.colors import get_degree_color


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


def plot_capacity_bars(capacity_dict, ax=None, cutoff=0., show_sums=False, max_degree_to_plot=None):
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = None

    capacities = [cap for cap in capacity_dict['all_capacities'] if cap['score'] > cutoff]
    cap_matrix = cap2vec(capacities)
    degrees = np.unique([cap['degree'] for cap in capacities])
    if max_degree_to_plot is not None:
        degrees = degrees[degrees <= max_degree_to_plot]
    max_delay = np.max([cap['delay'] for cap in capacities])

    previous_heights = np.zeros(max_delay)
    max_degree_per_delay = np.zeros(max_delay)
    for deg in degrees:
        degree_capacities = cap_matrix[deg]
        degreecolor = get_degree_color(deg)
        ax.bar(x=list(range(max_delay)), height=degree_capacities, bottom=previous_heights, label=f'degree {deg}',
               linewidth=0, color=degreecolor)
        previous_heights += cap_matrix[deg]
        for delay in range(max_delay):
            if degree_capacities[delay] > 0.:
                max_degree_per_delay[delay] = deg

    capsums = np.sum(cap_matrix, axis=0)
    if show_sums:
        for i, cap in enumerate(capsums):
            ax.text(i, 0.01 * capsums.max() + cap, f'{round(cap, 3)}', va='center', ha='center')
            # ax.text(i, 0.04 * capsums.max() + cap, f'max degree: {int(max_degree_per_delay[i])}', va='center', ha='center')

    ax.set_xlabel('delay')
    ax.set_ylabel('capacity')

    return fig, ax


def parse_cmd():
    parser = argparse.ArgumentParser()

    parser.add_argument('pickled_capacity')
    parser.add_argument('--title', default='Capacities')
    parser.add_argument('--figure_path', default=None)
    parser.add_argument('--cutoff', default=0.)
    parser.add_argument('--show_sums', action='store_true')
    parser.add_argument('--max_degree_to_plot', default=None, type=int)

    return parser.parse_args()


def main():
    args = parse_cmd()
    with open(args.pickled_capacity, 'rb') as pickled_file:
        capacity_dict = pickle.load(pickled_file)

    fig, ax = plt.subplots(figsize=(6.4, 2.4))
    _, ax = plot_capacity_bars(capacity_dict, cutoff=args.cutoff, show_sums=args.show_sums, ax=ax,
                               max_degree_to_plot=args.max_degree_to_plot)
    ax.set_title(args.title)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[::-1], labels[::-1])

    if args.figure_path is None:
        plt.show()
    else:
        plt.savefig(args.figure_path)


if __name__ == "__main__":
    main()
