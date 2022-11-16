import argparse
import numpy as np
import matplotlib.pyplot as plt

from heatmaps import plot_heatmap


def get_title(title, net, plot_max_degrees=False, plot_max_delays=False, plot_num_trials=False):
    if title is None:
        if plot_max_degrees:
            title = f"{net} maximum capacity degrees"
        elif plot_max_delays:
            title = f"{net} maximum capacity delays"
        elif plot_num_trials:
            title = f"{net} number of trials"
        else:
            title = f"{net} total capacity"

    return title


def parse_cmd():
    parser = argparse.ArgumentParser()

    parser.add_argument('--x_name', default='dur', help='Defines which variable will be on the x axis of the heatmap')
    parser.add_argument('--y_name', default='max', help='Defines which variable will be on the y axis of the heatmap')
    parser.add_argument('--capacity_folder', default='/home/schultetobrinke/projects/recurrence/repos/capacity_visualisation/capacity_visualisation/data/spatial_randomnoise_dur1-50_max0.2-3.0__net=brunel__std=5.0__inp=spatial_DC__steps=200000/capacity')
    parser.add_argument('--title', default=None)
    parser.add_argument('--steps', default=200000)
    # parser.add_argument('--nodes', default=50)
    # parser.add_argument('--nwarmup', default=500)
    parser.add_argument('--max', default=0.2)
    parser.add_argument('--dur', default=10.0)
    parser.add_argument('--net', default='brunel')
    parser.add_argument('--inp', default='spatial_DC')
    parser.add_argument('--cutoff', default=0., type=float)
    parser.add_argument('--figure_path', default=None)
    parser.add_argument('--plot_max_degrees', action='store_true', help="Plot the maximum degrees instead of the capacities")
    parser.add_argument('--plot_max_delays', action='store_true', help="Plot the maximum delays instead of the capacities")
    parser.add_argument('--plot_num_trials', action='store_true', help="Plot the number of trials instead of the capacities")
    parser.add_argument('--annotate', action='store_true', help="Turn on the annotations inside the heatmap")
    parser.add_argument('--use_filtered_spikes', action='store_true', help="")
    parser.add_argument('--mindegree', type=int, default=0)
    parser.add_argument('--maxdegree', type=int, default=np.inf)
    parser.add_argument('--mindelay', type=int, default=0)
    parser.add_argument('--maxdelay', type=int, default=np.inf)
    parser.add_argument('--y_axis_factor', type=float, default=1.)

    return parser.parse_args()


def main():
    args = parse_cmd()
    # fig, ax = plt.subplots(figsize=(6.4, 4.8))
    fig, ax = plt.subplots(figsize=(3.4, 2.7))
    # fac = 0.8 * 0.5  # Poster
    # fig, ax = plt.subplots(figsize=(fac*6.4, fac*4.8))  # Poster
    params_to_filter = {
        # 'steps': args.steps,
        'max': args.max,
        'dur': args.dur,
        'net': args.net,
        'inp': args.inp
    }
    args_dict = vars(args).copy()
    net = args.net
    for paramname in params_to_filter.keys():
        del args_dict[paramname]
    del args_dict['steps']
    del params_to_filter[args.x_name]
    del params_to_filter[args.y_name]
    args_dict['params_to_filter'] = params_to_filter
    args_dict['title'] = get_title(args.title, net, args.plot_max_degrees, args.plot_max_delays, args.plot_num_trials)

    if args.use_filtered_spikes:
        other_filter_keys = ['filter']
    else:
        other_filter_keys = ['vm']

    # other_filter_keys = ['encoder']
    other_filter_keys = None

    del args_dict['use_filtered_spikes']
    del args_dict['y_axis_factor']

    _, ax = plot_heatmap(**args_dict, other_filter_keys=other_filter_keys, ax=ax)
    # fig, ax = plot_heatmap(**args_dict, other_filter_keys=other_filter_keys)
    # ax.set_yticklabels([f'{args.y_axis_factor * float(n.get_text()):.3f}' for n in ax.get_yticklabels()])
    # ax.set_yticklabels([])  # Poster
    # ax.set_xticklabels([])  # Poster
    plt.tight_layout()
    # plt.tight_layout(pad=0.3)  # Poster
    if args.figure_path is None:
        plt.show()
    else:
        plt.savefig(args.figure_path)


if __name__ == "__main__":
    main()
