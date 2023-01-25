import argparse
import numpy as np
from scipy.stats import spearmanr

from plotting.heatmaps import get_heatmap_data, get_task_results


def get_correlation(cap_data_dict, task_data_dict, use_spearmanr=False):
    cap_results_list = []
    task_results_list = []

    for x_key, x_value_dict in cap_data_dict.items():
        for y_key in x_value_dict.keys():
            cap_results_list.append(cap_data_dict[x_key][y_key])
            task_results_list.append(task_data_dict[x_key][y_key])

    if use_spearmanr:
        correlation = spearmanr(cap_results_list, task_results_list)[0]
    else:
        correlation = np.corrcoef(cap_results_list, task_results_list)[0][1]

    return correlation


def parse_cmd():
    parser = argparse.ArgumentParser()

    parser.add_argument('--capacity_folder', required=True)
    parser.add_argument('--task_folder', required=True)
    parser.add_argument('--x_name', default='dur', help='Defines which variable will be on the x axis of the heatmap')
    parser.add_argument('--y_name', default='max', help='Defines which variable will be on the y axis of the heatmap')
    parser.add_argument('--get_max_degrees', action='store_true',
                        help="Use the maximum degrees instead of the capacities")
    parser.add_argument('--get_max_delays', action='store_true',
                        help="Use the maximum delays instead of the capacities")
    parser.add_argument('--mindegree', type=int, default=0)
    parser.add_argument('--maxdegree', type=int, default=np.inf)
    parser.add_argument('--mindelay', type=int, default=0)
    parser.add_argument('--maxdelay', type=int, default=np.inf)
    parser.add_argument('--metric', default='kappa')

    return parser.parse_args()


def main():
    args = parse_cmd()

    cap_data_args = vars(args).copy()
    del cap_data_args['task_folder']
    cap_data_args['params_to_filter'] = {}
    metric = args.metric
    del cap_data_args['metric']

    cap_data_dict = get_heatmap_data(**cap_data_args)
    task_data_dict = get_task_results(args.task_folder, x_param_name=args.x_name, y_param_name=args.y_name,
                                      metric=metric).to_dict()

    cc = get_correlation(cap_data_dict, task_data_dict)
    print(cc)


if __name__ == "__main__":
    main()
