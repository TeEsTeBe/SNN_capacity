import os
import argparse

import yaml
import numpy as np
import matplotlib.pyplot as plt

from evaluation.calculate_task_correlations import get_correlation
from heatmaps import get_heatmap_data, get_task_results
from colors import get_color


def parse_cmd():
    parser = argparse.ArgumentParser()

    parser.add_argument('--parameter_file', help='Path to a parameter file.', default='correlation_plot_parameters_BRN.yaml')

    return parser.parse_args()


def main():
    args = parse_cmd()
    with open(args.parameter_file, 'r') as parameters_file:
        parameters = yaml.safe_load(parameters_file)

    bar_width = 0.25

    for cap_title, cap_dict in parameters['cap_to_tasks_dict'].items():
        if os.path.exists(cap_dict['cap_groupname']) and len([x for x in os.listdir(cap_dict['cap_groupname']) if x.endswith(".pkl")]) > 0:
            print(f'The cap_groupname for {cap_title} is a full capacity folder. '
                  f'We use this instead of constructing a path.')
            cap_folder = cap_dict['cap_groupname']
        else:
            cap_folder = os.path.join(parameters['cap_base_folder'], cap_dict['cap_groupname'], 'capacity')
        # fig, ax = plt.subplots(figsize=(5, 8))
        # fac = 1.25  # Poster
        # fig, ax = plt.subplots(figsize=(fac*5, fac*4))  # Poster
        fac = 1.  # Poster
        fig, ax = plt.subplots(figsize=(fac*4, fac*4))  # Poster
        ax.set_ylim((-1, 1))
        print(cap_title)

        colors = {
            # 'capacity': 'black',
            # 'capacity': '#628395',
            # 'capacity': '#297373',
            # 'capacity': '#3A2E39',
            # 'capacity': '#3A2E39',
            # 'capacity': '#D1495B',
            # 'capacity': '#3E6F84',
            'capacity': get_color('capacity'),
            # 'capacity': '#4A6A78',
            # 'nonlin. cap. delay 5': 'grey',
            'nonlin. cap. delay 5': '#77CBB9',
            # 'nonlinear capacity\ndelay 5': 'grey',
            'nonlinear capacity\ndelay 5': '#77CBB9',
            'nonlin. cap. delay 10': '#2C0735',
            # 'nonlinear capacity\ndelay 10': 'darkblue',
            'nonlinear capacity\ndelay 10': '#2C0735',
            # 'degrees': 'darkorchid',  # 'wheat',
            # 'degrees': '#96897B',  # 'wheat',
            # 'degrees': '#FF8552',  # 'wheat',
            # 'degrees': '#9FAF90',  # 'wheat',
            # 'degrees': '#EDAE49',  # 'wheat',
            # 'degrees': '#934E53',  # 'wheat',
            'degrees': get_color('degree'),  # 'wheat',
            # 'degrees': '#875A5D',  # 'wheat',
            # 'delays': 'teal',  # 'plum',
            # 'delays': '#DFD5A5',  # 'plum',
            # 'delays': '#E9D758',  # 'plum',
            # 'delays': '#D68FD6',  # 'plum',
            # 'delays': '#00798C',  # 'plum',
            # 'delays': '#C6A052',  # 'plum',
            'delays': get_color('delay'),  # 'plum',
            # 'delays': '#B29967',  # 'plum',
        }

        # capacity_info_types = ['degrees', 'capacity', 'delays', 'nonlin. cap. delay 10']
        # capacity_info_types = ['degrees', 'capacity', 'delays', 'nonlinear capacity\ndelay 5']
        capacity_info_types = ['degrees', 'capacity', 'delays', 'nonlinear capacity\ndelay 10']
        # capacity_info_types = ['degrees', 'capacity', 'delays', 'nonlinear capacity\ndelay 5', 'nonlinear capacity\ndelay 10']

        for info_type_idx, cap_info_type in enumerate(capacity_info_types):
            print(f'\t{cap_info_type}')
            shift_factor = info_type_idx - 1
            get_max_degrees = (cap_info_type == 'degrees')
            get_max_delays = (cap_info_type == 'delays')

            if cap_info_type in ['nonlin. cap. delay 5', 'nonlinear capacity\ndelay 5']:
                cap_data_dict = get_heatmap_data(x_name='dur', y_name='max', capacity_folder=cap_folder,
                                                 params_to_filter={}, mindelay=5, maxdelay=5, mindegree=2)
                                                 # params_to_filter={}, mindelay=6, maxdelay=6, mindegree=2)
            elif cap_info_type in ['nonlin. cap. delay 10', 'nonlinear capacity\ndelay 10']:
                cap_data_dict = get_heatmap_data(x_name='dur', y_name='max', capacity_folder=cap_folder,
                                                 params_to_filter={}, mindelay=10, maxdelay=10, mindegree=1)
                                                 # params_to_filter={}, mindelay=11, maxdelay=11, mindegree=1)
            else:
                cap_data_dict = get_heatmap_data(x_name='dur', y_name='max', capacity_folder=cap_folder,
                                                 params_to_filter={}, get_max_delays=get_max_delays,
                                                 get_max_degrees=get_max_degrees)

            if np.max(list(cap_data_dict[1.0].keys())) == 3.0:
                tmp_cap_data_dict = {}
                for dur, amp_dict in cap_data_dict.items():
                    tmp_cap_data_dict[dur] = {}
                    for amp, cap in amp_dict.items():
                        tmp_cap_data_dict[dur][round(amp * 0.2, 2)] = cap
                cap_data_dict = tmp_cap_data_dict

            tasknames = []
            correlations = []

            for task_name, task_group in cap_dict['tasks'].items():
                print(f'\n______ {task_name} ___________')
                task_folder = os.path.join(parameters['task_base_folder'], task_group)

                if task_name == "class. del. sum":
                    task_data_dict = get_task_results(task_folder, x_param_name='dur', y_param_name='max',
                                                      aggregation_type='sum_over_delays', metric='accuracy').to_dict()
                elif task_name in ["class. max. del.", "classification", "classi-\nfication", "class."]:
                    task_data_dict = get_task_results(task_folder, x_param_name='dur', y_param_name='max',
                                                      aggregation_type='max_delay', metric='accuracy').to_dict()
                elif "NARMA" in task_name:
                    task_data_dict = get_task_results(task_folder, x_param_name='dur', y_param_name='max',
                                                      metric='squared_corr_coeff').to_dict()
                else:
                    task_data_dict = get_task_results(task_folder, x_param_name='dur', y_param_name='max').to_dict()

                if cap_info_type in ['nonlin. cap. delay 5', 'nonlinear capacity\ndelay 5', 'nonlin. cap. delay 10', 'nonlinear capacity\ndelay 10'] and task_name not in ['NARMA5', 'NARMA10']:
                    corr = 0.
                # elif task_name == 'NARMA10' and '5' in cap_info_type:
                #     corr = 0.
                elif task_name == 'NARMA5' and '10' in cap_info_type:
                    corr = 0.
                else:
                    corr = get_correlation(cap_data_dict, task_data_dict, use_spearmanr=parameters['use_spearmanr'])
                print(f'\n\t\t{cap_title} correlation({cap_info_type},{task_name}): {corr}')
                correlations.append(corr)
                tasknames.append(task_name)

            x_positions = np.array(list(range(len(tasknames))))
            w = bar_width - (bar_width / 2) * min(abs(shift_factor), 1)
            cap_info_type = cap_info_type if '5' not in cap_info_type else "nonlin. cap. delay 5"  # Poster
            cap_info_type = cap_info_type if '10' not in cap_info_type else "nonlin. cap. delay 10"  # Poster
            ax.bar(x=x_positions + ((w + bar_width) / 2 * shift_factor), height=correlations, width=w,
                   color=colors[cap_info_type], label=cap_info_type)
            ax.set_xticks(x_positions)
            ax.set_xticklabels(tasknames, rotation=90)
            ax.set_ylabel('correlation coefficient')

        ax.set_title(cap_title)
        # ax.legend()
        # ax.legend(ncol=2)  # Poster
        ax.legend(ncol=2, prop={'size': 6})  # Poster
        fig.tight_layout()
        plt.savefig(os.path.join(parameters['fig_folder'], cap_dict['figname']))


if __name__ == "__main__":
    main()
