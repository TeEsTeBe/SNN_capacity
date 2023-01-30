from evaluation.calculate_task_correlations import get_correlation
import os
import heatmaps
from heatmaps import get_heatmap_data, get_task_results
import numpy as np
import matplotlib.pyplot as plt
from colors import get_degree_color, get_color, adjust_color



# heatmaps.plot_heatmap(
#     x_name='input_amplitude',
#     y_name='input_duration',
#     capacity_folder='../Data/FPUT_capacities',
#     title='',
#     params_to_filter={},
#     cutoff=0.,
#     figure_path='./test.pdf',
#     plot_max_degrees=False,
#     plot_max_delays=False,
#     plot_num_trials=False,
#     annotate=False,
#     plot_degree_delay_product=False,
#     ax=None,
#     other_filter_keys=None,
#     cmap=None,
#     mindegree=0,
#     maxdegree=np.inf,
#     mindelay=0,
#     maxdelay=np.inf,
#     use_cache=False,
#     max_marker_color=None,
#     colorbar_label=None,
#     cbar_ticks=None,
# )


def plot_single_correlations_plot(ax, bar_width, cap_dict, cap_folder, cap_title, task_base_folder, use_spearmanr, use_cache):
    # fig, ax = plt.subplots(figsize=(5, 8))
    # fac = 1.25  # Poster
    # fig, ax = plt.subplots(figsize=(fac*5, fac*4))  # Poster
    # fac = 1.  # Poster
    # fig, ax = plt.subplots(figsize=(fac * 4, fac * 4))  # Poster
    ax.set_ylim((-1, 1))
    print(cap_title)
    colors = {
        'capacity': get_color('capacity'),
        # 'nonlin. cap. delay 5': '#77CBB9',
        'nonlin. cap. delay 5': get_color('accent', desaturated=True),
        'nonlinear capacity\ndelay 5': '#77CBB9',
        # 'nonlin. cap. delay 10': '#2C0735',
        'nonlin. cap. delay 10': get_color('accent', desaturated=True),
        'nonlinear capacity\ndelay 10': '#2C0735',
        'degrees': get_color('degree'),  # 'wheat',
        'delays': get_color('delay'),  # 'plum',
    }
    # capacity_info_types = ['degrees', 'capacity', 'delays', 'nonlin. cap. delay 10']
    # capacity_info_types = ['degrees', 'capacity', 'delays', 'nonlinear capacity\ndelay 5']
    capacity_info_types = ['degrees', 'capacity', 'delays', 'nonlin. cap. delay 5']
    # capacity_info_types = ['degrees', 'capacity', 'delays', 'nonlinear capacity\ndelay 10']
    # capacity_info_types = ['degrees', 'capacity', 'delays', 'nonlinear capacity\ndelay 5', 'nonlinear capacity\ndelay 10']
    for info_type_idx, cap_info_type in enumerate(capacity_info_types):
        print(f'\t{cap_info_type}')
        shift_factor = info_type_idx - 1
        get_max_degrees = (cap_info_type == 'degrees')
        get_max_delays = (cap_info_type == 'delays')

        if cap_info_type in ['nonlin. cap. delay 5', 'nonlinear capacity\ndelay 5']:
            cap_data_dict = get_heatmap_data(x_name='input_duration', y_name='input_amplitude', capacity_folder=cap_folder,
                                             params_to_filter={}, mindelay=5, maxdelay=5, mindegree=2, use_cache=False)
            # params_to_filter={}, mindelay=6, maxdelay=6, mindegree=2)
        elif cap_info_type in ['nonlin. cap. delay 10', 'nonlinear capacity\ndelay 10']:
            cap_data_dict = get_heatmap_data(x_name='input_duration', y_name='input_amplitude', capacity_folder=cap_folder,
                                             params_to_filter={}, mindelay=10, maxdelay=10, mindegree=1, use_cache=False)
            # params_to_filter={}, mindelay=11, maxdelay=11, mindegree=1)
        else:
            cap_data_dict = get_heatmap_data(x_name='input_duration', y_name='input_amplitude', capacity_folder=cap_folder,
                                             params_to_filter={}, get_max_delays=get_max_delays,
                                             get_max_degrees=get_max_degrees, use_cache=False)

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
            task_folder = os.path.join(task_base_folder, task_group)

            if task_name == "class. del. sum":
                task_data_dict = get_task_results(task_folder, x_param_name='input_duration', y_param_name='input_amplitude',
                                                  aggregation_type='sum_over_delays', metric='accuracy', use_cache=use_cache).to_dict()
            elif task_name in ["class. max. del.", "classification", "classi-\nfication", "class."]:
                task_data_dict = get_task_results(task_folder, x_param_name='input_duration', y_param_name='input_amplitude',
                                                  aggregation_type='max_delay', metric='accuracy', use_cache=use_cache).to_dict()
            elif "NARMA" in task_name:
                task_data_dict = get_task_results(task_folder, x_param_name='input_duration', y_param_name='input_amplitude',
                                                  metric='squared_corr_coeff', use_cache=use_cache).to_dict()
            else:
                task_data_dict = get_task_results(task_folder, x_param_name='input_duration', y_param_name='input_amplitude', use_cache=use_cache).to_dict()

            if cap_info_type in ['nonlin. cap. delay 5', 'nonlinear capacity\ndelay 5', 'nonlin. cap. delay 10',
                                 'nonlinear capacity\ndelay 10'] and task_name not in ['NARMA5', 'NARMA10']:
                corr = 0.
            # elif task_name == 'NARMA10' and '5' in cap_info_type:
            #     corr = 0.
            elif task_name == 'NARMA5' and '10' in cap_info_type:
                corr = 0.
            else:
                corr = get_correlation(cap_data_dict, task_data_dict, use_spearmanr=use_spearmanr)
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
        ax.set_xticklabels(tasknames, rotation=90, fontsize=8)
        ax.tick_params(axis="x", direction="in", pad=-8, length=0.)
        # ax.set_ylabel('correlation coefficient')
    ax.set_title(cap_title)
    ax.spines['top'].set_color('none')
    ax.spines['bottom'].set_color('none')
    ax.spines['right'].set_color('none')
    # ax.legend()
    # ax.legend(ncol=2)  # Poster
    # ax.legend(ncol=2, prop={'size': 6})  # Poster
    # fig.tight_layout()
    # plt.savefig(os.path.join(parameters['fig_folder'], cap_dict['figname']))

fig = plt.figure()
ax = fig.subplots()
plot_single_correlations_plot(
    ax=ax,
    bar_width=0.25,
    cap_dict= {
            "axes_letter": 'tasks1',
            "cap_groupname": "FPUT_capacities",
            "tasks": {
                "XOR": 'xor',
            },
            "figname": 'delete_me.pdf'
        },
    cap_folder="../Data/FPUT_capacities",
    cap_title='',
    task_base_folder='../Data/FPUT_tasks',
    use_spearmanr=False,
    use_cache=False
)

plt.savefig('corr.pdf')

