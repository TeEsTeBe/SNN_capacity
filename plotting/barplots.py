import copy
import os
import pickle
from copy import deepcopy

import numpy as np
from colour import Color
from matplotlib import pyplot as plt

# import utils
from plotting.colors import get_degree_color
from SNN.utils.general_utils import translate, get_param_from_path, filter_paths, get_cached_file, store_cached_file


def plot_bars_delay_level(dict_list, names, title='Capacities', savepath=None, delay_shading_step=10,
                          minimum_for_text=-1.,
                          xlabel='density of recurrent connections', figsize=None, xlabelrotation=None, dont_show=False,
                          part_cap_text_color='white', ax=None, label_fontsize=None, title_fontsize=None,
                          part_cap_text_fontsize=None, annotate_sums=False, disable_delay_shading=True):
    """
    Input: a list of delay level dicts
    ```
    dict[degree][delay] = CAPSUM
    ```

    Output: a plot with capacity bars (separated for different degrees) for every dict. Every bar is also separated into one bar for every delay.

    """

    # deg_delay_caplist = lambda dictlist, degree, delay: [d[degree][delay] for d in dictlist]
    def deg_delay_caplist(dictlist, degree, delay):
        caplist = []
        for d in dictlist:
            if degree not in d.keys():
                d[degree] = {}
            if delay not in d[degree].keys():
                d[degree][delay] = 0
            caplist.append(d[degree][delay])
        return caplist

    n_bars = len(names)

    if ax is None:
        if figsize is None:
            fig, ax = plt.subplots()
        else:
            fig, ax = plt.subplots(figsize=figsize)

    # degrees = sorted(dict_list[0].keys()) #[:1]
    # delays = sorted(dict_list[0][1].keys()) #[:11]
    degrees = []
    delays = []
    for dict_ in dict_list:
        degrees.extend(dict_.keys())
        for delay_dict in dict_.values():
            delays.extend(delay_dict.keys())
    degrees = np.unique(degrees)
    delays = np.unique(delays)

    # degrees = sorted(dict_list[0].keys()) #[:1]
    # delays = sorted(dict_list[0][1].keys()) #[:11]

    # colors = sns.color_palette()
    # colors = degree_colors
    # hsv = plt.get_cmap('hsv')
    # colors = hsv(np.linspace(0, 1., np.max(delays)))
    #     saturations =  np.interp(delays, (np.min(delays), np.max(delays)), (1., 0.6))
    saturations = np.interp(delays, (np.min(delays), np.max(delays)), (0.85, 0.3))

    total_sums = np.zeros(len(names))
    last_degree_sums = np.zeros(n_bars)
    last_part_sums = np.zeros(n_bars)

    for degree in degrees:
        # degree_color = Color(rgb=colors[(degree - 1) % len(colors)])
        degree_color = Color(get_degree_color(degree))
        if not disable_delay_shading:
            degree_color.saturation = saturations[0]

        current_degree_sums = np.zeros(n_bars)
        current_part_sums = np.zeros(n_bars)

        for delay in delays:
            capacities = np.array(deg_delay_caplist(dict_list, degree, delay))

            total_sums += capacities
            current_degree_sums += capacities
            current_part_sums += capacities

            if np.max(capacities) > 0:
                if delay % delay_shading_step == 0 or delay == delays[-1]:
                    delay_color = Color(degree_color.hex)
                    if not disable_delay_shading:
                        if delay_shading_step <= 0:
                            delay_color.saturation = saturations[0]
                            delay_color.luminance = saturations[0] / 2
                        else:
                            delay_color.saturation = saturations[delay]
                            delay_color.luminance = saturations[delay] / 2
                    if delay == 0:
                        ax.bar(range(n_bars), current_part_sums, color=delay_color.rgb,
                               tick_label=names, bottom=last_part_sums, linewidth=0, label=f'Degree {degree}')
                    #                     plt.bar(range(n_bars), current_part_sums,
                    #                         tick_label=names, bottom=last_part_sums, linewidth=0, label=f'Degree {degree}')
                    else:
                        ax.bar(range(n_bars), current_part_sums, color=delay_color.rgb,
                               tick_label=names, bottom=last_part_sums, linewidth=0)
                    #                     plt.bar(range(n_bars), current_part_sums,
                    #                         tick_label=names, bottom=last_part_sums, linewidth=0)
                    current_part_sums = np.zeros(n_bars)
                    last_part_sums = total_sums.copy()

        for i, (current_degree, total) in enumerate(zip(current_degree_sums, total_sums)):
            if current_degree > minimum_for_text:
                # if part_cap_text_color is None:
                #     ax.text(i, total - current_degree/2, round(current_degree, 2), ha='center', va='center')
                # else:
                #     ax.text(i, total - current_degree/2, round(current_degree, 2), ha='center', va='center', color=part_cap_text_color)
                ax.text(i, total - current_degree / 2, round(current_degree, 2), fontsize=part_cap_text_fontsize,
                        ha='center', va='center', color=part_cap_text_color)

        last_degree_sum = total_sums.copy()

    max_y = 0
    for i, s in enumerate(total_sums):
        y_val = s * 1.01
        if annotate_sums:
            ax.text(i, y_val, round(s, 2), ha='center', va='bottom', fontsize=part_cap_text_fontsize)
        max_y = max(max_y, y_val)

    print(f'max capacity: {max_y}; max degree: {np.max(degrees)}; max delay: {np.max(delays)}')

    ax.set_ylim(0, max_y * 1.05)
    # if change_fontsize:
    if label_fontsize is not None:
        ax.set_ylabel('capacity', fontsize=label_fontsize)
        ax.set_xlabel(xlabel, fontsize=label_fontsize)
    else:
        ax.set_ylabel('capacity')
        ax.set_xlabel(xlabel)

    if title_fontsize is not None:
        ax.set_title(title, fontsize=title_fontsize)
    else:
        ax.set_title(title)

    # else:
    #     ax.set_ylabel('capacity')
    #     ax.set_xlabel(xlabel)
    #     ax.set_title(title)

    # ax.legend()
    if xlabelrotation is not None:
        ax.tick_params('x', labelrotation=xlabelrotation)

    if savepath is not None:
        if not os.path.exists(os.path.dirname(savepath)):
            os.makedirs(os.path.dirname(savepath))
        plt.savefig(savepath)

    if not dont_show:
        plt.tight_layout()
        plt.show()

    return ax


def degree_delay_fct_dambre(dict_path, fct=np.sum, max_degree=None, max_delay=None, use_dambre_delay=False):
    """
    Same function as degree_delay_fct, but for dictionaries which are computed by the original Dambre code.
    """
    with open(dict_path, 'rb') as results_file_dambre:
        results_dambre = pickle.load(results_file_dambre)
    dambre_cap_list = results_dambre['all_capacities']

    dambre_dict = {}
    for cap_item in dambre_cap_list:
        degree = cap_item['degree']
        if use_dambre_delay:
            delay = cap_item['delay'] - 1
        else:
            delay = cap_item['delay'] + cap_item['window'] - 2

        if degree not in dambre_dict.keys():
            dambre_dict[degree] = {}
        if delay not in dambre_dict[degree].keys():
            dambre_dict[degree][delay] = []
        dambre_dict[degree][delay].append(cap_item['score'])

    if max_degree is not None and max_delay is not None:
        for degree in range(1, max_degree + 1):
            if degree not in dambre_dict.keys():
                dambre_dict[degree] = {}
            for delay in range(max_delay + 1):
                if delay not in dambre_dict[degree].keys():
                    dambre_dict[degree][delay] = 0.

    for degree, degree_dict in copy.deepcopy(dambre_dict).items():
        for delay, cap_list in copy.deepcopy(degree_dict).items():
            cap_value = fct(cap_list)
            if cap_value > 0.:
                dambre_dict[degree][delay] = cap_value
            else:
                del dambre_dict[degree][delay]
        if len(dambre_dict[degree].keys()) == 0:
            del dambre_dict[degree]

    return dambre_dict


def average_capacity_dicts(dict_list, cutoff=0, avg_fct=np.mean):
    """
    input: [cap_dict1, cap_dict2, ...]
    output: averaged_cap_dict

    dict structure: dict[Degree][Delay] = Capacity

    """

    max_degree = -1
    max_delay = -1
    for cap_dict in dict_list:
        if len(list(cap_dict.keys())) == 0:
            cap_dict = {0: {0: 0}}
        max_degree = max(max_degree, np.max(list(cap_dict.keys())))
        for degree, delay_dict in cap_dict.items():
            max_delay = max(max_delay, np.max(list(delay_dict.keys())))

    averaged_dict = {}
    for degree in range(1, max_degree + 1):
        averaged_dict[degree] = {}
        for delay in range(max_delay + 1):
            for cap_dict in dict_list:
                if degree not in cap_dict.keys():
                    cap_dict[degree] = {}
                if delay not in cap_dict[degree].keys():
                    cap_dict[degree][delay] = 0
            averaged_dict[degree][delay] = avg_fct([d[degree][delay] for d in dict_list])
            if averaged_dict[degree][delay] < cutoff:
                averaged_dict[degree][delay] = 0.

    return averaged_dict


def default_cast_function(value):
    if value.isnumeric():
        casted_value = int(value)
    else:
        try:
            casted_value = float(value)
        except:
            casted_value = str(value)

    return casted_value


def plot_capacity_bars(x_name, capacity_folder, title, params_to_filter, cutoff, delay_shading_step, annotate,
                       annotate_sums, ax=None, other_filter_keys=None, disable_legend=False, use_cache=False,
                       overwrite_cache=False, precalculated_data_path=None):
    capacity_dict_paths = [os.path.join(capacity_folder, filename) for filename in os.listdir(capacity_folder)]
    capacity_dict_paths = filter_paths(capacity_dict_paths, params_to_filter, other_filter_keys=other_filter_keys)
    x_values = np.unique([get_param_from_path(dictpath, x_name) for dictpath in capacity_dict_paths])

    params = deepcopy(locals())
    cached_data = get_cached_file(params)
    if precalculated_data_path is not None:
        with open(precalculated_data_path, 'rb') as data_file:
            averaged_capacities_per_x = pickle.load(data_file)
    elif use_cache and cached_data is not None:
        averaged_capacities_per_x = cached_data
        print('using cached file')
    else:
        averaged_capacities_per_x = []
        for x in x_values:
            x_paths = [dp for dp in capacity_dict_paths if f"{x_name}={x}_" in dp]
            capacity_dicts = [degree_delay_fct_dambre(xp) for xp in x_paths]
            averaged_capacity_dict = average_capacity_dicts(capacity_dicts, cutoff=cutoff)
            averaged_capacities_per_x.append(averaged_capacity_dict)
    if precalculated_data_path is None and ((use_cache and cached_data is None) or overwrite_cache):
        store_cached_file(averaged_capacities_per_x, params)

    if annotate:
        minimum_for_text = 1.5
    else:
        minimum_for_text = 99999999999
    ax = plot_bars_delay_level(averaged_capacities_per_x, names=x_values, title=title, xlabel=translate(x_name),
                               minimum_for_text=minimum_for_text, dont_show=True, delay_shading_step=delay_shading_step,
                               annotate_sums=annotate_sums, ax=ax)
    if not disable_legend:
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles[::-1], labels[::-1])
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90)

    return ax
