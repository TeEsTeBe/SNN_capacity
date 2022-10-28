import os
import pickle
from collections import defaultdict
from copy import deepcopy

import numpy as np
import pandas as pd
import seaborn as sns
import yaml
from matplotlib import pyplot as plt

from utils.general_utils import filter_paths, translate, get_param_from_path, get_cached_file, store_cached_file
from colors import get_color


def cap_array(file_path, cutoff=0., mindegree=0, maxdegree=np.inf, mindelay=0, maxdelay=np.inf):
    with open(file_path, 'rb') as f:
        data = pickle.load(f)

    all_capacities = data['all_capacities']
    all_capacities = [cap for cap in all_capacities if cap['degree'] >= mindegree]
    all_capacities = [cap for cap in all_capacities if cap['degree'] <= maxdegree]
    all_capacities = [cap for cap in all_capacities if cap['delay'] >= mindelay]
    all_capacities = [cap for cap in all_capacities if cap['delay'] <= maxdelay]

    capacities = [x['score'] if x['score'] > cutoff else 0 for x in all_capacities]
    # capacities = [x['score'] if x['score'] > cutoff else 0 for x in data['all_capacities']]

    return np.array(capacities)


def max_degree(file_path, cutoff=0.):
    with open(file_path, 'rb') as f:
        data = pickle.load(f)

    capacities = [x for x in data['all_capacities'] if x['score'] > cutoff]
    if len(capacities) == 0:
        # max_degree = np.nan
        max_degree = 0.
    else:
        max_degree = np.max([x['degree'] for x in capacities])

    return max_degree


def max_delay(file_path, cutoff=0., dambre_delay=False):
    with open(file_path, 'rb') as f:
        data = pickle.load(f)

    capacities = [x for x in data['all_capacities'] if x['score'] > cutoff]
    if len(capacities) == 0:
        # max_delay = np.nan
        max_delay = 0.
    elif dambre_delay:
        max_delay = np.max([x['delay'] for x in capacities]) - 1
    else:
        max_delay = np.max([x['delay'] + x['window'] for x in capacities]) - 2

    return max_delay


def get_colorbar_label(plot_max_degrees=False, plot_max_delays=False, plot_num_trials=False,
                       plot_degree_delay_product=False):
    if plot_max_degrees:
        colorbar_label = 'max. degree'
    elif plot_max_delays:
        colorbar_label = 'max. delay'
    elif plot_num_trials:
        colorbar_label = 'num. trials'
    elif plot_degree_delay_product:
        colorbar_label = None
    else:
        colorbar_label = 'capacity'

    return colorbar_label


def get_heatmap_data(x_name, y_name, capacity_folder, params_to_filter, cutoff=0., get_max_degrees=None,
                     get_max_delays=None, get_num_trials=None, other_filter_keys=None, use_cache=True,
                     overwrite_cache=False, mindegree=0, maxdegree=np.inf, mindelay=0, maxdelay=np.inf,
                     avg_fct=np.mean, use_dambre_delay=False):
    params = deepcopy(locals())
    cached_data = get_cached_file(params)
    if use_cache and cached_data is not None:
        avg_results_dict = cached_data
        print('using cached file')
    else:
        dict_paths = [os.path.join(capacity_folder, filename) for filename in os.listdir(capacity_folder)]
        dict_paths = filter_paths(dict_paths, params_to_filter, other_filter_keys=other_filter_keys)
        y_values = np.unique([get_param_from_path(dictpath, y_name) for dictpath in dict_paths])
        x_values = np.unique([get_param_from_path(dictpath, x_name) for dictpath in dict_paths])
        avg_results_dict = {}
        for x in x_values:
            avg_results_dict[x] = {}
            for y in y_values:
                x_y_paths = [dp for dp in dict_paths if f'{x_name}={x}_' in dp and f'{y_name}={y}_' in dp]

                if get_max_degrees:
                    avg_results_dict[x][y] = avg_fct([max_degree(dp) for dp in x_y_paths])
                elif get_max_delays:
                    avg_results_dict[x][y] = avg_fct([max_delay(dp, dambre_delay=use_dambre_delay) for dp in x_y_paths])
                elif get_num_trials:
                    avg_results_dict[x][y] = len(x_y_paths)
                else:
                    capacities = [cap_array(dp, cutoff=cutoff, mindegree=mindegree, maxdegree=maxdegree,
                                            mindelay=mindelay, maxdelay=maxdelay).sum() for dp in x_y_paths]
                    avg_results_dict[x][y] = avg_fct(capacities)
                # avg_results_dict[dur_dict_key][amp_dict_key] = len(capacities)
    if (use_cache and cached_data is None) or overwrite_cache:
        store_cached_file(avg_results_dict, params)

    return avg_results_dict


def plot_heatmap(x_name, y_name, capacity_folder, title, params_to_filter, cutoff, figure_path, plot_max_degrees,
                 plot_max_delays, plot_num_trials, annotate, plot_degree_delay_product=False, ax=None,
                 other_filter_keys=None, cmap=None, mindegree=0, maxdegree=np.inf, mindelay=0, maxdelay=np.inf,
                 use_cache=False):
    if cmap is None:
        if plot_max_degrees:
            cmap = sns.light_palette(get_color('degree'), as_cmap=True)
        elif plot_max_delays:
            cmap = sns.light_palette(get_color('delay'), as_cmap=True)
        elif not plot_degree_delay_product:
            cmap = sns.light_palette(get_color('capacity'), as_cmap=True)
        else:
            cmap = 'rocket'

    if plot_degree_delay_product:
        avg_degrees_dict = get_heatmap_data(x_name, y_name, capacity_folder, params_to_filter, cutoff,
                                            get_max_degrees=True,
                                            other_filter_keys=other_filter_keys, use_cache=use_cache)
        df_degrees = pd.DataFrame.from_dict(avg_degrees_dict)
        avg_delays_dict = get_heatmap_data(x_name, y_name, capacity_folder, params_to_filter, cutoff,
                                           get_max_delays=True,
                                           other_filter_keys=other_filter_keys, use_cache=use_cache)
        df_delays = pd.DataFrame.from_dict(avg_delays_dict)

        # df_degrees -= df_degrees.min().min()
        # df_delays -= df_delays.min().min()
        df_degrees /= df_degrees.max().max()
        df_delays /= df_delays.max().max()

        df = df_degrees * df_delays
    else:
        avg_results_dict = get_heatmap_data(x_name, y_name, capacity_folder, params_to_filter, cutoff, plot_max_degrees,
                                            plot_max_delays, plot_num_trials, other_filter_keys, mindegree=mindegree,
                                            maxdegree=maxdegree, mindelay=mindelay, maxdelay=maxdelay,
                                            use_cache=use_cache)
        df = pd.DataFrame.from_dict(avg_results_dict)

    # fig, ax = plt.subplots(figsize=(4, 3))
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = None
    colorbar_label = get_colorbar_label(plot_max_degrees, plot_max_delays, plot_num_trials, plot_degree_delay_product)
    # ax = sns.heatmap(df, annot=True, annot_kws={"fontsize":6}, cbar_kws={'label': colorbar_label}, ax=ax, fmt=".0f")
    ax = sns.heatmap(df, annot=annotate, cbar_kws={'label': colorbar_label}, ax=ax, fmt=".0f", cmap=cmap)
    ax.invert_yaxis()

    # title = get_title(title, plot_max_degrees, plot_max_delays, plot_num_trials)
    ax.set_title(title)
    ax.set_ylabel(translate(y_name))
    ax.set_xlabel(translate(x_name))

    ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)

    # plt.yticks(rotation=0.)
    # plt.tight_layout()
    if figure_path is not None:
        plt.savefig(figure_path)

    return fig, ax


def plot_task_results_heatmap(results_df, xlabel, ylabel, fig=None, ax=None, title='XOR performance',
                              cbar_label='kappa', vmin=0, vmax=1, cmap='rocket'):
    if None in [fig, ax] and fig != ax:
        raise ValueError(
            "fig and ax can only be set together. Either set both to None (or don't set them) or give a value for both.")
    if fig is None:
        fig, ax = plt.subplots()
    ax = sns.heatmap(results_df, annot=False, cbar_kws={'label': cbar_label}, ax=ax, fmt=".0f", cmap=cmap, vmin=vmin,
                     vmax=vmax)
    ax.invert_yaxis()
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    # fig.tight_layout()

    return fig, ax


def get_task_results(data_dir, x_param_name, y_param_name, aggregation_type='normal', metric='kappa',
                     max_delay_cutoff=0.11):
    implemented_agg_types = ['normal', 'sum_over_delays', 'max_delay']
    assert aggregation_type in implemented_agg_types, f'Aggregation type "{aggregation_type}" not implemented (implemented types: {implemented_agg_types}).'

    result_dict_paths = [os.path.join(data_dir, d, 'test_results.yml') for d in os.listdir(data_dir)]
    x_param_values = np.unique(
        [get_param_from_path(r, param_name=x_param_name, cast_function=float) for r in result_dict_paths])
    y_param_values = np.unique(
        [get_param_from_path(r, param_name=y_param_name, cast_function=float) for r in result_dict_paths])
    results_dict = defaultdict(dict)
    for x_value in x_param_values:
        print(f'\n{x_param_name}: {x_value}')
        print(f'\t{y_param_name}: ', end='')
        for y_value in y_param_values:
            print(f'{y_value}, ', end='')
            filtered_dict_paths = [d for d in result_dict_paths if
                                   get_param_from_path(d, param_name=x_param_name, cast_function=float) == x_value]
            filtered_dict_paths = [d for d in filtered_dict_paths if
                                   get_param_from_path(d, param_name=y_param_name, cast_function=float) == y_value]
            if aggregation_type == 'normal':
                task_values = [get_metric_value(d, metric=metric) for d in filtered_dict_paths]
                results_dict[x_value][y_value] = np.mean(task_values)
            else:
                task_values = []
                delay_list = np.unique(
                    [get_param_from_path(d, param_name='DELAY', cast_function=int) for d in filtered_dict_paths])
                # print(f'\t\tDelays: {delay_list}')
                for delay in delay_list:
                    filtered_delay_dict_paths = [d for d in filtered_dict_paths if
                                                 get_param_from_path(d, param_name='DELAY',
                                                                     cast_function=int) == delay]
                    delay_task_values = [get_metric_value(d, metric=metric) for d in filtered_delay_dict_paths]
                    task_values.append(np.mean(delay_task_values))

                if aggregation_type == 'sum_over_delays':
                    results_dict[x_value][y_value] = np.sum(np.array(task_values) - max_delay_cutoff)
                else:  # max_delay
                    bigger_than_cutoff = list(np.array(task_values) > max_delay_cutoff)
                    if False in bigger_than_cutoff:
                        max_delay = bigger_than_cutoff.index(False)
                    else:
                        max_delay = len(task_values) - 1
                        print(f'\t\tNo value below cutoff. Min: {np.min(task_values)}')
                        print(f'\t\tAll values: {task_values}')
                    results_dict[x_value][y_value] = max_delay

    results_df = pd.DataFrame.from_dict(results_dict)
    return results_df


def get_metric_value(result_yaml_path, metric='kappa'):
    with open(result_yaml_path, 'r') as result_file:
        result_dict = yaml.safe_load(result_file)

    return result_dict[metric]
