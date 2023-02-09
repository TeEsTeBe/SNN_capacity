import os
import pickle
import argparse

import matplotlib.pyplot as plt
plt.rc('axes', labelsize=14)
ticksize = 12
plt.rc('xtick', labelsize=ticksize)
plt.rc('ytick', labelsize=ticksize)
import matplotlib as mpl
mpl.rcParams["text.usetex"] = True
from matplotlib.patches import Rectangle
import numpy as np

from heatmaps import plot_heatmap, get_heatmap_data, get_task_results, plot_task_results_heatmap
from colors import get_color
from evaluation.calculate_task_correlations import get_correlation
import cap_bars_single_run


def data_path():
    return '/home/schultetobrinke/projects/recurrence/repos/capacity_visualisation/capacity_visualisation/data'


def get_capacity_directory(step_spatial_or_uniform, random_or_frozen, rate_or_DC, p=None, std=None):
    assert step_spatial_or_uniform in ['step', 'spatial', 'uniform'], 'step_spatial_or_uniform should be "step", "spatial" or "uniform"'
    assert random_or_frozen in ['randomnoise', 'frozennoise'], 'random_or_frozen should be "randomnoise" or "frozennoise"'
    assert rate_or_DC in ['rate', 'DC'], 'rate_or_DC should be "rate" or "DC"'
    assert (p is not None or std is not None), 'p or std should be set'

    if step_spatial_or_uniform == 'step':
        if random_or_frozen == 'frozennoise':
            folder_name = f'frozennoise_dur1-50_max0.2-3.0__net=brunel__p={p}__inp=step_{rate_or_DC}__steps=200000'
        else:
            if rate_or_DC == 'rate':
                folder_name = f'randomnoise_rate_dur1-50_max200-3000__net=brunel__inp=step_rate__p={p}__steps=200000'
            else:
                folder_name = f'randomnoise_dur1-50_max0.2-3.0__net=brunel__inp=step_DC__p={p}__steps=200000'
    elif step_spatial_or_uniform == 'uniform':
        noise_loop_dur = 'step_duration' if random_or_frozen == 'frozennoise' else 'None'
        folder_name = f'uniform-encoding-fullscan__inp=uniform_{rate_or_DC}__net=brunel__g=5.0__J=0.2__p={p}__noise_loop_duration={noise_loop_dur}'
    else:
        folder_name = f'spatial_{random_or_frozen}_dur1-50_max0.2-3.0__net=brunel__std={std}__inp=spatial_{rate_or_DC}__steps=200000'

    capacity_directory = os.path.join(data_path(), folder_name, 'capacity')

    return capacity_directory


def get_max(avg_cap_results, return_keys=False):
    capacity_list = []
    durations_and_amplitudes = []
    for duration, amplitude_cap_map in avg_cap_results.items():
        for amplitude, capacity in amplitude_cap_map.items():
            capacity_list.append(capacity)
            durations_and_amplitudes.append({'duration': duration, 'amplitude': amplitude})

    max_idx = np.nanargmax(capacity_list)
    max_capacity = capacity_list[max_idx]

    if return_keys:
        max_keys = durations_and_amplitudes[max_idx]
        return max_capacity, max_keys
    else:
        return max_capacity


def add_capacity_percent_twinx(ax, N=1000, add_label=False):
    axtwin = ax.twinx()
    ytwin_ticklabels = [int(100 * t / N) for t in ax.get_yticks()][1:]
    ytwin_tickpositions = [N * t / 100. for t in ytwin_ticklabels]
    axtwin.set_yticks(ytwin_tickpositions, labels=ytwin_ticklabels)
    if add_label:
        axtwin.set_ylabel("\% of max.", size=12)
    ylim = ax.get_ylim()
    axtwin.set_ylim(ylim)

def plot_max_cap_per_p_or_std(step_spatial_or_uniform, ax=None, plot_degrees=False, plot_memory=False, use_cache=False,
                              plot_stds=True, use_label=False, use_precalculated=True):
    colors = {
        'DC': '#E84855',
        'rate': '#403F4C',
    }
    linestyles = {
        'DC': {
            'frozennoise': '-',
            'randomnoise': (1., (5, 5)),
        },
        'rate': {
            'frozennoise': '-',
            'randomnoise': (4., (5, 5)),
        }
    }
    if ax is None:
        _, ax = plt.subplots()
    if step_spatial_or_uniform == 'spatial':
        p_or_std_values = [1.0, 2.0, 5.0, 10.0, 15.0, 20.0]
    else:
        p_or_std_values = [0.25, 0.5, 0.75, 1.0]
    if plot_memory:
        data_type = 'memory'
    else:
        data_type = 'capacity'

    if use_precalculated:
        with open(os.path.join('data', f'BRN_encoding_lines_{step_spatial_or_uniform}_{data_type}.pkl'), 'rb') as data_file:
            precalculated_data = pickle.load(data_file)
    for random_or_frozen in ['frozennoise', 'randomnoise']:
        for rate_or_DC in ['DC', 'rate']:
            if use_precalculated:
                max_capacities = precalculated_data[random_or_frozen][rate_or_DC]['max_values']
                stds = precalculated_data[random_or_frozen][rate_or_DC]['stds']
                p_or_std_values_with_data = precalculated_data[random_or_frozen][rate_or_DC]['p_or_std_values_with_data']
            else:
                max_capacities = []
                p_or_std_values_with_data = []
                stds = []  # standard error of mean
                for p_or_std in p_or_std_values:
                    if step_spatial_or_uniform == 'spatial':
                        p = None
                        std = p_or_std
                    else:
                        p = p_or_std
                        std = None
                    cap_dir = get_capacity_directory(step_spatial_or_uniform=step_spatial_or_uniform, rate_or_DC=rate_or_DC,
                                                     random_or_frozen=random_or_frozen, p=p, std=std)
                    params_to_filter = {}
                    try:
                        avg_cap_results = get_heatmap_data('dur', 'max', cap_dir, params_to_filter=params_to_filter,
                                                           get_max_degrees=plot_degrees, get_max_delays=plot_memory,
                                                           other_filter_keys=['vm'], use_cache=use_cache)
                        std_cap_results = get_heatmap_data('dur', 'max', cap_dir, params_to_filter=params_to_filter,
                                                           get_max_degrees=plot_degrees, get_max_delays=plot_memory,
                                                           other_filter_keys=['vm'], use_cache=use_cache,
                                                           avg_fct=np.nanstd)  # scipy.stats.sem)
                        if plot_memory:
                            for duration, max_delay_dict in avg_cap_results.items():
                                for max_amplitude, delay in max_delay_dict.items():
                                    avg_cap_results[duration][max_amplitude] = duration * delay
                        max_cap, dur_and_amp = get_max(avg_cap_results, return_keys=True)
                        duration = dur_and_amp['duration']
                        amplitude = dur_and_amp['amplitude']
                        max_capacities.append(max_cap)
                        stds.append(std_cap_results[duration][amplitude])
                        p_or_std_values_with_data.append(p_or_std)
                    except FileNotFoundError:
                        print(
                            f"Capacity data for {step_spatial_or_uniform}_{rate_or_DC}_p={p_or_std} ({random_or_frozen}) does not exist "
                            f"and will not be added to the data!")
                    except BaseException as err:
                        print(f"Unexpected {type(err)}: {err}")

            if len(p_or_std_values_with_data) == 0:
                print(
                    f"No data for {step_spatial_or_uniform}_{rate_or_DC} ({random_or_frozen}) does exist for any p ({p_or_std_values})."
                    f" This won't be part of the plot!")
            else:
                if len(p_or_std_values_with_data) == 1:
                    half_width = 0.025
                    p_or_std_values_with_data = [p_or_std_values_with_data[0] - half_width,
                                                 p_or_std_values_with_data[0] + half_width]
                    max_capacities = [max_capacities[0], max_capacities[0]]
                    stds = [stds[0], stds[0]]
                stds = np.array(stds)
                max_capacities = np.array(max_capacities)

                if plot_stds:
                    ax.fill_between(p_or_std_values_with_data, max_capacities + stds, max_capacities - stds,
                                    color=colors[rate_or_DC], alpha=0.3)
                linewidth = 1  # 3  # Poster
                if not use_label:
                    label = None
                elif random_or_frozen == 'randomnoise':
                    label = f'{rate_or_DC}, changing noise'
                else:
                    label = f'{rate_or_DC}, frozen noise'
                ax.plot(p_or_std_values_with_data, max_capacities, label=label,
                        linestyle=linestyles[rate_or_DC][random_or_frozen], color=colors[rate_or_DC], lw=linewidth)

    ax.set_xticks(p_or_std_values)
    if step_spatial_or_uniform == 'spatial':
        ax.set_xlabel(r'$\sigma$')
    else:
        ax.set_xlabel(r'$p$')

    if plot_degrees:
        ylabel = 'degree'
    elif plot_memory:
        ylabel = 'memory [ms]'
    else:
        ylabel = 'capacity'
    ax.set_ylabel(ylabel)

    return ax


def create_xor_subplot(ax, fig, data_dir):
    results_df = get_task_results(data_dir, x_param_name='dur', y_param_name='max',
                                  aggregation_type='normal', metric='kappa')
    plot_task_results_heatmap(results_df, xlabel='$\Delta s$', ylabel='$a_{max}$', title="XOR $\sigma 1$",
                              vmin=0., vmax=1.,
                              cbar_label='kappa score', ax=ax, fig=fig)

    ax.set_xticklabels([])
    ax.set_yticklabels([])

    return ax


def set_color_markers(axes):
    barmarker_sigma1 = get_color('XOR')
    barmarker_sigma20 = get_color('accent')

    for ax_label in ['J', 'K', 'L']:
        axes[ax_label].add_patch(Rectangle((1, 14), 1, 1, fill=False, edgecolor=barmarker_sigma1, lw=2, clip_on=False))
    for ax_label in ['E', 'F', 'G']:  # , 'X']:
        axes[ax_label].add_patch(Rectangle((7, 0), 1, 1, fill=False, edgecolor=barmarker_sigma20, lw=2, clip_on=False))

    axes['bar1'].spines['bottom'].set_color(barmarker_sigma1)
    axes['bar1'].spines['bottom'].set_linewidth(3)
    axes['bar1'].spines['top'].set_color(barmarker_sigma1)
    axes['bar1'].spines['top'].set_linewidth(3)
    axes['bar1'].spines['right'].set_color(barmarker_sigma1)
    axes['bar1'].spines['right'].set_linewidth(3)
    axes['bar1'].spines['left'].set_color(barmarker_sigma1)
    axes['bar1'].spines['left'].set_linewidth(3)

    axes['bar2'].spines['bottom'].set_color(barmarker_sigma20)
    axes['bar2'].spines['bottom'].set_linewidth(3)
    axes['bar2'].spines['top'].set_color(barmarker_sigma20)
    axes['bar2'].spines['top'].set_linewidth(3)
    axes['bar2'].spines['right'].set_color(barmarker_sigma20)
    axes['bar2'].spines['right'].set_linewidth(3)
    axes['bar2'].spines['left'].set_color(barmarker_sigma20)
    axes['bar2'].spines['left'].set_linewidth(3)


def plot_bars(axes):
    cap_dict_path_1 = os.path.join('data', 'BRN_capbars_std1.pkl')
    with open(cap_dict_path_1, 'rb') as cap_file_enc:
        cap_dict_enc = pickle.load(cap_file_enc)
    cap_bars_single_run.plot_capacity_bars(cap_dict_enc, ax=axes['bar2'])
    axes['bar2'].set_xticks([0])
    axes['bar2'].set_xticklabels([0])
    xmin, xmax = axes['bar2'].get_xlim()
    axes['bar2'].set_xlim((xmin * 2, xmax * 2))
    axes['bar2'].set_ylabel(None)
    axes['bar2'].set_xlabel(None)
    cap_dict_path_20 = os.path.join('data', 'BRN_capbars_std20.pkl')
    with open(cap_dict_path_20, 'rb') as cap_file_20:
        cap_dict_20 = pickle.load(cap_file_20)
    cap_bars_single_run.plot_capacity_bars(cap_dict_20, ax=axes['bar1'])
    axes['bar1'].set_xlabel('delay', labelpad=-10)
    axes['bar1'].set_xticks([0, 10])
    axes['bar2'].set_ylim((0, 830.27))


def plot_encoder_capacity_bars(ax):
    cap_dict_path_enc = os.path.join('data', 'BRN_capbars_enc.pkl')
    with open(cap_dict_path_enc, 'rb') as cap_file_enc:
        cap_dict_enc = pickle.load(cap_file_enc)
    cap_bars_single_run.plot_capacity_bars(cap_dict_enc, ax=ax)
    ax.set_xticks([0])
    ax.set_xticklabels([0])
    xmin, xmax = ax.get_xlim()
    ax.set_xlim((xmin * 2, xmax * 2))
    ax.set_title('encoder')
    ax.set_ylabel(None)
    ax.set_yticklabels(["", 500])
    ax.set_xlabel(None)

    return ax


def plot_heatmaps(axes, use_cache, use_precalculated=True):
    # DC spatial frozen , std 1 and 20
    cap_folder = get_capacity_directory('spatial', 'frozennoise', 'DC', std=1.0)
    params_to_filter = {}

    if use_precalculated:
        precalculated_data_path = os.path.join('data', 'BRN_capacities_std1.pkl')
    else:
        precalculated_data_path = None
    _, axes['E'] = plot_heatmap('dur', 'max', capacity_folder=cap_folder, title=r'Total Capacity',
                                params_to_filter=params_to_filter, ax=axes['E'],
                                cutoff=0., figure_path=None, plot_max_degrees=False, plot_max_delays=False,
                                plot_num_trials=False,
                                annotate=False, other_filter_keys=['vm'], use_cache=use_cache, colorbar_label='',
                                precalculated_data_path=precalculated_data_path)
    axes['E'].set_title('Total Capacity', pad=12)
    axes['E'].set_xticklabels([])
    axes['E'].set_xticks(np.arange(0.5, 7.6, 1))
    axes['E'].set_yticklabels([f'{0.2 * float(n.get_text()):.2f}' for n in axes['E'].get_yticklabels()])
    axes['E'].set_xlabel(None)

    if use_precalculated:
        precalculated_data_path = os.path.join('data', 'BRN_degrees_std1.pkl')
    else:
        precalculated_data_path = None
    _, axes['F'] = plot_heatmap('dur', 'max', capacity_folder=cap_folder, title='Max. Degrees',
                                params_to_filter=params_to_filter, ax=axes['F'],
                                cutoff=0., figure_path=None, plot_max_degrees=True, plot_max_delays=False,
                                plot_num_trials=False,
                                annotate=False, other_filter_keys=['vm'], use_cache=use_cache, colorbar_label='',
                                precalculated_data_path=precalculated_data_path)
    axes['F'].set_title('Max. Degrees', pad=12)
    axes['F'].set_ylabel(None)
    axes['F'].set_xlabel(None)
    axes['F'].set_yticklabels([])
    axes['F'].set_yticks([])
    axes['F'].set_xticklabels([])
    axes['F'].set_xticks(np.arange(0.5, 7.6, 1))

    if use_precalculated:
        precalculated_data_path = os.path.join('data', 'BRN_delays_std1.pkl')
    else:
        precalculated_data_path = None
    _, axes['G'] = plot_heatmap('dur', 'max', capacity_folder=cap_folder, title='Max. Delays',
                                params_to_filter=params_to_filter, ax=axes['G'],
                                cutoff=0., figure_path=None, plot_max_degrees=False, plot_max_delays=True,
                                plot_num_trials=False,
                                annotate=False, other_filter_keys=['vm'], use_cache=use_cache, colorbar_label='',
                                precalculated_data_path=precalculated_data_path)
    axes['G'].set_title('Max. Delays', pad=12)
    axes['G'].set_ylabel(None)
    axes['G'].set_xlabel(None)
    axes['G'].set_yticklabels([])
    axes['G'].set_yticks([])
    axes['G'].set_xticks(np.arange(0.5, 7.6, 1))
    axes['G'].set_xticklabels([])
    if use_precalculated:
        precalculated_data_path = os.path.join('data', 'BRN_capacities_std20.pkl')
    else:
        precalculated_data_path = None
    cap_folder = get_capacity_directory('spatial', 'frozennoise', 'DC', std=20.0)
    _, axes['J'] = plot_heatmap('dur', 'max', capacity_folder=cap_folder, title=r'Total Capacity ($\sigma$ 20.0)',
                                params_to_filter=params_to_filter, ax=axes['J'],
                                cutoff=0., figure_path=None, plot_max_degrees=False, plot_max_delays=False,
                                plot_num_trials=False,
                                annotate=False, other_filter_keys=['vm'], use_cache=use_cache, colorbar_label='',
                                precalculated_data_path=precalculated_data_path)
    axes['J'].set_title(None)
    axes['J'].set_xticks(np.arange(0.5, 7.6, 1))
    axes['J'].set_xticklabels([1, 2, 5, 10, 20, 30, 40, 50], rotation=90.)
    axes['J'].set_yticklabels([f'{0.2 * float(n.get_text()):.2f}' for n in axes['J'].get_yticklabels()])

    if use_precalculated:
        precalculated_data_path = os.path.join('data', 'BRN_degrees_std20.pkl')
    else:
        precalculated_data_path = None
    _, axes['K'] = plot_heatmap('dur', 'max', capacity_folder=cap_folder, title=r'Max. Degrees ($\sigma$ 20.0)',
                                params_to_filter=params_to_filter, ax=axes['K'],
                                cutoff=0., figure_path=None, plot_max_degrees=True, plot_max_delays=False,
                                plot_num_trials=False,
                                annotate=False, other_filter_keys=['vm'], use_cache=use_cache, colorbar_label='',
                                precalculated_data_path=precalculated_data_path)
    axes['K'].set_title(None)
    axes['K'].set_ylabel(None)
    axes['K'].set_xticks(np.arange(0.5, 7.6, 1))
    axes['K'].set_xticklabels([1, 2, 5, 10, 20, 30, 40, 50], rotation=90.)
    axes['K'].set_yticklabels([])
    axes['K'].set_yticks([])

    if use_precalculated:
        precalculated_data_path = os.path.join('data', 'BRN_delays_std20.pkl')
    else:
        precalculated_data_path = None
    _, axes['L'] = plot_heatmap('dur', 'max', capacity_folder=cap_folder, title=r'Max. Delays ($\sigma$ 20.0)',
                                params_to_filter=params_to_filter, ax=axes['L'],
                                cutoff=0., figure_path=None, plot_max_degrees=False, plot_max_delays=True,
                                plot_num_trials=False,
                                annotate=False, other_filter_keys=['vm'], use_cache=use_cache, colorbar_label='',
                                precalculated_data_path=precalculated_data_path)
    axes['L'].set_title(None)
    axes['L'].set_ylabel(None)
    axes['L'].set_yticklabels([])
    axes['L'].set_yticks([])
    axes['L'].set_xticks(np.arange(0.5, 7.6, 1))
    axes['L'].set_xticklabels([1, 2, 5, 10, 20, 30, 40, 50], rotation=90.)


def plot_encoding_line_graphs(axes, use_cache, use_precalculated=True):
    plot_stds = True
    axes['1'] = plot_max_cap_per_p_or_std('spatial', axes['1'], use_cache=use_cache, plot_stds=plot_stds, use_precalculated=use_precalculated)
    add_capacity_percent_twinx(ax=axes['1'], add_label=True)
    axes['1'].scatter([1], [686.4552198555509], marker='o', facecolor='none', s=75, linewidth=1, edgecolor='black', zorder=10)
    axes['1'].scatter([20], [100.54747580302804], marker='s', facecolor='none', s=75, linewidth=1, edgecolor='black', zorder=10)
    xmin, xmax = axes['1'].get_xlim()
    axes['1'].set_xlim(xmin-0.5, xmax+0.5)
    ymin, ymax = axes['1'].get_ylim()
    axes['1'].set_ylim(ymin, ymax+25)
    axes['1'].set_title('spatial')
    axes['1'].set_xlabel(None)
    axes['1'].set_xticklabels([])
    axes['2'] = plot_max_cap_per_p_or_std('spatial', axes['2'], plot_memory=True, use_cache=use_cache,
                                          plot_stds=plot_stds, use_precalculated=use_precalculated)
    axes['2'].scatter([1], [30.], marker='o', facecolor='none', s=75, linewidth=1, edgecolor='black', zorder=10)
    axes['2'].scatter([20], [80.], marker='s', facecolor='none', s=75, linewidth=1, edgecolor='black', zorder=10)
    xmin, xmax = axes['2'].get_xlim()
    axes['2'].set_xlim(xmin-0.5, xmax+0.5)
    ymin, ymax = axes['2'].get_ylim()
    axes['2'].set_ylim(ymin, ymax+3)
    axes['3'] = plot_max_cap_per_p_or_std('step', axes['3'], use_cache=use_cache, plot_stds=plot_stds, use_precalculated=use_precalculated)
    add_capacity_percent_twinx(ax=axes['3'])
    axes['3'].set_title('amplitude')
    axes['3'].set_xlabel(None)
    axes['3'].set_xticklabels([])
    axes['4'] = plot_max_cap_per_p_or_std('step', axes['4'], plot_memory=True, use_cache=use_cache, plot_stds=plot_stds, use_label=True, use_precalculated=use_precalculated)
    axes['4'].set_xticklabels([0.25, '', '', 1.0])
    axes['4'].set_xlabel(r'$p$', labelpad=-6)
    legend = axes['4'].legend(ncol=1, loc='upper left', bbox_to_anchor=(0.15, -0.50), fontsize=10, handlelength=3)
    for legobj in legend.legendHandles:
        legobj.set_linewidth(2.5)
    axes['5'] = plot_max_cap_per_p_or_std('uniform', ax=axes['5'], plot_memory=False, use_cache=use_cache,
                                          plot_stds=plot_stds, use_precalculated=use_precalculated)
    add_capacity_percent_twinx(ax=axes['5'])
    axes['5'].set_title('distributed')
    axes['5'].set_ylabel(None)
    axes['5'].set_xlabel(None)
    axes['5'].set_xticklabels([])
    axes['6'] = plot_max_cap_per_p_or_std('uniform', ax=axes['6'], plot_memory=True, use_cache=use_cache,
                                          plot_stds=plot_stds, use_label=True, use_precalculated=use_precalculated)
    axes['6'].set_ylabel(None)
    axes['6'].set_xticklabels([0.25, '', '', 1.0])
    axes['6'].set_xlabel(r'$p$', labelpad=-6)


def plot_task_correlations(axes, use_cache, use_precalculated=True):
    cap_base_folder = '/home/schultetobrinke/nextcloud/Juelich/projects/recurrence/repos/capacity_visualisation/capacity_visualisation/data'
    task_base_folder = '/home/schultetobrinke/projects/SNN_capacity/repos/ESN/scripts'
    use_spearmanr = False
    bar_width = 0.25

    cap_to_tasks_dict = {
        "distributed": {
            "axes_letter": 'tasks1',
            "cap_groupname": "uniform-encoding-fullscan__inp=uniform_DC__net=brunel__g=5.0__J=0.2__p=1.0__noise_loop_duration=step_duration",
            "tasks": {
                "XOR": 'BRN-scan-uniformXOR-test__net=brunel__inp=uniform_DC_XOR__p=1.0__noise_loop_duration=step_duration__g=5.0__J=0.2',
                "tXOR": 'BRN-scan-uniform-discrete-temporal-XOR-test__net=brunel__inp=uniform_DC_temporal_XOR__p=1.0__noise_loop_duration=step_duration__g=5.0__J=0.2',
                "XORXOR": 'BRN-scan-uniform-XORXOR-test__net=brunel__inp=uniform_DC_XORXOR__p=1.0__noise_loop_duration=step_duration__g=5.0__J=0.2',
                "class.": 'BRN-scan-uniform-classification-test__net=brunel__inp=uniform_DC_classification__p=1.0__noise_loop_duration=step_duration__g=5.0__J=0.2',
                "NARMA5": 'NARMA5_BRN-scan-uniform-NARMA10-test__net=brunel__inp=uniform_DC__p=1.0__noise_loop_duration=step_duration__g=5.0__J=0.2',
            },
            "figname": 'cap-task-correlations_no-ctXOR_uniform_p=1.0.narma5.no-delsum.newcolors.pdf'
        },
        "spatial": {
            "axes_letter": 'tasks2',
            "cap_groupname": "spatial-encoding-rerun__inp=spatial_DC__net=brunel__std=20__noise_loop_duration=step_duration",
            "tasks": {
                "XOR": 'BRN-scan-normalXOR-test__net=brunel__inp=spatial_DC_XOR__std=20__loop=step_duration__g=5.0__J=0.2',
                "tXOR": 'BRN-scan-discrete-temporal-XOR-test__net=brunel__inp=spatial_DC_temporal_XOR__std=20__noise_loop_duration=step_duration__g=5.0__J=0.2',
                "XORXOR": 'BRN-scan-spatial-XORXOR-test__net=brunel__inp=spatial_DC_XORXOR__std=20__noise_loop_duration=step_duration__g=5.0__J=0.2',
                "class.": 'BRN-scan-spatial-classification-test__net=brunel__inp=spatial_DC_classification__std=20__noise_loop_duration=step_duration__g=5.0__J=0.2',
                "NARMA5": 'BRN-scan-NARMA5-test__net=brunel__inp=spatial_DC__std=20__loop=step_duration__g=5.0__J=0.2',
            },
            "figname": 'cap-task-correlations_no-ctXOR_spatial_std=20_with-encoder-cap.pdf'
        },
    }

    for cap_title, cap_dict in cap_to_tasks_dict.items():
        # for cap_title, cap_dict in parameters['cap_to_tasks_dict'].items():
        if os.path.exists(cap_dict['cap_groupname']) and len([x for x in os.listdir(cap_dict['cap_groupname']) if x.endswith(".pkl")]) > 0:
            print(f'The cap_groupname for {cap_title} is a full capacity folder. '
                  f'We use this instead of constructing a path.')
            cap_folder = cap_dict['cap_groupname']
        else:
            cap_folder = os.path.join(cap_base_folder, cap_dict['cap_groupname'], 'capacity')
        ax = axes[cap_dict['axes_letter']]
        plot_single_correlations_plot(ax, bar_width, cap_dict, cap_folder, cap_title, task_base_folder, use_spearmanr,
                                      use_cache, use_precalculated=use_precalculated)

    axes['tasks1'].set_ylabel('correlation', labelpad=-5)
    axes['tasks2'].set_yticklabels([])

    axes['tasks2'].legend(ncol=4, loc='upper center', bbox_to_anchor=(-0.25, -0.37), fontsize=10)


def plot_single_correlations_plot(ax, bar_width, cap_dict, cap_folder, cap_title, task_base_folder, use_spearmanr,
                                  use_cache, use_precalculated=True):
    ax.set_ylim((-1, 1))
    print(cap_title)
    colors = {
        'capacity': get_color('capacity'),
        'nonlin. cap. delay 5': get_color('accent', desaturated=True),
        'nonlinear capacity\ndelay 5': '#77CBB9',
        'nonlin. cap. delay 10': get_color('accent', desaturated=True),
        'nonlinear capacity\ndelay 10': '#2C0735',
        'degrees': get_color('degree'),  # 'wheat',
        'delays': get_color('delay'),  # 'plum',
    }

    capacity_info_types = ['degrees', 'capacity', 'delays', 'nonlin. cap. delay 5']
    if use_precalculated:
        if '(' in cap_title:
            network_type = 'distributed-spatial'
        else:
            network_type = cap_title
        with open(f'data/BRN_task_correlations_{network_type}.pkl', 'rb') as task_correlations_file:
            precalculated_data = pickle.load(task_correlations_file)
    for info_type_idx, cap_info_type in enumerate(capacity_info_types):
        print(f'\t{cap_info_type}')
        shift_factor = info_type_idx - 1

        if use_precalculated:
            tasknames = precalculated_data[cap_info_type]['tasknames']
            correlations = precalculated_data[cap_info_type]['correlations']
        else:
            get_max_degrees = (cap_info_type == 'degrees')
            get_max_delays = (cap_info_type == 'delays')

            if cap_info_type in ['nonlin. cap. delay 5', 'nonlinear capacity\ndelay 5']:
                cap_data_dict = get_heatmap_data(x_name='dur', y_name='max', capacity_folder=cap_folder,
                                                 params_to_filter={}, mindelay=5, maxdelay=5, mindegree=2)
            elif cap_info_type in ['nonlin. cap. delay 10', 'nonlinear capacity\ndelay 10']:
                cap_data_dict = get_heatmap_data(x_name='dur', y_name='max', capacity_folder=cap_folder,
                                                 params_to_filter={}, mindelay=10, maxdelay=10, mindegree=1)
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
                task_folder = os.path.join(task_base_folder, task_group)

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

                if cap_info_type in ['nonlin. cap. delay 5', 'nonlinear capacity\ndelay 5', 'nonlin. cap. delay 10',
                                     'nonlinear capacity\ndelay 10'] and task_name not in ['NARMA5', 'NARMA10']:
                    corr = 0.
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
        ax.tick_params(axis="x", direction="in", pad=-4, length=0.)

    ax.set_title(cap_title)
    ax.spines['top'].set_color('none')
    ax.spines['bottom'].set_color('none')
    ax.spines['right'].set_color('none')


def plot_removed_encoder_heatmaps(axes, use_cache, use_precalculated=True):

    cap_folder = '/home/schultetobrinke/projects/recurrence/repos/capacity_visualisation/capacity_visualisation/data/spatial-encoding-rerun__inp=spatial_DC__net=brunel__std=1__noise_loop_duration=step_duration/network_capacity_remove_full-transform'
    params_to_filter = {}
    if use_precalculated:
        precalculated_data_path = os.path.join('data', 'BRN_capacities_std1_noencoder.pkl')
    else:
        precalculated_data_path = None
    _, axes[0] = plot_heatmap('dur', 'max', capacity_folder=cap_folder, title=r'$\sigma 1$',
                                params_to_filter=params_to_filter, ax=axes[0],
                                cutoff=0., figure_path=None, plot_max_degrees=False, plot_max_delays=False,
                                plot_num_trials=False,
                                annotate=False, other_filter_keys=['network'], use_cache=use_cache, colorbar_label='',
                                precalculated_data_path=precalculated_data_path)
    axes[0].set_xticklabels([])
    axes[0].set_xticks([])
    axes[0].set_yticklabels([])
    axes[0].set_yticks([])
    axes[0].set_xlabel(None)
    axes[0].set_ylabel(None)

    cap_folder = '/home/schultetobrinke/projects/recurrence/repos/capacity_visualisation/capacity_visualisation/data/spatial-encoding-rerun__inp=spatial_DC__net=brunel__std=20__noise_loop_duration=step_duration/network_capacity_identity-subtractall-1-transform'
    params_to_filter = {}
    if use_precalculated:
        precalculated_data_path = os.path.join('data', 'BRN_capacities_std20_noencoder.pkl')
    else:
        precalculated_data_path = None
    _, axes[1] = plot_heatmap('dur', 'max', capacity_folder=cap_folder, title=r'$\sigma 20$',
                                   params_to_filter=params_to_filter, ax=axes[1],
                                   cutoff=0., figure_path=None, plot_max_degrees=False, plot_max_delays=False,
                                   plot_num_trials=False,
                                   annotate=False, other_filter_keys=['network'], use_cache=use_cache, colorbar_label='',
                                   precalculated_data_path=precalculated_data_path)
    axes[1].set_xticklabels([])
    axes[1].set_xticks([])
    axes[1].set_xlabel(None)
    axes[1].set_ylabel(None)
    axes[1].set_yticklabels([])
    axes[1].set_yticks([])


def setup_axes():
    fig = plt.figure(figsize=(9, 7))
    left_space = 8
    top_space = 5
    gs = fig.add_gridspec(100+top_space, 100+left_space)
    gs.update(left=0., right=1., top=1., bottom=0., wspace=0., hspace=0.)

    axes = {}

    heatmaps_start_x = 0
    heatmaps_start_y = 0
    heatmap_height = 17
    heatmap_width = 20
    heatmap_space_horizontal = 3
    heatmap_space_vertical = 5

    heatmap_positions = {
        '1': (0, 0),
        '2': (0, 1),
        'E': (1, 0),
        'F': (2, 0),
        'G': (3, 0),
        'J': (1, 1),
        'K': (2, 1),
        'L': (3, 1),
    }

    for subplot_label, grid_position in heatmap_positions.items():
        pos_x = grid_position[0]
        pos_y = grid_position[1]

        y_1 = top_space + heatmaps_start_y + pos_y * heatmap_height + pos_y * heatmap_space_vertical
        y_2 = y_1 + heatmap_height

        x_1 = left_space + heatmaps_start_x + pos_x * heatmap_width + pos_x * heatmap_space_horizontal
        if pos_x > 0:
            x_1 += 7
        x_2 = x_1 + heatmap_width

        if pos_x == 0:
            x_2 -= 5
            y_2 -=2

        axes[subplot_label] = fig.add_subplot(gs[y_1:y_2, x_1:x_2])
    
    lines_start_x = 0
    lines_start_y = 12 + heatmaps_start_y + 2*heatmap_height + heatmap_space_vertical
    lines_height = 12
    lines_width = 12
    lines_space_horizontal = 7
    lines_space_vertical = 5
    
    lines_positions = {
        '3': (0, 0),
        '4': (0, 1),
        '5': (1, 0),
        '6': (1, 1),
    }

    for subplot_label, grid_position in lines_positions.items():
        pos_x = grid_position[0]
        pos_y = grid_position[1]

        y_1 = top_space + lines_start_y + pos_y * lines_height + pos_y * lines_space_vertical
        y_2 = y_1 + lines_height

        x_1 = left_space + lines_start_x + pos_x * lines_width + pos_x * lines_space_horizontal
        x_2 = x_1 + lines_width

        axes[subplot_label] = fig.add_subplot(gs[y_1:y_2, x_1:x_2])

    bars_start_x = 12 + lines_start_x + lines_space_horizontal + 2 * lines_width
    bars_start_y = lines_start_y
    bars_widths = [33, 8]
    bars_height = 10
    bars_space_horizontal = 4
    bwidth_sum = 0
    for bar_nr, twidth in enumerate(bars_widths):
        y_1 = top_space + bars_start_y
        y_2 = y_1 + bars_height
        x_1 = left_space + bars_start_x + bwidth_sum + bar_nr * bars_space_horizontal
        if bar_nr > 0:
            x_1 += 4
        x_2 = x_1 + twidth
        axes[f'bar{bar_nr+1}'] = fig.add_subplot(gs[y_1:y_2, x_1:x_2])
        bwidth_sum += twidth
    
    tasks_start_x = 18 + lines_start_x + lines_space_horizontal + 2 * lines_width
    tasks_start_y = 11 + lines_start_y + bars_height
    tasks_width = 18
    tasks_height = 17
    tasks_space_horizontal = 6
    bwidth_sum = 0
    for bar_nr in range(2):
        y_1 = top_space + tasks_start_y
        y_2 = y_1 + tasks_height
        x_1 = left_space + tasks_start_x + bwidth_sum + bar_nr * tasks_space_horizontal
        x_2 = x_1 + tasks_width
        axes[f'tasks{bar_nr+1}'] = fig.add_subplot(gs[y_1:y_2, x_1:x_2])
        bwidth_sum += tasks_width

    return fig, axes


def main(use_cache=True, use_precalculated=True):

    fig1, axes1 = setup_axes()

    plot_heatmaps(axes1, use_cache, use_precalculated=use_precalculated)
    plot_bars(axes1)
    set_color_markers(axes1)
    plot_encoding_line_graphs(axes1, use_cache)
    plot_task_correlations(axes1, use_cache, use_precalculated=use_precalculated)
    fig1.savefig('figures/BRN_plots/BRN_results_figure.pdf')
    fig1.savefig('figures/BRN_plots/BRN_results_figure.svg')

    plt.close('all')
    fig2, axes2 = plt.subplots(nrows=2, ncols=1, figsize=(1.1, 2))
    plot_removed_encoder_heatmaps(axes2, use_cache, use_precalculated=use_precalculated)
    plt.tight_layout()
    fig2.savefig('figures/BRN_plots/BRN_removed_encoder_heatmaps.svg')
    fig2.savefig('figures/BRN_plots/BRN_removed_encoder_heatmaps.pdf')

    plt.close('all')
    fig3, ax = plt.subplots(figsize=(1.2, 1.5))
    plot_encoder_capacity_bars(ax)
    plt.tight_layout()
    fig3.savefig('figures/BRN_plots/BRN_encoder_capbars.svg')
    fig3.savefig('figures/BRN_plots/BRN_encoder_capbars.pdf')


def parse_cmd():
    parser = argparse.ArgumentParser()

    parser.add_argument('--disable_precalculated', action='store_true', help='Do not use precalculated data')
    parser.add_argument('--disable_cache', action='store_true', help='Do not use cached data')

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_cmd()
    use_cache = not args.disable_cache
    use_precalculated = not args.disable_precalculated
    main(use_cache=use_cache, use_precalculated=use_precalculated)
