import os
import yaml
import pickle
import argparse

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle, FancyBboxPatch, Circle
# from matplotlib.patches import ConnectionPatch, FancyBboxPatch, FancyArrowPatch, ConnectionStyle

from evaluation.calculate_task_correlations import get_correlation
from heatmaps import plot_heatmap, get_heatmap_data, get_task_results, plot_task_results_heatmap
from colors import get_color
import cap_bars_single_run


def setup_pyplot():
    # plt.rcParams['figure.constrained_layout.use'] = True
    # plt.rcParams['figure.constrained_layout.w_pad'] = 0.05
    # plt.rcParams['figure.constrained_layout.w_pad'] = 0.0
    # plt.rcParams['figure.constrained_layout.h_pad'] = 0.05
    # plt.rcParams['figure.constrained_layout.h_pad'] = 0.0
    SMALL_SIZE = 6
    MEDIUM_SIZE = 8
    BIGGER_SIZE = 10
    # SMALL_SIZE = 12
    # MEDIUM_SIZE = 18
    # BIGGER_SIZE = 18
    plt.rc('font', size=SMALL_SIZE)  # controls default text sizes
    plt.rc('axes', titlesize=BIGGER_SIZE)  # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)  # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
    # plt.rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
    # plt.rc('font', family='serif')
    matplotlib.rcParams['figure.dpi'] = 600

def data_path():
    return '/home/schultetobrinke/projects/recurrence/repos/capacity_visualisation/capacity_visualisation/data'


def get_capacity_directory(spatial_or_uniform, random_or_frozen, rate_or_DC, p=None, std=None, capacity_name='capacity'):
    assert spatial_or_uniform in ['step', 'spatial',
                                  'uniform'], 'step_spatial_or_uniform should be "step", "spatial" or "uniform"'
    assert random_or_frozen in ['randomnoise',
                                'frozennoise'], 'random_or_frozen should be "randomnoise" or "frozennoise"'
    assert rate_or_DC in ['rate', 'DC'], 'rate_or_DC should be "rate" or "DC"'
    assert (p is not None or std is not None), 'p or std should be set'

    noise_loop_dur = 'step_duration' if random_or_frozen == 'frozennoise' else 'None'
    if spatial_or_uniform == 'uniform':
        p_or_std_string = f'p={p}'
    else:
        p_or_std_string = f'std={std}'

    folder_name = f'iaf-diffVth-microcircuit_{spatial_or_uniform}-encoding_{random_or_frozen}__inp={spatial_or_uniform}_{rate_or_DC}__net=microcircuit__{p_or_std_string}__noise_loop_duration={noise_loop_dur}'
    # iaf-diffVth-microcircuit_spatial-encoding_frozennoise__inp=spatial_rate__net=microcircuit__std=15__noise_loop_duration=step_duration
    capacity_directory = os.path.join(data_path(), folder_name, capacity_name)

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


def add_capacity_percent_twinx(ax, N=447, add_label=False):
    axtwin = ax.twinx()
    ytwin_ticklabels = [int(100 * t / N) for t in ax.get_yticks()][1:]
    if ytwin_ticklabels[0] == 4 and ytwin_ticklabels[1] == 8:  # for spatial microcircuit plot (looks nicer that way)
        ytwin_ticklabels[2] = 12
    if ytwin_ticklabels[3] == 6 and ytwin_ticklabels[4] == 8 and ytwin_ticklabels[5] == 11:  # looks nicer for MC uniform
        ytwin_ticklabels[5] = 10
    ytwin_tickpositions = [N * t / 100. for t in ytwin_ticklabels]
    axtwin.set_yticks(ytwin_tickpositions, labels=ytwin_ticklabels)
    if add_label:
        axtwin.set_ylabel("% of max.", size=8)
    ylim = ax.get_ylim()
    axtwin.set_ylim(ylim)


def plot_max_cap_per_p_or_std(step_spatial_or_uniform, ax=None, plot_degrees=False, plot_memory=False, use_cache=False,
                              plot_stds=True, use_label=False, capacity_name='capacity'):
    if capacity_name == 'capacity':
        other_filter_keys = ['vm']
    else:
        other_filter_keys = ['network.pkl']

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
        p_or_std_values = [1, 2, 5, 10, 15, 20]
    else:
        p_or_std_values = [0.25, 0.5, 0.75, 1.0]
    for random_or_frozen in ['frozennoise', 'randomnoise']:
        # for rate_or_DC in ['DC', 'rate']:
        # noise_types = ['frozennoise']
        # if rate_or_DC == 'DC':
        #     noise_types.append('randomnoise')
        # for random_or_frozen in noise_types:
        # for random_or_frozen in ['frozennoise', 'randomnoise']:
        for rate_or_DC in ['DC', 'rate']:
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
                cap_dir = get_capacity_directory(spatial_or_uniform=step_spatial_or_uniform, rate_or_DC=rate_or_DC,
                                                 random_or_frozen=random_or_frozen, p=p, std=std, capacity_name=capacity_name)
                params_to_filter = {}
                try:
                    print(f'\t{p_or_std}, degrees: {plot_degrees}, delays: {plot_memory}')
                    avg_cap_results = get_heatmap_data('dur', 'max', cap_dir, params_to_filter=params_to_filter,
                                                       get_max_degrees=plot_degrees, get_max_delays=plot_memory,
                                                       other_filter_keys=other_filter_keys, use_cache=use_cache)
                    std_cap_results = get_heatmap_data('dur', 'max', cap_dir, params_to_filter=params_to_filter,
                                                       get_max_degrees=plot_degrees, get_max_delays=plot_memory,
                                                       other_filter_keys=other_filter_keys, use_cache=use_cache,
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
                        f"Capacity data for {step_spatial_or_uniform}_{rate_or_DC}_p={p_or_std} ({random_or_frozen}, {capacity_name}) does not exist "
                        f"and will not be added to the data!")
                except BaseException as err:
                    print(f"Unexpected {type(err)}: {err}")

            if len(p_or_std_values_with_data) == 0:
                print(
                    f"No data for {step_spatial_or_uniform}_{rate_or_DC} ({random_or_frozen}, {capacity_name}) does exist for any p ({p_or_std_values})."
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


def colorize_spines(ax, color):
    ax.spines['bottom'].set_color(color)
    ax.spines['bottom'].set_linewidth(3)
    ax.spines['top'].set_color(color)
    ax.spines['top'].set_linewidth(3)
    ax.spines['right'].set_color(color)
    ax.spines['right'].set_linewidth(3)
    ax.spines['left'].set_color(color)
    ax.spines['left'].set_linewidth(3)


def setup_axes():
    fig = plt.figure(figsize=(8, 7))
    left_space = 10
    top_space = 5
    gs = fig.add_gridspec(100+top_space, 100+left_space)
    gs.update(left=0., right=1., top=1., bottom=0., wspace=0., hspace=0.)
    height_buffer_large = 7
    height_buffer_smaller = 5
    width_buffer_large = 10
    width_buffer_smaller = 7
    height_large = 15
    height_smaller = 5
    width_large = 18
    width_smaller = 13
    extra_bottom_row_space = 6

    main_x_positions = [left_space, left_space + 33, left_space + 55, left_space + 80]
    bottom_row_x_positions = [left_space, left_space + 18, left_space + 44, left_space + 63, left_space + 82]

    main_grid_positions = {
        '1': (0, 0),
        '2': (0, 1),
        '3': (0, 2),
        'A': (1, 0),
        'C': (1, 1),
        'E': (1, 2),
        'B': (2, 0),
        'D': (2, 1),
        'F': (2, 2),
    }

    axes = {}
    for subplot_label, grid_position in main_grid_positions.items():
        pos_x = grid_position[0]
        pos_y = grid_position[1]
        y_1 = top_space + pos_y * height_buffer_large + pos_y * height_large
        # y_1 = main_y_positions[pos_y]
        y_2 = y_1 + height_large
        # x_1 = left_space + pos_x * width_buffer_large + pos_x * width_large
        # if subplot_label in ['A', 'C', 'E']:
        #     x_1 += int(0.25 * width_buffer_large)
        # elif subplot_label in ['B', 'D', 'F']:
        #     x_1 -= int(0.25 * width_buffer_large)
        x_1 = main_x_positions[pos_x]
        x_2 = x_1 + width_large
        axes[subplot_label] = fig.add_subplot(gs[y_1:y_2, x_1:x_2])

    bar_grid_positions = {
        'G': (3, 0),
        'H': (3, 1),
        'I': (3, 2),
        'J': (3, 3),
        'K': (3, 4),
        'L': (3, 5),
    }

    for subplot_label, grid_position in bar_grid_positions.items():
        pos_x = grid_position[0]
        pos_y = grid_position[1]

        if pos_y == 0:
            y_1 = top_space + pos_y
        elif pos_y % 2 == 0:
            y_1 = top_space + (pos_y/2) * height_buffer_large + pos_y/2 * height_large
        else:
            y_1 = top_space + (pos_y-1)/2 * height_buffer_large + (pos_y-1)/2 * height_large + height_buffer_smaller + height_smaller

        y_1 = int(y_1)
        y_2 = y_1 + height_smaller
        # x_1 = left_space + pos_x * width_buffer_large + pos_x * width_large
        x_1 = main_x_positions[pos_x]
        x_2 = x_1 + width_large
        # print(f'{y_1}, {y_2}, {x_1}, {x_2}')
        axes[subplot_label] = fig.add_subplot(gs[y_1:y_2, x_1:x_2])

    bottom_row_positions = {
        '4': (0, 3),
        '5': (1, 3),
        '6': (2, 3),
        '7': (3, 3),
        '8': (4, 3),
    }

    for subplot_label, grid_position in bottom_row_positions.items():
        pos_x = grid_position[0]
        pos_y = grid_position[1]

        y_1 = top_space + pos_y * height_buffer_large + pos_y * height_large + extra_bottom_row_space
        y_2 = y_1 + height_large
        # x_1 = left_space + pos_x * width_buffer_smaller + pos_x * width_smaller
        x_1 = bottom_row_x_positions[pos_x]
        x_2 = x_1 + width_smaller
        axes[subplot_label] = fig.add_subplot(gs[y_1:y_2, x_1:x_2])

    return fig, axes


def parse_cmd():
    parser = argparse.ArgumentParser()

    parser.add_argument('--disable_cache', action='store_true')

    return parser.parse_args()


def main():
    args = parse_cmd()
    use_cache = not args.disable_cache

    setup_pyplot()

    fig, axes = setup_axes()
    cap_base_folder = '/home/schultetobrinke/nextcloud/Juelich/projects/recurrence/repos/capacity_visualisation/capacity_visualisation/data'

    plot_heatmaps_and_bars(axes, cap_base_folder, use_cache)

    plot_encoding_line_graphs(axes, use_cache)

    plot_correlations(axes, use_cache)

    draw_lines(fig, axes)

    fontsize = 14
    fig.text(0.01, 0.83, 'uniform', size=fontsize, rotation=90)
    fig.text(0.01, 0.63, 'spatial', size=fontsize, rotation=90)
    fig.text(0.01, 0.38, 'no encoder', size=fontsize, rotation=90)


    # plt.show()
    # plt.savefig('figures/microcircuit_plots/mc_result_heatmaps.pdf')
    # plt.savefig('/home/schultetobrinke/Downloads/deleteme/mc_results_figure.pdf')
    plt.savefig('/home/schultetobrinke/projects/SNN_capacity/repos/SNN_capacity/plotting/figures/microcircuit_plots/mc_results_figure.svg')


def draw_lines(fig, axes):

    # circle_uniform = Circle((1., 49.493587700551565), 5, color='black', fill=False)
    # axes['1'].add_patch(circle_uniform)

    linewidth=1.
    box_uniform = FancyBboxPatch(xy=(0.34, 0.78), width=0.64, height=0.175, boxstyle=mpatches.BoxStyle("Round", pad=0.01),
                                 transform=fig.transFigure, edgecolor='black', facecolor='white', zorder=-10, lw=linewidth)
    box_spatial = FancyBboxPatch(xy=(0.34, 0.57), width=0.64, height=0.18, boxstyle=mpatches.BoxStyle("Round", pad=0.01),
                                 transform=fig.transFigure, edgecolor='black', facecolor='white', zorder=-10, lw=linewidth)
    box_noenc = FancyBboxPatch(xy=(0.34, 0.335), width=0.64, height=0.205, boxstyle=mpatches.BoxStyle("Round", pad=0.01),
                               transform=fig.transFigure, edgecolor='black', facecolor='white', zorder=-10, lw=linewidth)
    fig.patches.extend([box_uniform, box_spatial, box_noenc])


def plot_encoding_line_graphs(axes, use_cache):
    encoder_cap_name = "network_capacity_identity-subtractall-1-transform"
    # encoder_cap_name = "network_capacity_sqrt-subtractall-1-transform"
    axes['1'] = plot_max_cap_per_p_or_std('uniform', ax=axes['1'], plot_memory=False, use_cache=use_cache,
                                          plot_stds=True)
    axes['1'].scatter([1], [49.493587700551565], marker='o', facecolor='none', s=75, linewidth=1, edgecolor='black', zorder=10)
    add_capacity_percent_twinx(ax=axes['1'], add_label=True)
    # axes['1'].set_title('uniform')
    axes['1'].set_title('Capacity per Scan', y=1.05)
    axes['1'].set_ylabel('capacity')
    # axes['1'].set_ylabel(None)
    # axes['1'].set_xlabel(None)
    # axes['1'].set_xticklabels([])

    axes['2'] = plot_max_cap_per_p_or_std('spatial', axes['2'], use_cache=use_cache, plot_stds=True)
    axes['2'].scatter([20], [40.11322767995303], marker='o', facecolor='none', s=75, linewidth=1, edgecolor='black', zorder=10)
    # axes['2'].legend(prop={'size': 5})
    add_capacity_percent_twinx(ax=axes['2'], add_label=True)
    # axes['2'].set_title('spatial')
    axes['2'].set_ylabel('capacity')
    # axes['2'].set_ylabel(None)
    # axes['2'].set_xlabel(None)
    # axes['2'].set_xticklabels([])

    axes['3'] = plot_max_cap_per_p_or_std('spatial', axes['3'], use_cache=use_cache, plot_stds=True,
                                          capacity_name=encoder_cap_name)
    axes['3'].scatter([20], [17.866824369470578], marker='o', facecolor='none', s=75, linewidth=1, edgecolor='black', zorder=10)
    add_capacity_percent_twinx(ax=axes['3'], add_label=True)
    # axes['3'].set_title('spatial\n(removed enc.)')
    axes['3'].set_ylabel('capacity')
    # axes['3'].set_ylabel(None)
    # axes['3'].set_xlabel(None)
    # axes['3'].set_xticklabels([])
    # memory
    axes['4'] = plot_max_cap_per_p_or_std('uniform', ax=axes['4'], plot_memory=True, use_cache=use_cache,
                                          plot_stds=True, use_label=True)
    axes['4'].set_ylabel('memory [ms]')
    # axes['4'].set_ylabel(None)
    axes['4'].set_title('uniform')
    axes['4'].legend(ncol=2, loc='upper left', bbox_to_anchor=(-0.2, -0.4), fontsize=7)

    axes['5'] = plot_max_cap_per_p_or_std('spatial', axes['5'], plot_memory=True, use_cache=use_cache,
                                          plot_stds=True)
    axes['5'].set_ylabel(None)
    axes['5'].set_title('spatial')


def plot_heatmaps_and_bars(axes, cap_base_folder, use_cache):
    # uniform_color1 = get_color('accent')
    uniform_color1 = get_color('degree')
    # uniform_color2 = get_color('XOR')
    uniform_color2 = get_color('accent', desaturated=True)
    cap_uniform_name = 'iaf-diffVth-microcircuit_uniform-encoding_frozennoise__inp=uniform_DC__net=microcircuit__p=1.0__noise_loop_duration=step_duration'
    cap_uniform_folder = os.path.join(cap_base_folder, cap_uniform_name, 'capacity')
    params_to_filter = {}
    _, axes['A'] = plot_heatmap('dur', 'max', capacity_folder=cap_uniform_folder, title=r'',
                                params_to_filter=params_to_filter, ax=axes['A'], cutoff=0., figure_path=None,
                                plot_max_degrees=False, plot_max_delays=False, plot_num_trials=False,
                                annotate=False, other_filter_keys=['vm'], use_cache=use_cache,
                                max_marker_color=uniform_color1, colorbar_label='')
    axes['A'].add_patch(Rectangle((1, 1), 1, 1, fill=False, edgecolor=uniform_color2, lw=2, clip_on=False))
    axes['A'].set_yticklabels([int(float(n.get_text())) for n in axes['A'].get_yticklabels()])
    # axes['A'].set_yticklabels([])
    axes['A'].set_xticklabels([])
    axes['A'].set_xlabel(None)
    axes['A'].set_title('Total Capacity', y=1.05)

    _, axes['B'] = plot_heatmap('dur', 'max', capacity_folder=cap_uniform_folder, title=r'',
                                params_to_filter=params_to_filter, ax=axes['B'], cutoff=0., figure_path=None,
                                plot_max_degrees=False, plot_max_delays=True, plot_num_trials=False,
                                annotate=False, other_filter_keys=['vm'], use_cache=use_cache,
                                max_marker_color=uniform_color2, colorbar_label='')
    axes['B'].add_patch(Rectangle((7, 7), 1, 1, fill=False, edgecolor=uniform_color1, lw=2, clip_on=False))
    axes['B'].set_yticklabels([])
    axes['B'].set_ylabel(None)
    axes['B'].set_xticklabels([])
    axes['B'].set_xlabel(None)
    axes['B'].set_title('Max Delay', y=1.05)
    # spatial_color1 = get_color('accent')
    spatial_color1 = get_color('degree')
    # spatial_color2 = get_color('XOR')
    spatial_color2 = get_color('accent', desaturated=True)
    cap_spatial_name = 'iaf-diffVth-microcircuit_spatial-encoding_frozennoise__inp=spatial_DC__net=microcircuit__std=20__noise_loop_duration=step_duration'
    cap_spatial_folder = os.path.join(cap_base_folder, cap_spatial_name, 'capacity')
    params_to_filter = {}
    _, axes['C'] = plot_heatmap('dur', 'max', capacity_folder=cap_spatial_folder, title=r'',
                                params_to_filter=params_to_filter, ax=axes['C'], cutoff=0., figure_path=None,
                                plot_max_degrees=False, plot_max_delays=False, plot_num_trials=False,
                                annotate=False, other_filter_keys=['vm'], use_cache=use_cache,
                                max_marker_color=spatial_color1, colorbar_label='')
    axes['C'].add_patch(Rectangle((1, 4), 1, 1, fill=False, edgecolor=spatial_color2, lw=2, clip_on=False))
    axes['C'].set_yticklabels([int(float(n.get_text())) for n in axes['C'].get_yticklabels()])
    # axes['C'].set_yticklabels([])
    axes['C'].set_xticklabels([])
    axes['C'].set_xlabel(None)
    _, axes['D'] = plot_heatmap('dur', 'max', capacity_folder=cap_spatial_folder, title=r'',
                                params_to_filter=params_to_filter, ax=axes['D'], cutoff=0., figure_path=None,
                                plot_max_degrees=False, plot_max_delays=True, plot_num_trials=False,
                                annotate=False, other_filter_keys=['vm'], use_cache=use_cache,
                                max_marker_color=None, colorbar_label='')
    axes['D'].add_patch(Rectangle((7, 14), 1, 1, fill=False, edgecolor=spatial_color1, lw=2, clip_on=False))
    axes['D'].add_patch(Rectangle((1, 4), 1, 1, fill=False, edgecolor=spatial_color2, lw=2, clip_on=False))
    axes['D'].set_yticklabels([])
    axes['D'].set_ylabel(None)
    axes['D'].set_xticklabels([])
    axes['D'].set_xlabel(None)
    # noenc_color1 = get_color('accent')
    noenc_color1 = get_color('degree')
    # noenc_color2 = get_color('XOR')
    noenc_color2 = get_color('accent', desaturated=True)
    cap_noenc_name = 'iaf-diffVth-microcircuit_spatial-encoding_frozennoise__inp=spatial_DC__net=microcircuit__std=20__noise_loop_duration=step_duration'
    cap_noenc_folder = os.path.join(cap_base_folder, cap_noenc_name,
                                    'network_capacity_identity-subtractall-1-transform')
    params_to_filter = {}
    _, axes['E'] = plot_heatmap('dur', 'max', capacity_folder=cap_noenc_folder, title=r'',
                                params_to_filter=params_to_filter, ax=axes['E'], cutoff=0., figure_path=None,
                                plot_max_degrees=False, plot_max_delays=False, plot_num_trials=False,
                                annotate=False, other_filter_keys=None, use_cache=use_cache,
                                max_marker_color=noenc_color1, colorbar_label='')
    axes['E'].add_patch(Rectangle((1, 4), 1, 1, fill=False, edgecolor=noenc_color2, lw=2, clip_on=False))
    axes['E'].set_yticklabels([int(float(n.get_text())) for n in axes['E'].get_yticklabels()])
    # axes['E'].set_yticklabels([])
    axes['E'].set_xticklabels([int(float(n.get_text())) for n in axes['E'].get_xticklabels()])
    _, axes['F'] = plot_heatmap('dur', 'max', capacity_folder=cap_noenc_folder, title=r'',
                                params_to_filter=params_to_filter, ax=axes['F'], cutoff=0., figure_path=None,
                                plot_max_degrees=False, plot_max_delays=True, plot_num_trials=False,
                                annotate=False, other_filter_keys=None, use_cache=use_cache,
                                max_marker_color=noenc_color2, colorbar_label='', cbar_ticks=[0, 5, 10])
    axes['F'].add_patch(Rectangle((7, 14), 1, 1, fill=False, edgecolor=noenc_color1, lw=2, clip_on=False))
    axes['F'].set_yticklabels([])
    axes['F'].set_ylabel(None)
    axes['F'].set_xticklabels([int(float(n.get_text())) for n in axes['F'].get_xticklabels()])
    cap_uniform_dict_path1 = os.path.join(cap_uniform_folder,
                                          'capacities_iaf-diffVth-microcircuit_uniform-encoding_frozennoise__inp=uniform_DC__net=microcircuit__p=1.0__noise_loop_duration=step_duration__dur=50.0__max=1600.0__2_vm.pkl')
    with open(cap_uniform_dict_path1, 'rb') as cap_file_uniform1:
        cap_dict_uniform1 = pickle.load(cap_file_uniform1)
    cap_bars_single_run.plot_capacity_bars(cap_dict_uniform1, ax=axes['G'])
    axes['G'].set_xlabel(None)
    colorize_spines(axes['G'], uniform_color1)
    axes['G'].set_title('Capacity per Delay', y=1.2)
    axes['G'].set_ylabel(None)
    cap_uniform_dict_path2 = os.path.join(cap_uniform_folder,
                                          'capacities_iaf-diffVth-microcircuit_uniform-encoding_frozennoise__inp=uniform_DC__net=microcircuit__p=1.0__noise_loop_duration=step_duration__dur=2.0__max=400.0__1_vm.pkl')
    with open(cap_uniform_dict_path2, 'rb') as cap_file_uniform2:
        cap_dict_uniform2 = pickle.load(cap_file_uniform2)
    cap_bars_single_run.plot_capacity_bars(cap_dict_uniform2, ax=axes['H'])
    axes['H'].set_xlabel(None)
    colorize_spines(axes['H'], uniform_color2)
    axes['H'].set_ylabel(None)
    cap_spatial_dict_path1 = os.path.join(cap_spatial_folder,
                                          'capacities_iaf-diffVth-microcircuit_spatial-encoding_frozennoise__inp=spatial_DC__net=microcircuit__std=20__noise_loop_duration=step_duration__dur=50.0__max=3000.0__2_vm.pkl')
    with open(cap_spatial_dict_path1, 'rb') as cap_file_spatial1:
        cap_dict_spatial1 = pickle.load(cap_file_spatial1)
    cap_bars_single_run.plot_capacity_bars(cap_dict_spatial1, ax=axes['I'])
    axes['I'].set_xlabel(None)
    colorize_spines(axes['I'], spatial_color1)
    axes['I'].set_ylabel(None)
    cap_spatial_dict_path2 = os.path.join(cap_spatial_folder,
                                          'capacities_iaf-diffVth-microcircuit_spatial-encoding_frozennoise__inp=spatial_DC__net=microcircuit__std=20__noise_loop_duration=step_duration__dur=2.0__max=1000.0__2_vm.pkl')
    with open(cap_spatial_dict_path2, 'rb') as cap_file_spatial2:
        cap_dict_spatial2 = pickle.load(cap_file_spatial2)
    cap_bars_single_run.plot_capacity_bars(cap_dict_spatial2, ax=axes['J'])
    axes['J'].set_xlabel(None)
    colorize_spines(axes['J'], spatial_color2)
    axes['J'].set_ylabel(None)
    cap_noenc_dict_path1 = os.path.join(cap_noenc_folder,
                                        'capacities_iaf-diffVth-microcircuit_spatial-encoding_frozennoise__inp=spatial_DC__net=microcircuit__std=20__noise_loop_duration=step_duration__dur=50.0__max=3000.0__2_network.pkl')
    with open(cap_noenc_dict_path1, 'rb') as cap_file_noenc1:
        cap_dict_noenc1 = pickle.load(cap_file_noenc1)
    cap_bars_single_run.plot_capacity_bars(cap_dict_noenc1, ax=axes['K'])
    axes['K'].set_xlabel(None)
    colorize_spines(axes['K'], noenc_color1)
    axes['K'].set_ylabel(None)
    cap_noenc_dict_path2 = os.path.join(cap_noenc_folder,
                                        'capacities_iaf-diffVth-microcircuit_spatial-encoding_frozennoise__inp=spatial_DC__net=microcircuit__std=20__noise_loop_duration=step_duration__dur=2.0__max=1000.0__2_network.pkl')
    with open(cap_noenc_dict_path2, 'rb') as cap_file_noenc2:
        cap_dict_noenc2 = pickle.load(cap_file_noenc2)
    cap_bars_single_run.plot_capacity_bars(cap_dict_noenc2, ax=axes['L'])
    # axes['L'].set_xlabel(None)
    colorize_spines(axes['L'], noenc_color2)
    axes['L'].set_ylabel(None)


def plot_correlations(axes, use_cache):

    with open('correlation_plot_parameters_MC.yaml', 'r') as parameters_file:
        parameters = yaml.safe_load(parameters_file)

    task_base_folder = '/home/schultetobrinke/projects/SNN_capacity/repos/ESN/scripts/MC_task_second_try'
    cap_base_folder = '/home/schultetobrinke/nextcloud/Juelich/projects/recurrence/repos/capacity_visualisation/capacity_visualisation/data'
    use_spearmanr = False
    bar_width = 0.25

    cap_to_tasks_dict = {
        "uniform": {
            "axes_letter": '6',
            "cap_groupname": "iaf-diffVth-microcircuit_uniform-encoding_frozennoise__inp=uniform_DC__net=microcircuit__p=1.0__noise_loop_duration=step_duration",
            "tasks": {
                "XOR": 'MCiaf-diffVth-scan-uniform-XOR-test__inp=uniform_DC_XOR__net=microcircuit__p=1.0__noise_loop_duration=step_duration',
                "tXOR": 'MCiaf-diffVth-scan-uniform-XOR-test__inp=uniform_DC_temporal_XOR__net=microcircuit__p=1.0__noise_loop_duration=step_duration',
                "XORXOR": 'MCiaf-diffVth-scan-uniform-XOR-test__inp=uniform_DC_XORXOR__net=microcircuit__p=1.0__noise_loop_duration=step_duration',
                "class.": 'MCiaf-diffVth-scan-uniform-classification-test__inp=uniform_DC_classification__net=microcircuit__p=1.0__noise_loop_duration=step_duration',
                "NARMA10": 'NARMA10_MCiaf-diffVth-scan-uniform-continuous-temporal-XOR-test__inp=uniform_DC__net=microcircuit__p=1.0__noise_loop_duration=step_duration',
            },
            "figname": 'MC_cap-task-correlations_uniform_p=1.0..pdf'
        },
        "spatial": {
            "axes_letter": '7',
            "cap_groupname": "iaf-diffVth-microcircuit_spatial-encoding_frozennoise__inp=spatial_DC__net=microcircuit__std=20__noise_loop_duration=step_duration",
            "tasks": {
                "XOR": 'MCiaf-diffVth-scan-spatial_XOR-test__inp=spatial_DC_XOR__net=microcircuit__std=20__noise_loop_duration=step_duration',
                "tXOR": 'MCiaf-diffVth-scan-spatial_discrete-temporal-XOR-test__inp=spatial_DC_temporal_XOR__net=microcircuit__std=20__noise_loop_duration=step_duration',
                "XORXOR": 'MCiaf-diffVth-scan-spatial_XORXOR-test__inp=spatial_DC_XORXOR__net=microcircuit__std=20__noise_loop_duration=step_duration',
                "class.": 'MCiaf-diffVth-scan-spatial_classification-test__inp=spatial_DC_classification__net=microcircuit__std=20__noise_loop_duration=step_duration',
                "NARMA10": 'NARMA10_MCiaf-diffVth-scan-spatial_continuous-temporal-XOR-test__inp=spatial_DC__net=microcircuit__std=20__noise_loop_duration=step_duration',
            },
            "figname": 'MC_cap-task-correlations_spatial_std=20.pdf'
        },
        "uniform\n(spatial tasks)": {
            "axes_letter": '8',
            "cap_groupname": "iaf-diffVth-microcircuit_uniform-encoding_frozennoise__inp=uniform_DC__net=microcircuit__p=1.0__noise_loop_duration=step_duration",
            "tasks": {
                "XOR": 'MCiaf-diffVth-scan-spatial_XOR-test__inp=spatial_DC_XOR__net=microcircuit__std=20__noise_loop_duration=step_duration',
                "tXOR": 'MCiaf-diffVth-scan-spatial_discrete-temporal-XOR-test__inp=spatial_DC_temporal_XOR__net=microcircuit__std=20__noise_loop_duration=step_duration',
                "XORXOR": 'MCiaf-diffVth-scan-spatial_XORXOR-test__inp=spatial_DC_XORXOR__net=microcircuit__std=20__noise_loop_duration=step_duration',
                "class.": 'MCiaf-diffVth-scan-spatial_classification-test__inp=spatial_DC_classification__net=microcircuit__std=20__noise_loop_duration=step_duration',
                "NARMA10": 'NARMA10_MCiaf-diffVth-scan-spatial_continuous-temporal-XOR-test__inp=spatial_DC__net=microcircuit__std=20__noise_loop_duration=step_duration',
            },
            "figname": 'MC_cap-task-correlations_uniform_p=1.0..pdf'
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
        plot_single_correlations_plot(ax, bar_width, cap_dict, cap_folder, cap_title, task_base_folder, use_spearmanr, use_cache)

    axes['6'].set_ylabel('correlation')
    axes['7'].set_yticklabels([])
    axes['8'].set_yticklabels([])

    axes['7'].legend(ncol=4, loc='upper center', bbox_to_anchor=(0.5, -0.35), fontsize=7)


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
        'nonlin. cap. delay 5': '#77CBB9',
        'nonlinear capacity\ndelay 5': '#77CBB9',
        # 'nonlin. cap. delay 10': '#2C0735',
        'nonlin. cap. delay 10': get_color('accent', desaturated=True),
        'nonlinear capacity\ndelay 10': '#2C0735',
        'degrees': get_color('degree'),  # 'wheat',
        'delays': get_color('delay'),  # 'plum',
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
        ax.set_xticklabels(tasknames, rotation=90)
        ax.tick_params(axis="x", direction="in", pad=-12, length=0.)
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


if __name__ == "__main__":
    main()
