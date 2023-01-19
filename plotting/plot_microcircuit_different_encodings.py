import os
import argparse

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from heatmaps import get_heatmap_data


def data_path():
    return '/home/schultetobrinke/projects/recurrence/repos/capacity_visualisation/capacity_visualisation/data'


def get_capacity_directory(spatial_or_uniform, random_or_frozen, rate_or_DC, p=None, std=None, capacity_name='capacity'):
    assert spatial_or_uniform in ['step', 'spatial',
                                       'uniform'], 'step_spatial_or_uniform should be "step", "spatial" or "uniform"'
    assert random_or_frozen in ['randomnoise',
                                'frozennoise'], 'random_or_frozen should be "randomnoise" or "frozennoise"'
    assert rate_or_DC in ['rate', 'DC'], 'rate_or_DC should be "rate" or "DC"'
    assert (p is not None or std is not None), 'p or std should be set'

    # if step_spatial_or_uniform == 'step':
    #     if random_or_frozen == 'frozennoise':
    #         folder_name = f'frozennoise_dur1-50_max0.2-3.0__net=brunel__p={p}__inp=step_{rate_or_DC}__steps=200000'
    #     else:
    #         if rate_or_DC == 'rate':
    #             folder_name = f'randomnoise_rate_dur1-50_max200-3000__net=brunel__inp=step_rate__p={p}__steps=200000'
    #         else:
    #             folder_name = f'randomnoise_dur1-50_max0.2-3.0__net=brunel__inp=step_DC__p={p}__steps=200000'
    # elif step_spatial_or_uniform == 'uniform':
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
    ytwin_tickpositions = [N * t / 100. for t in ytwin_ticklabels]
    axtwin.set_yticks(ytwin_tickpositions, labels=ytwin_ticklabels)
    if add_label:
        axtwin.set_ylabel("\% of max. capacity", size=8)
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


def setup_pyplot():
    # plt.rcParams['figure.constrained_layout.use'] = True
    # plt.rcParams['figure.constrained_layout.w_pad'] = 0.05
    # plt.rcParams['figure.constrained_layout.w_pad'] = 0.0
    # plt.rcParams['figure.constrained_layout.h_pad'] = 0.05
    # plt.rcParams['figure.constrained_layout.h_pad'] = 0.0
    SMALL_SIZE = 8
    MEDIUM_SIZE = 10
    BIGGER_SIZE = 12
    plt.rc('font', size=SMALL_SIZE)  # controls default text sizes
    plt.rc('axes', titlesize=BIGGER_SIZE)  # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)  # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
    plt.rc('font', family='serif')
    matplotlib.rcParams['figure.dpi'] = 600
    matplotlib.rcParams["text.usetex"] = True


def parse_cmd():
    parser = argparse.ArgumentParser()

    parser.add_argument('--figure_path', required=True, help='Path at which the figure will be stored.')
    parser.add_argument('--disable_cache', action='store_true', help='Disables caching')
    parser.add_argument('--plot_stds', action='store_true', help='Activate the plotting of the standard deviations')

    return parser.parse_args()


def main():
    args = parse_cmd()
    os.makedirs(os.path.dirname(args.figure_path), exist_ok=True)
    use_cache = not args.disable_cache
    encoder_cap_name = "network_capacity_identity-subtractall-1-transform"
    # encoder_cap_name = "network_capacity_sqrt-subtractall-1-transform"

    plot_stds = args.plot_stds
    if args.figure_path.endswith('.eps') and plot_stds:
        plot_stds = False
        print(
            "\n\n !!!!!!!!! standard deviations are not plotted because .eps files can't handle transparency! !!!!!!!\n\n")

    sns.set_style('dark', {'xtick.bottom': True, 'ytick.left': True})
    setup_pyplot()

    fig = plt.figure(constrained_layout=True, figsize=(5.2, 2.6))
    # fig.set_constrained_layout_pads(w_pad=0.05, h_pad=0.05)
    # fig = plt.figure(figsize=(4.5, 2.8))
    fig.set_constrained_layout_pads(w_pad=0.05, h_pad=0.05)
    constrained_layout_engine = fig.get_layout_engine()
    left = 0
    bottom = 0.075
    right_left = 1
    top_bottom = 1 - bottom
    constrained_layout_engine.set(rect=(left, bottom, right_left, top_bottom))
    constrained_layout_engine.execute(fig)
    # axes = fig.subplot_mosaic(
    #     """
    #     AEC
    #     BFD
    #     """
    #     , gridspec_kw={'left': 0.1, 'right': 0.9, 'bottom': 0.22, 'top': 0.9, 'wspace': 0.5, 'hspace': 0.2})
    # axes = fig.subplot_mosaic(
    #     """
    #     EC
    #     FD
    #     """
    # , gridspec_kw={'left': 0.1, 'right': 0.9, 'bottom': 0.22, 'top': 0.9, 'wspace': 0.5, 'hspace': 0.2})
    axes = fig.subplot_mosaic(
        """
        ECG
        FDH
        """
    )
        # , gridspec_kw={'left': 0.1, 'right': 0.9, 'bottom': 0.22, 'top': 0.9, 'wspace': 0.1, 'hspace': 0.1})

    # axes['A'] = plot_max_cap_per_p_or_std('step', axes['A'], use_cache=use_cache, plot_stds=plot_stds)
    # add_capacity_percent_twinx(ax=axes['A'])
    # axes['A'].set_title('amplitude')
    # axes['A'].set_xlabel(None)
    # axes['A'].set_xticklabels([])
    print('C')
    axes['C'] = plot_max_cap_per_p_or_std('spatial', axes['C'], use_cache=use_cache, plot_stds=plot_stds)
    # axes['C'].legend(prop={'size': 5})
    add_capacity_percent_twinx(ax=axes['C'], add_label=False)
    axes['C'].set_title('spatial')
    axes['C'].set_ylabel(None)
    axes['C'].set_xlabel(None)
    axes['C'].set_xticklabels([])
    print('D')
    # axes['B'] = plot_max_cap_per_p_or_std('step', axes['B'], plot_memory=True, use_cache=use_cache, plot_stds=plot_stds)
    axes['D'] = plot_max_cap_per_p_or_std('spatial', axes['D'], plot_memory=True, use_cache=use_cache,
                                          plot_stds=plot_stds)
    axes['D'].set_ylabel(None)

    print('G')
    axes['G'] = plot_max_cap_per_p_or_std('spatial', axes['G'], use_cache=use_cache, plot_stds=plot_stds,
                                          capacity_name=encoder_cap_name)
    add_capacity_percent_twinx(ax=axes['G'], add_label=True)
    axes['G'].set_title('spatial\n(removed enc.)')
    axes['G'].set_ylabel(None)
    axes['G'].set_xlabel(None)
    axes['G'].set_xticklabels([])
    print('H')
    axes['H'] = plot_max_cap_per_p_or_std('spatial', axes['H'], plot_memory=True, use_cache=use_cache,
                                          plot_stds=plot_stds, use_label=True, capacity_name=encoder_cap_name)
    axes['H'].set_ylabel(None)

    print('E')
    axes['E'] = plot_max_cap_per_p_or_std('uniform', ax=axes['E'], plot_memory=False, use_cache=use_cache,
                                          plot_stds=plot_stds)
    add_capacity_percent_twinx(ax=axes['E'])
    axes['E'].set_title('uniform')
    axes['E'].set_ylabel('capacity')
    axes['E'].set_xlabel(None)
    axes['E'].set_xticklabels([])
    print('F')
    axes['F'] = plot_max_cap_per_p_or_std('uniform', ax=axes['F'], plot_memory=True, use_cache=use_cache,
                                          plot_stds=plot_stds, use_label=False)
    axes['F'].set_ylabel('memory [ms]')
    # axes['F'].legend(prop={'size': 5}, ncol=4, loc='upper center', fancybox=True, bbox_to_anchor=(0.5, -0.05))
    fig.legend(prop={'size': 6}, ncol=4, fancybox=True, loc='lower center', bbox_to_anchor=(0.5, 0.))  # , loc='upper center', bbox_to_anchor=(0.5, -0.05))

    fontsize = 8
    x1 = 0.04
    x2 = 0.51
    y1 = 0.95
    y2 = 0.53
    # fig.text(x1, y1, "A", size=fontsize)
    # fig.text(x2, y1, "B", size=fontsize)
    # fig.text(x1, y2, "C", size=fontsize)
    # fig.text(x2, y2, "D", size=fontsize)
    # plt.tight_layout()

    if args.figure_path[-3:] in ['eps', 'pdf', 'png', 'jpg', 'tif']:
        plt.savefig(args.figure_path)
    else:
        # plt.savefig(args.figure_path + '.eps')
        plt.savefig(args.figure_path + '.pdf')
        plt.savefig(args.figure_path + '.png')
        plt.savefig(args.figure_path + '.jpg')
        plt.savefig(args.figure_path + '.tif')


if __name__ == "__main__":
    main()
