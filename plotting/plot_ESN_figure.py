import os
import pickle
import argparse

# import skunk
import numpy as np
import yaml
import matplotlib
import seaborn as sns

matplotlib.rcParams["text.usetex"] = True
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from matplotlib.gridspec import GridSpec
from matplotlib.patches import ConnectionPatch, FancyBboxPatch, FancyArrowPatch, ConnectionStyle

import barplots
import cap_bars_single_run
from heatmaps import plot_heatmap, get_heatmap_data, translate
from colors import get_degree_color, get_color, adjust_color


def draw_lines(axes, fig, fill_background=True):
    linewidth = 2.5
    # rho09_color = 'wheat'
    rho09_color = 'darkseagreen'
    rho113_color = 'wheat'
    if fill_background:
        rho09_face_color = rho09_color
        rho113_face_color = rho113_color
        additional_box_width = 0.1
    else:
        rho09_face_color = 'none'
        rho113_face_color = 'none'
        additional_box_width = 0.0
    conn_path = ConnectionPatch(xyA=(0.01, 1.1), xyB=(0.01, -0.23), coordsA='axes fraction', coordsB='axes fraction',
                                axesA=axes['cap_heatmap'], axesB=axes['del_heatmap'], color=rho09_color, lw=linewidth,
                                zorder=-1)
    axes['deg_heatmap'].add_artist(conn_path)
    box_rho9_1 = FancyBboxPatch(xy=(0.33, 0.68), width=0.275+additional_box_width, height=0.3, boxstyle=mpatches.BoxStyle("Round", pad=0.01),
                                transform=fig.transFigure, edgecolor=rho09_color, facecolor=rho09_face_color, zorder=-10, lw=linewidth)
    box_rho9_2 = FancyBboxPatch(xy=(0.65, 0.33), width=0.33, height=0.65, boxstyle=mpatches.BoxStyle("Round", pad=0.01),
                                transform=fig.transFigure, edgecolor=rho09_color, facecolor=rho09_face_color, zorder=-10, lw=linewidth)
    line_rho9 = FancyArrowPatch(posA=(0.67, 0.99), posB=(0.0518, 0.96), arrowstyle='-', transform=fig.transFigure,
                                connectionstyle=ConnectionStyle('Angle', angleA=-180, angleB=-90, rad=0.2),
                                lw=linewidth, color=rho09_color)
    # arrow_rho9 = FancyArrowPatch(posA=(0.5, 0.99), posB=(0.6, 0.99), arrowstyle='->', transform=fig.transFigure, lw=linewidth, color=rho09_color)
    # arrow_rho9 = FancyArrowPatch(posA=(0.25, 0.99), posB=(0.3, 0.99), arrowstyle='->', transform=fig.transFigure, lw=linewidth, color=rho09_color, mutation_scale=10)
    arrow_rho9 = FancyArrowPatch(posA=(0.27, 0.98), posB=(0.3, 0.98), arrowstyle='->', transform=fig.transFigure, lw=1, color='k', mutation_scale=4)
    fig.patches.extend([box_rho9_1, box_rho9_2, line_rho9, arrow_rho9])
    # rho113_color = 'lightgrey'
    conn_path = ConnectionPatch(xyA=(0.382, 1.03), xyB=(0.382, -0.23), coordsA='axes fraction', coordsB='axes fraction',
                                axesA=axes['cap_heatmap'], axesB=axes['del_heatmap'], color=rho113_face_color, lw=linewidth,
                                zorder=-1)
    axes['deg_heatmap'].add_artist(conn_path)
    box_rho113 = FancyBboxPatch(xy=(0.33, 0.33), width=0.275, height=0.3, boxstyle=mpatches.BoxStyle("Round", pad=0.01),
                                transform=fig.transFigure, edgecolor=rho113_color, facecolor=rho113_color, zorder=-10, lw=linewidth)
    line_rho113 = FancyArrowPatch(posA=(0.321, 0.33), posB=(0.119, 0.33), arrowstyle='-', transform=fig.transFigure,
                                  connectionstyle=ConnectionStyle('Angle', angleA=-180, angleB=-90, rad=0.2),
                                  lw=linewidth, color=rho113_color)
    # arrow_rho113 = FancyArrowPatch(posA=(0.25, 0.33), posB=(0.3, 0.33), arrowstyle='->', transform=fig.transFigure, lw=linewidth, color=rho113_color, mutation_scale=10)
    arrow_rho113 = FancyArrowPatch(posA=(0.27, 0.34), posB=(0.3, 0.34), arrowstyle='->', transform=fig.transFigure, lw=1., color='k', mutation_scale=4)
    arrow_01_113 = FancyArrowPatch(posA=(0.39, 0.35), posB=(0.39, 0.28), arrowstyle='->', mutation_scale=10,
                                   transform=fig.transFigure, lw=1, color='black')  # , zorder=-2)
    arrow_2_09 = FancyArrowPatch(posA=(0.59, 0.7), posB=(0.70, 0.27), arrowstyle='->', mutation_scale=10,
                                 transform=fig.transFigure, lw=1, color='black')  # , zorder=-2)
    # arrow_09_2 = FancyArrowPatch(posA=(0.32, 0.33), posB=(0.12, 0.33), arrowstyle='->', transform=fig.transFigure, connectionstyle=ConnectionStyle('Angle', angleA=-180, angleB=-90, rad=0.2), lw=linewidth, color=rho113_color)
    fig.patches.extend([box_rho113, line_rho113, arrow_01_113, arrow_2_09, arrow_rho113])


def plot_delay_tasks(ax, capacity_folder, specrad=0.9, classification_results_folder=None, use_cache=False, use_precalculated=False):
    params_to_filter = {
        'steps': 100000,
        'nodes': 50,
        'nwarmup': 500,
    }

    ax2 = ax.twinx()
    ax2.grid(False)


    cmap = matplotlib.cm.Greys

    classification_delays = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    delay_colors = cmap(np.linspace(0, 1, len(classification_delays)))

    if use_precalculated:
        with open(os.path.join('data', 'ESN_delayed_classification_results.pkl'), 'rb') as class_results_file:
            delay_to_classification_accuracies = pickle.load(class_results_file)
    else:
        # classification_results_folder = '/home/schultetobrinke/nextcloud/Juelich/projects/SNN_capacity/repos/ESN/data/DelayedClassificationdiff-iscaling'
        filter_strings = [
            '_testratio=0.3_',
            '_steps=100000_',
            f'_specrad={specrad}_',
            '_nwarmup=500_',
            '_nodes=50_',
        ]

        classification_folders = os.listdir(classification_results_folder)
        for filterstr in filter_strings:
            classification_folders = [f for f in classification_folders if filterstr in f]
        delay_to_classification_accuracies = {}
        for delay_idx, delay in enumerate(classification_delays):
            print(f'-------------- Delay: {delay} -------------')
            classification_accuracies = {}
            for iscaling in [round(x, 1) for x in np.arange(0.1, 2.001, 0.1)]:
                print(iscaling)

                classification_is_files = [os.path.join(classification_results_folder, f, 'test_results.yml') for f in
                                           classification_folders if
                                           f"inpscaling={iscaling}" in f and f'delay={delay}_' in f]
                classification_accuracies[iscaling] = []
                for classificationfile in classification_is_files:
                    with open(classificationfile, 'r') as classification_results_file:
                        classification_accuracies[iscaling].append(yaml.safe_load(classification_results_file)['accuracy'])
            delay_to_classification_accuracies[delay] = classification_accuracies.copy()

    for delay_idx, delay in enumerate(classification_delays):
        classification_accuracies = delay_to_classification_accuracies[delay]
        ax.plot(list(classification_accuracies.keys()), [np.mean(x) for x in classification_accuracies.values()],
                label=f'class, del {delay}', color=delay_colors[delay_idx])

    ax.set_xlabel(translate('inpscaling'))
    ax.set_ylabel('classification accuracy')

    if use_precalculated:
        with open(os.path.join('data', 'ESN_delays.pkl'), 'rb') as delays_file:
            data = pickle.load(delays_file)
    else:
        data = get_heatmap_data(x_name='specrad', y_name='inpscaling', capacity_folder=capacity_folder,
                                params_to_filter=params_to_filter, get_max_delays=True, use_cache=use_cache)
    inpscalings = list(data[specrad].keys())
    delays = list(data[specrad].values())

    ax2.set_ylabel('max. capacity delay')
    ax2.plot(inpscalings, delays, color=get_color('delay', desaturated=True), label='max. capacity\ndelay',
             linestyle='--', lw=2)
    ax2.legend(loc='center right')

    norm = plt.Normalize(vmin=1, vmax=15)
    cbar = plt.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap), orientation='horizontal',
                        label='classification delay', ticks=classification_delays[::2], pad=0.2, ax=ax)
    cbar.ax.tick_params(labelsize=10)

    return ax


def plot_degree_tasks(ax, capacity_folder, specrad=0.9, use_cache=False, use_precalculated=False,
                      xor_results_folder=None, xorxor_results_folder=None):

    params_to_filter = {
        'steps': 100000,
        'nodes': 50,
        'nwarmup': 500,
    }

    if use_precalculated:
        with open(os.path.join('data', 'ESN_degrees.pkl'), 'rb') as degrees_file:
            data = pickle.load(degrees_file)
    else:
        data = get_heatmap_data(x_name='specrad', y_name='inpscaling', capacity_folder=capacity_folder,
                                params_to_filter=params_to_filter, get_max_degrees=True, use_cache=use_cache)
    inpscalings = list(data[specrad].keys())
    degrees = list(data[specrad].values())

    if use_precalculated:
        with open(os.path.join('data', 'ESN_xor_results.pkl'), 'rb') as xor_file:
            xor_kappas = pickle.load(xor_file)
        with open(os.path.join('data', 'ESN_xorxor_results.pkl'), 'rb') as xorxor_file:
            xorxor_kappas = pickle.load(xorxor_file)
    else:
        # xor_results_folder = '/home/schultetobrinke/nextcloud/Juelich/projects/SNN_capacity/repos/ESN/data/XORdiff-iscaling'
        # xorxor_results_folder = '/home/schultetobrinke/nextcloud/Juelich/projects/SNN_capacity/repos/ESN/data/XORXORdiff-iscaling'

        filter_strings = [
            '_testratio=0.3_',
            '_steps=20000_',
            f'_specrad={specrad}_',
            '_nwarmup=500_',
            '_nodes=50_',
        ]

        xor_folders = os.listdir(xor_results_folder)
        xorxor_folders = os.listdir(xorxor_results_folder)
        for filterstr in filter_strings:
            xor_folders = [f for f in xor_folders if filterstr in f]
            xorxor_folders = [f for f in xorxor_folders if filterstr in f]

        xor_kappas = {}
        xorxor_kappas = {}
        for iscaling in [round(x, 1) for x in np.arange(0.1, 2.001, 0.1)]:
            print(iscaling)

            xor_is_files = [os.path.join(xor_results_folder, f, 'test_results.yml') for f in xor_folders if
                            f"inpscaling={iscaling}" in f]
            xor_kappas[iscaling] = []
            for xorfile in xor_is_files:
                with open(xorfile, 'r') as xor_results_file:
                    xor_kappas[iscaling].append(yaml.safe_load(xor_results_file)['kappa'])

            xorxor_is_files = [os.path.join(xorxor_results_folder, f, 'test_results.yml') for f in xorxor_folders if
                               f"inpscaling={iscaling}" in f]
            xorxor_kappas[iscaling] = []
            for xorxorfile in xorxor_is_files:
                with open(xorxorfile, 'r') as xorxor_results_file:
                    xorxor_kappas[iscaling].append(yaml.safe_load(xorxor_results_file)['kappa'])
    xorline = ax.plot(list(xor_kappas.keys()), [np.mean(x) for x in xor_kappas.values()], label='XOR',
                      color=get_color('XOR'))
    xorxorline = ax.plot(list(xorxor_kappas.keys()), [np.mean(x) for x in xorxor_kappas.values()], label='XORXOR',
                         color=get_color('XORXOR'))
    ax.set_ylabel('kappa score')
    ax.set_xlabel(translate('inpscaling'))
    ax2 = ax.twinx()
    ax2.grid(False)
    degreeline = ax2.plot(inpscalings, degrees, color=get_color('degree'), label='degree')
    ax2.set_ylabel('max degree')

    lines = xorline + xorxorline + degreeline
    labels = [l.get_label() for l in lines]
    ax.legend(lines, labels)

    return ax


def plot_heatmaps(capacity_folder, axes, use_cache=False, use_precalculated=False):
    heatmap_params = {
        'cap_heatmap': {
            'title': 'total capacity',
            'plot_max_degrees': False,
            'plot_max_delays': False,
            'cmap': LinearSegmentedColormap.from_list('', [
                adjust_color(get_color('capacity'), lightness_value=0.95, saturation_value=0.1),
                adjust_color(get_color('capacity'), lightness_value=0.75, saturation_value=0.6),
                adjust_color(get_color('capacity'), saturation_factor=1.),
                adjust_color(get_color('capacity'), saturation_factor=1.5, lightness_factor=0.7), 'black']),
            # 'cmap': LinearSegmentedColormap.from_list('', ['white', get_color('capacity'), get_color('accent')]),
            # 'cmap': sns.color_palette('magma', as_cmap=True),
            # 'cmap': sns.color_palette('rocket', as_cmap=True),
            # 'figure_name': 'ESN_total_cap_heatmap.pdf'
        },
        'deg_heatmap': {
            'title': 'maximum degrees',
            'plot_max_degrees': True,
            'plot_max_delays': False,
            # 'figure_name': 'ESN_max_degree_heatmap.pdf'
            'cmap': LinearSegmentedColormap.from_list('', [
                adjust_color(get_color('degree'), lightness_value=0.95, saturation_value=0.1),
                adjust_color(get_color('degree'), saturation_factor=0.75), 'black']),
            # 'cmap': sns.color_palette('viridis', as_cmap=True),
            # 'cmap': ListedColormap(utils.degree_colors),
            # 'cmap': LinearSegmentedColormap.from_list('mycmap', utils.degree_colors),
        },
        'del_heatmap': {
            'title': 'maximum delays',
            'plot_max_degrees': False,
            'plot_max_delays': True,
            'cmap': LinearSegmentedColormap.from_list('', [
                adjust_color(get_color('delay'), lightness_value=0.95, saturation_factor=1.),
                adjust_color(get_color('delay'), saturation_factor=1.5), 'black']),
            # 'cmap': LinearSegmentedColormap.from_list('', ['white', get_color('delay'), get_color('accent')]),
            # 'cmap': sns.color_palette('mako', as_cmap=True),
            # 'cmap': 'icefire',
            # 'figure_name': 'ESN_max_delay_heatmap.pdf'
        },
    }
    if use_precalculated:
        heatmap_params['cap_heatmap']['precalculated_data_path'] = os.path.join('data', 'ESN_capacities.pkl')
        heatmap_params['deg_heatmap']['precalculated_data_path'] = os.path.join('data', 'ESN_degrees.pkl')
        heatmap_params['del_heatmap']['precalculated_data_path'] = os.path.join('data', 'ESN_delays.pkl')
    for i, (ax_label, params) in enumerate(heatmap_params.items()):
        ax = axes[ax_label]
        params['params_to_filter'] = {
            'steps': 100000,
            'nodes': 50,
            'nwarmup': 500,
        }
        _, ax = plot_heatmap(
            x_name='specrad',
            y_name='inpscaling',
            capacity_folder=capacity_folder,
            cutoff=0.,
            figure_path=None,
            plot_num_trials=False,
            annotate=False,
            ax=ax,
            use_cache=use_cache,
            **params
        )
        ylabel = ax.yaxis.get_label().get_text()
        ax.set_ylabel(ylabel, rotation=0.)

    return axes


def plot_bars_for_different_inputscaling(capacity_folder, axes, use_cache=False, use_precalculated=False):
    nodes = 50
    ax1 = axes['capbars09']
    if use_precalculated:
        precalculated_data_path = os.path.join('data', 'ESN_capbars_rho0.9.pkl')
    else:
        precalculated_data_path = None

    ax1 = barplots.plot_capacity_bars(
        x_name='inpscaling',
        capacity_folder=capacity_folder,
        title=r'capacities ($\rho = 0.9$)',
        params_to_filter={
            'steps': 100000,
            'nodes': nodes,
            'nwarmup': 500,
            'specrad': 0.9,
        },
        cutoff=0.,
        delay_shading_step=1,
        annotate=False,
        annotate_sums=False,
        ax=ax1,
        disable_legend=True,
        use_cache=use_cache,
        precalculated_data_path=precalculated_data_path
    )
    ax1.set_ylim((0, nodes))

    ax2 = axes['capbars113']
    if use_precalculated:
        precalculated_data_path = os.path.join('data', 'ESN_capbars_rho1.13.pkl')
    else:
        precalculated_data_path = None

    ax2 = barplots.plot_capacity_bars(
        x_name='inpscaling',
        capacity_folder=capacity_folder,
        title=r'capacities ($\rho = 1.13$)',
        params_to_filter={
            'steps': 100000,
            'nodes': nodes,
            'nwarmup': 500,
            'specrad': 1.13,
        },
        cutoff=0.,
        delay_shading_step=1,
        annotate=False,
        annotate_sums=False,
        ax=ax2,
        disable_legend=True,
        use_cache=use_cache,
        precalculated_data_path=precalculated_data_path
    )
    ax2.set_ylim((0, nodes))

    return axes


def filter_cap_dict_paths(capacity_folder, inpscaling, specrad):
    dict_paths = [os.path.join(capacity_folder, filename) for filename in os.listdir(capacity_folder)]
    params_to_filter = {
        'steps': 100000,
        'nodes': 50,
        'nwarmup': 500,
    }

    filtered_paths = dict_paths

    for paramname, paramvalue in params_to_filter.items():
        filtered_paths = [p for p in dict_paths if f'{paramname}={paramvalue}_' in p]

    filtered_paths = [dp for dp in filtered_paths if
                      f"_inpscaling={inpscaling}_" in dp and f"_specrad={specrad}_" in dp]

    return filtered_paths


def plot_single_run_bars(capacity_folder, axes, use_precalculated=False):
    if use_precalculated:
        dict_path_f = os.path.join('data', 'ESN_single_capacities_i2.0_rho0.9.pkl')
        dict_path_g = os.path.join('data', 'ESN_single_capacities_i0.1_rho1.13.pkl')
    else:
        dict_path_f = filter_cap_dict_paths(capacity_folder, inpscaling=2.0, specrad=0.9)[0]
        dict_path_g = filter_cap_dict_paths(capacity_folder, inpscaling=0.1, specrad=1.13)[0]

    with open(dict_path_f, 'rb') as dict_file_f:
        dict_f = pickle.load(dict_file_f)

    _, axes['single_cap09'] = cap_bars_single_run.plot_capacity_bars(dict_f, axes['single_cap09'])
    axes['single_cap09'].set_title(r"capacities ($\iota$ = 2, $\rho$ = 0.9)")

    with open(dict_path_g, 'rb') as dict_file_g:
        dict_g = pickle.load(dict_file_g)
    _, axes['single_cap113'] = cap_bars_single_run.plot_capacity_bars(dict_g, axes['single_cap113'])
    axes['single_cap113'].set_title(r"capacities ($\iota$ = 0.1, $\rho$ = 1.13)")

    return axes


def create_legend(figures_folder, axes):
    ax = axes['legend']

    # linewidth = 8
    linewidth = 5
    handles = []

    maxdegree = 11
    for i in range(maxdegree):
        degree = i + 1
        handle, = ax.plot([-10], [0], label=f'degree {degree}', color=get_degree_color(degree), linewidth=linewidth)
        handles.append(handle)
    ax.set_xlim((0, 1))
    ax.set_ylim((0, 1))
    ax.legend(loc='center', ncol=6, prop={'size': 7})
    ax.axis('off')

    return axes


def add_subplot_letters(fig):
    letter_size = 16
    y1 = 0.955
    y1_5 = 0.71
    y2 = 0.63
    y3 = 0.29
    y3_5 = 0.265
    x1 = 0.01
    x2 = 0.325
    x3 = 0.645
    fig.text(x1, y1, "A", size=letter_size)
    fig.text(x1, y2, "B", size=letter_size)
    fig.text(x1, y3, "C", size=letter_size)
    fig.text(x2, y1, "D", size=letter_size)
    fig.text(x2, y2-0.025, "E", size=letter_size)
    fig.text(x2, y3_5, "F", size=letter_size)
    fig.text(x3, y1, "G", size=letter_size)
    fig.text(x3, y1_5, "H", size=letter_size)
    fig.text(x3, y3_5, "I", size=letter_size)


def create_capbars_inputscaling_svgs(figures_folder, capacity_folder, height, width, use_cache=False,
                                     use_old_svgs=False):
    # plt.rcParams['figure.constrained_layout.use'] = True
    fig_capbars1_path = os.path.join(figures_folder, 'capbars1.svg')
    fig_capbars2_path = os.path.join(figures_folder, 'capbars2.svg')
    if not use_old_svgs:
        fig_capbars1, ax_capbars1, fig_capbars2, ax_capbars2 = plot_bars_for_different_inputscaling(capacity_folder,
                                                                                                    axes=None,
                                                                                                    width=width,
                                                                                                    height=height,
                                                                                                    use_cache=use_cache)
        fig_capbars1.savefig(fig_capbars1_path)
        fig_capbars2.savefig(fig_capbars2_path)

    return fig_capbars1_path, fig_capbars2_path


def create_heatmap_svgs(figures_folder, capacity_folder, height, width, use_cache=False):
    heatmap_figlist, heatmap_axlist = plot_heatmaps(capacity_folder, axes=None, width=width, height=height,
                                                    use_cache=use_cache)
    heatmap_paths = {}
    for name, heatmap_fig in zip(['capacity', 'degrees', 'delays'], heatmap_figlist):
        heatmap_fig_path = os.path.join(figures_folder, f'heatmap_{name}.asdf.svg')
        heatmap_paths[name] = heatmap_fig_path
        heatmap_fig.savefig(heatmap_fig_path)
    # heatmap_paths = {}
    # for name in ['capacity', 'degrees', 'delays']:
    #     heatmap_fig_path = os.path.join(figures_folder, f'heatmap_{name}.asdf.svg')
    #     heatmap_paths[name] = heatmap_fig_path

    return heatmap_paths


def create_single_bars_svgs(figures_folder, capacity_folder, height, width, use_old_svgs=False):
    fig_single1_path = os.path.join(figures_folder, 'single1.svg')
    fig_single2_path = os.path.join(figures_folder, 'single2.svg')

    if not use_old_svgs:
        fig_single1, ax_single1, fig_single2, ax_single2 = plot_single_run_bars(capacity_folder, axes=None)
        fig_single1.savefig(fig_single1_path)
        fig_single2.savefig(fig_single2_path)

    return fig_single1_path, fig_single2_path


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


def parse_cmd():
    parser = argparse.ArgumentParser()

    parser.add_argument('--capacity_folder', type=str, help='Path to the folder with the capacity results',
                        default=None)
    parser.add_argument('--figures_folder', type=str, help='Path to the folder where the figures will be stored',
                        default='figures')
    parser.add_argument('--xor_results_folder', type=str, help='Path to the folder with the XOR task results',
                        default=None)
    parser.add_argument('--xorxor_results_folder', type=str, help='Path to the folder with the XORXOR task results',
                        default=None)
    parser.add_argument('--classification_results_folder', type=str, help='Path to the folder with the delayed '
                                                                          'classification task results',
                        default=None)
    parser.add_argument('--use_precalculated', action='store_true', help='Use precalculated data for the figure.')

    return parser.parse_args()


def get_axes(fig):
    # inner_vspace = 0.1
    # outer_vspace = 0.05
    # inner_hspace = 0.12
    # outer_hspace = 0.1
    # w = (1 / 3) - (2 / 3) * inner_vspace - (2 / 3) * outer_vspace
    # h = (1 / 3) - (2 / 3) * inner_hspace - (2 / 3) * outer_hspace
    # task_diff = 0.2 * h
    # legend_diff = 0.35 * h / 2
    # legend_offset = 0.05

    inner_vspace = 0.1
    left = 0.05
    right = 0.07
    inner_hspace = 0.12
    top = 0.05
    bottom = 0.08
    w = (1 / 3) - (2 / 3) * inner_vspace - (1 / 3) * left - (1 / 3) * right
    h = (1 / 3) - (2 / 3) * inner_hspace - (1 / 3) * top - (1 / 3) * bottom
    task_diff = 0.2 * h
    legend_diff = 0.4 * h / 2
    legend_offset = 0.05

    x1 = 0. + left
    x2 = (1 / 3) + (1 / 3) * inner_vspace + (2 / 3) * left - (1 / 3) * right
    x3 = (2 / 3) + (2 / 3) * inner_vspace + (1 / 3) * left - (2 / 3) * right
    y3 = (2 / 3) + (2 / 3) * inner_hspace - (2 / 3) * top + (1 / 3) * bottom
    y2 = (1 / 3) + (1 / 3) * inner_hspace - (1 / 3) * top + (2 / 3) * bottom
    y1_5 = y2 / 2 - legend_diff + (1.5 / 3) * bottom - legend_offset
    y1 = 0. + bottom

    axes = {
        'deg_heatmap': fig.add_axes([x1, y2, w, h]),
        'del_heatmap': fig.add_axes([x1, y1, w, h]),
        'cap_heatmap': fig.add_axes([x1, y3, w, h]),
        'legend': fig.add_axes(
            [x2 - 0.03, y1 - legend_offset, w * 2 + inner_vspace, h / 2 - inner_hspace / 2 - legend_diff]),
        # 'capbars113': fig.add_axes([x2, y2, w, h]),
        'capbars113': fig.add_axes([x2, y2 - (1 / 6) * inner_hspace, w, h]),
        'capbars09': fig.add_axes([x2, y3, w, h]),
        'single_cap113': fig.add_axes([x2, y1_5 - (1 / 6) * inner_hspace, w,
                                       h / 2 - inner_hspace / 2 + legend_diff + legend_offset - (
                                                   1 / 6) * inner_hspace]),
        'single_cap09': fig.add_axes([x3, y1_5 - (1 / 6) * inner_hspace, w,
                                      h / 2 - inner_hspace / 2 + legend_diff + legend_offset - (
                                              1 / 6) * inner_hspace]),
        'deg_tasks': fig.add_axes([x3, y3 + task_diff, w, h - task_diff]),
        'del_tasks': fig.add_axes([x3, y2 - (1 / 3) * inner_hspace, w, h + task_diff + (2 / 3) * inner_hspace]),
    }

    return axes


def main(capacity_folder=None, figures_folder='figures', xor_results_folder=None, xorxor_results_folder=None,
         classification_results_folder=None, use_cache=True, use_precalculated=True):

    os.makedirs(figures_folder, exist_ok=True)

    setup_pyplot()

    fig = plt.figure(figsize=(7.5, 5.625))
    axes = get_axes(fig)

    print('start degree task')
    plot_degree_tasks(axes['deg_tasks'], capacity_folder=capacity_folder, use_cache=use_cache,
                      xor_results_folder=xor_results_folder, xorxor_results_folder=xorxor_results_folder,
                      use_precalculated=use_precalculated)
    print('starg delay tasks')
    plot_delay_tasks(axes['del_tasks'], capacity_folder=capacity_folder,
                     classification_results_folder=classification_results_folder, use_cache=use_cache,
                     use_precalculated=use_precalculated)

    sns.set()
    setup_pyplot()

    print('start single bars')
    plot_single_run_bars(capacity_folder=capacity_folder, axes=axes, use_precalculated=use_precalculated)
    print('start heatmaps')
    plot_heatmaps(capacity_folder=capacity_folder, axes=axes, use_cache=use_cache, use_precalculated=use_precalculated)
    print('start capbars')
    plot_bars_for_different_inputscaling(capacity_folder=capacity_folder, axes=axes, use_cache=use_cache,
                                         use_precalculated=use_precalculated)
    print('start legend')
    create_legend(figures_folder, axes=axes)

    draw_lines(axes, fig)

    add_subplot_letters(fig)

    fig.savefig(os.path.join(figures_folder, 'ESN_figure.pdf'))
    # fig.savefig(os.path.join(args.figures_folder, 'ESN_figure.eps'))
    # fig.savefig(os.path.join(args.figures_folder, 'ESN_figure.jpg'))
    # fig.savefig(os.path.join(args.figures_folder, 'ESN_figure.png'))
    # plt.show()


if __name__ == "__main__":
    main(**vars(parse_cmd()))
