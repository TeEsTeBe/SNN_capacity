import os
import pickle
import argparse

import skunk
import matplotlib
import seaborn as sns

matplotlib.rcParams["text.usetex"] = True
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, LinearSegmentedColormap

import barplots
import cap_bars_single_run
from heatmaps import plot_heatmap
from colors import get_degree_color, get_color, adjust_color


# import utils


def plot_task_figures():



def plot_heatmaps(capacity_folder, axes=None, width=None, height=None, use_cache=False):
    heatmap_params = {
        'A': {
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
        'B': {
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
        'C': {
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
    figlist = []
    axlist = []
    for i, (ax_label, params) in enumerate(heatmap_params.items()):
        if axes is None:
            fig, ax = plt.subplots(figsize=(width, height))
        else:
            fig = None
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
        figlist.append(fig)
        axlist.append(ax)

    return figlist, axlist


def plot_bars_for_different_inputscaling(capacity_folder, axes=None, width=None, height=None, use_cache=False):
    nodes = 50
    if axes is None:
        fig1, ax1 = plt.subplots(figsize=(width, height))
    else:
        fig1 = None
        ax1 = axes['D']
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
        use_cache=use_cache
    )
    ax1.set_ylim((0, nodes))

    if axes is None:
        fig2, ax2 = plt.subplots(figsize=(width, height))
    else:
        fig2 = None
        ax2 = axes['E']

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
        use_cache=use_cache
    )
    ax2.set_ylim((0, nodes))

    return fig1, ax1, fig2, ax2


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


def plot_single_run_bars(capacity_folder, axes=None, width=None, height=None):
    dict_path_f = filter_cap_dict_paths(capacity_folder, inpscaling=2.0, specrad=0.9)[0]
    with open(dict_path_f, 'rb') as dict_file_f:
        dict_f = pickle.load(dict_file_f)
    if axes is None:
        fig1, ax1 = plt.subplots(figsize=(width, height))
    else:
        ax1 = axes['F']
        fig1 = None
    _, ax1 = cap_bars_single_run.plot_capacity_bars(dict_f, ax1)
    ax1.set_title(r"capacities ($\iota$ = 2, $\rho$ = 0.9)")

    dict_path_g = filter_cap_dict_paths(capacity_folder, inpscaling=0.1, specrad=1.13)[0]
    with open(dict_path_g, 'rb') as dict_file_g:
        dict_g = pickle.load(dict_file_g)
    if axes is None:
        fig2, ax2 = plt.subplots(figsize=(width, height))
    else:
        ax2 = axes['G']
        fig2 = None
    _, ax2 = cap_bars_single_run.plot_capacity_bars(dict_g, ax2)
    ax2.set_title(r"capacities ($\iota$ = 0.1, $\rho$ = 1.13)")

    return fig1, ax1, fig2, ax2


def create_legend_svg(figures_folder, width, height):
    fig_legend, ax = plt.subplots(figsize=(width, height))

    # linewidth = 8
    linewidth = 10
    handles = []

    maxdegree = 11
    for i in range(maxdegree):
        degree = i + 1
        handle, = ax.plot([-10], [0], label=f'degree {degree}', color=get_degree_color(degree), linewidth=linewidth)
        handles.append(handle)
    ax.set_xlim((0, 1))
    ax.set_ylim((0, 1))
    ax.legend(loc='center', ncol=6, prop={'size': 12})
    ax.axis('off')

    fig_legend_path = os.path.join(figures_folder, 'legend.svg')
    fig_legend.savefig(fig_legend_path)

    return fig_legend_path


def add_subplot_letters(fig):
    letter_size = 28
    y1 = 0.965
    y1_5 = 0.74
    y2 = 0.66
    y3 = 0.31
    x1 = 0.01
    x2 = 0.335
    x3 = 0.68
    fig.text(x1, y1, "A", size=letter_size)
    fig.text(x1, y2, "B", size=letter_size)
    fig.text(x1, y3, "C", size=letter_size)
    fig.text(x2, y1, "D", size=letter_size)
    fig.text(x2, y2, "E", size=letter_size)
    fig.text(x2, y3, "F", size=letter_size)
    fig.text(x3, y1, "G", size=letter_size)
    fig.text(x3, y1_5, "H", size=letter_size)
    fig.text(x3, y3, "I", size=letter_size)


def create_capbars_inputscaling_svgs(figures_folder, capacity_folder, height, width, use_cache=False,
                                     use_old_svgs=False):
    plt.rcParams['figure.constrained_layout.use'] = True
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
        fig_single1, ax_single1, fig_single2, ax_single2 = plot_single_run_bars(capacity_folder, axes=None,
                                                                                width=width, height=height)
        fig_single1.savefig(fig_single1_path)
        fig_single2.savefig(fig_single2_path)

    return fig_single1_path, fig_single2_path


def setup_pyplot():
    plt.rcParams['figure.constrained_layout.use'] = True
    plt.rcParams['figure.constrained_layout.w_pad'] = 0.05
    # plt.rcParams['figure.constrained_layout.w_pad'] = 0.0
    plt.rcParams['figure.constrained_layout.h_pad'] = 0.05
    # plt.rcParams['figure.constrained_layout.h_pad'] = 0.0
    SMALL_SIZE = 12
    MEDIUM_SIZE = 18
    BIGGER_SIZE = 18
    plt.rc('font', size=SMALL_SIZE)  # controls default text sizes
    plt.rc('axes', titlesize=BIGGER_SIZE)  # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)  # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title


def parse_cmd():
    parser = argparse.ArgumentParser()

    parser.add_argument('--capacity_folder', type=str, help='Path to the folder with the capacity results',
                        required=True)
    parser.add_argument('--figures_folder', type=str, help='Path to the folder where the figures will be stored',
                        required=True)

    return parser.parse_args()


def main():
    args = parse_cmd()

    os.makedirs(args.figures_folder, exist_ok=True)

    setup_pyplot()

    use_cache = True
    use_old_capbar_svgs = True
    use_old_single_bars_svgs = True
    width = 4
    height = 3

    print('start single bars')
    fig_single1_path, fig_single2_path = create_single_bars_svgs(args.figures_folder, args.capacity_folder,
                                                                 height * 0.75, width,
                                                                 use_old_svgs=use_old_single_bars_svgs)
    print('start heatmaps')
    heatmap_paths = create_heatmap_svgs(args.figures_folder, args.capacity_folder, height, width, use_cache=use_cache)
    print('start capbars')
    fig_capbars1_path, fig_capbars2_path = create_capbars_inputscaling_svgs(args.figures_folder, args.capacity_folder,
                                                                            height, width, use_cache=use_cache,
                                                                            use_old_svgs=use_old_capbar_svgs)
    # fig_legend_path = create_legend_svg(args.figures_folder, width=width * 2, height=height / 4)
    print('start legend')
    fig_legend_path = create_legend_svg(args.figures_folder, width=width * 2, height=height * 0.25)

    plt.clf()
    plt.rcParams['figure.constrained_layout.h_pad'] = 0.0
    # fig = plt.figure(constrained_layout=True, figsize=(width*3,height*2.25))
    fig = plt.figure(constrained_layout=True, figsize=(width * 3, height * 3.))
    plt.axis('off')
    axes = fig.subplot_mosaic(
        """
        AAACCCHHH
        AAACCCHHH
        AAACCCHHH
        AAACCCHHH
        AAACCCHHH
        AAACCCHHH
        AAACCCIII
        AAACCCIII
        BBBDDDIII
        BBBDDDIII
        BBBDDDIII
        BBBDDDIII
        BBBDDDIII
        BBBDDDIII
        BBBDDDIII
        BBBDDDIII
        EEEFFFGGG
        EEEFFFGGG
        EEEFFFGGG
        EEEFFFGGG
        EEEFFFGGG
        EEEFFFGGG
        EEE000000
        EEE000000
        """
    )
    #     """
    #     AAACCCEEE
    #     AAACCCEEE
    #     AAACCCEEE
    #     AAACCCEEE
    #     BBBDDDFFF
    #     BBBDDDFFF
    #     BBBDDDGGG
    #     BBBDDDGGG
    #     XXXHHHHHH
    #     """
    # )

    add_subplot_letters(fig)

    for letter, name in zip(['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', '0'],
                            ['capacity', 'degrees', 'bars1', 'bars2', 'delays', 'single1', 'single2', 'task-degrees',
                             'task-delays', 'legend']):
        skunk.connect(axes[letter], name)
        axes[letter].axis('off')
    # axes['X'].axis('off')

    svg = skunk.insert({
        'capacity': heatmap_paths['capacity'],
        'degrees': heatmap_paths['degrees'],
        'bars1': fig_capbars1_path,
        'bars2': fig_capbars2_path,
        'delays': heatmap_paths['delays'],
        'single1': fig_single1_path,
        'single2': fig_single2_path,
        'task-degrees': '/home/schultetobrinke/projects/recurrence/repos/capacity_visualisation/capacity_visualisation/capacity_plots_new/ESN_plots/ESN_plots_v1/ESN-rho0.95_task_degree_plot.svg',
        'task-delays': '/home/schultetobrinke/projects/recurrence/repos/capacity_visualisation/capacity_visualisation/capacity_plots_new/ESN_plots/ESN_plots_v1/ESN-rho0.95_task_delay_plot.svg',
        'legend': fig_legend_path,
    })

    with open(os.path.join(args.figures_folder, 'ESN_plots_skunk_v3.svg'), 'w') as f:
        f.write(svg)

    import cairosvg
    cairosvg.svg2pdf(bytestring=svg, write_to=os.path.join(args.figures_folder, 'ESN_plots_skunk_v3.pdf'))

    # plt.savefig(os.path.join(args.figures_folder, 'ESN_plots.pdf'))
    # plt.savefig(os.path.join(args.figures_folder, 'ESN_plots.svg'))


if __name__ == "__main__":
    main()
