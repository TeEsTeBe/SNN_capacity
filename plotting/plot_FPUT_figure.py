import pickle
from evaluation.calculate_task_correlations import get_correlation
import cap_bars_single_run
import os
import heatmaps
from heatmaps import get_heatmap_data, get_task_results
import numpy as np
import matplotlib.pyplot as plt
from colors import get_degree_color, get_color, adjust_color
from pathlib import Path
import matplotlib
from matplotlib.patches import Rectangle
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

# import pylustrator; pylustrator.start()

params_to_filter={'trial': 0}
use_cache=True

def filter_cap_dict_paths(capacity_folder, input_amplitude, input_duration):
    dict_paths = [
        os.path.join(capacity_folder, filename)
        for filename in os.listdir(capacity_folder)
    ]
    # params_to_filter = {
    #     'steps': 100000,
    #     'nodes': 50,
    #     'nwarmup': 500,
    # }

    filtered_paths = dict_paths

    # for paramname, paramvalue in params_to_filter.items():
    #     filtered_paths = [p for p in dict_paths if f'{paramname}={paramvalue}_' in p]

    filtered_paths = [
        dp
        for dp in filtered_paths
        if f"input_amplitude={input_amplitude}_" in dp
        and f"input_duration={input_duration}_" in dp
    ]

    return filtered_paths


def plot_single_correlations_plot(
    ax,
    bar_width,
    cap_dict,
    cap_folder,
    cap_title,
    task_base_folder,
    use_spearmanr,
    use_cache=use_cache,
    params_to_filter={},
):
    # fig, ax = plt.subplots(figsize=(5, 8))
    # fac = 1.25  # Poster
    # fig, ax = plt.subplots(figsize=(fac*5, fac*4))  # Poster
    # fac = 1.  # Poster
    # fig, ax = plt.subplots(figsize=(fac * 4, fac * 4))  # Poster
    ax.set_ylim((-1, 1))
    print(cap_title)
    colors = {
        "capacity": get_color("capacity"),
        # 'nonlin. cap. delay 5': '#77CBB9',
        "nonlin. cap. delay 5": get_color("accent", desaturated=True),
        "nonlinear capacity\ndelay 5": "#77CBB9",
        # 'nonlin. cap. delay 10': '#2C0735',
        "nonlin. cap. delay 10": get_color("accent", desaturated=True),
        "nonlinear capacity\ndelay 10": "#2C0735",
        "degrees": get_color("degree"),  # 'wheat',
        "delays": get_color("delay"),  # 'plum',
    }
    # capacity_info_types = ['degrees', 'capacity', 'delays', 'nonlin. cap. delay 10']
    # capacity_info_types = ['degrees', 'capacity', 'delays', 'nonlinear capacity\ndelay 5']
    capacity_info_types = ["degrees", "capacity", "delays", "nonlin. cap. delay 5"]
    # capacity_info_types = ['degrees', 'capacity', 'delays', 'nonlinear capacity\ndelay 10']
    # capacity_info_types = ['degrees', 'capacity', 'delays', 'nonlinear capacity\ndelay 5', 'nonlinear capacity\ndelay 10']
    for info_type_idx, cap_info_type in enumerate(capacity_info_types):
        print(f"\t{cap_info_type}")
        shift_factor = info_type_idx - 1
        get_max_degrees = cap_info_type == "degrees"
        get_max_delays = cap_info_type == "delays"

        if cap_info_type in ["nonlin. cap. delay 5", "nonlinear capacity\ndelay 5"]:
            cap_data_dict = get_heatmap_data(
                x_name="input_duration",
                y_name="input_amplitude",
                capacity_folder=cap_folder,
                params_to_filter=params_to_filter,
                mindelay=5,
                maxdelay=5,
                mindegree=2,
                use_cache=use_cache,
            )
            # params_to_filter={}, mindelay=6, maxdelay=6, mindegree=2)
        elif cap_info_type in ["nonlin. cap. delay 10", "nonlinear capacity\ndelay 10"]:
            cap_data_dict = get_heatmap_data(
                x_name="input_duration",
                y_name="input_amplitude",
                capacity_folder=cap_folder,
                params_to_filter=params_to_filter,
                mindelay=10,
                maxdelay=10,
                mindegree=1,
                use_cache=use_cache,
            )
            # params_to_filter={}, mindelay=11, maxdelay=11, mindegree=1)
        else:
            cap_data_dict = get_heatmap_data(
                x_name="input_duration",
                y_name="input_amplitude",
                capacity_folder=cap_folder,
                params_to_filter=params_to_filter,
                get_max_delays=get_max_delays,
                get_max_degrees=get_max_degrees,
                use_cache=use_cache,
            )

        if np.max(list(cap_data_dict[1.0].keys())) == 3.0:
            tmp_cap_data_dict = {}
            for dur, amp_dict in cap_data_dict.items():
                tmp_cap_data_dict[dur] = {}
                for amp, cap in amp_dict.items():
                    tmp_cap_data_dict[dur][round(amp * 0.2, 2)] = cap
            cap_data_dict = tmp_cap_data_dict

        tasknames = []
        correlations = []

        for task_name, task_group in cap_dict["tasks"].items():
            print(f"\n______ {task_name} ___________")
            task_folder = os.path.join(task_base_folder, task_group)

            if task_name == "class. del. sum":
                task_data_dict = get_task_results(
                    task_folder,
                    x_param_name="input_duration",
                    y_param_name="input_amplitude",
                    aggregation_type="sum_over_delays",
                    metric="accuracy",
                    use_cache=use_cache,
                ).to_dict()
            elif task_name in [
                "class. max. del.",
                "classification",
                "classi-\nfication",
                "class.",
            ]:
                task_data_dict = get_task_results(
                    task_folder,
                    x_param_name="input_duration",
                    y_param_name="input_amplitude",
                    aggregation_type="max_delay",
                    metric="accuracy",
                    use_cache=use_cache,
                ).to_dict()
            elif "NARMA" in task_name:
                task_data_dict = get_task_results(
                    task_folder,
                    x_param_name="input_duration",
                    y_param_name="input_amplitude",
                    metric="squared_corr_coeff",
                    use_cache=use_cache,
                ).to_dict()
            else:
                task_data_dict = get_task_results(
                    task_folder,
                    x_param_name="input_duration",
                    y_param_name="input_amplitude",
                    use_cache=use_cache,
                ).to_dict()

            if (
                cap_info_type
                in [
                    "nonlin. cap. delay 5",
                    "nonlinear capacity\ndelay 5",
                    "nonlin. cap. delay 10",
                    "nonlinear capacity\ndelay 10",
                ]
                and task_name not in ["NARMA5", "NARMA10"]
            ):
                corr = 0.0
            # elif task_name == 'NARMA10' and '5' in cap_info_type:
            #     corr = 0.
            elif task_name == "NARMA5" and "10" in cap_info_type:
                corr = 0.0
            else:
                corr = get_correlation(
                    cap_data_dict, task_data_dict, use_spearmanr=use_spearmanr
                )
            print(f"\n\t\t{cap_title} correlation({cap_info_type},{task_name}): {corr}")
            correlations.append(corr)
            tasknames.append(task_name)

        x_positions = np.array(list(range(len(tasknames))))
        w = bar_width - (bar_width / 2) * min(abs(shift_factor), 1)
        cap_info_type = (
            cap_info_type if "5" not in cap_info_type else "nonlin. cap. delay 5"
        )  # Poster
        cap_info_type = (
            cap_info_type if "10" not in cap_info_type else "nonlin. cap. delay 10"
        )  # Poster
        ax.bar(
            x=x_positions + ((w + bar_width) / 2 * shift_factor),
            height=correlations,
            width=w,
            color=colors[cap_info_type],
            label=cap_info_type,
        )
        ax.set_xticks(x_positions)
        ax.set_xticklabels(tasknames, rotation=90, fontsize=8)
        ax.tick_params(axis="x", direction="in", pad=-8, length=0.0)
        # ax.set_ylabel('correlation coefficient')
    ax.set_title(cap_title)
    ax.spines["top"].set_color("none")
    ax.spines["bottom"].set_color("none")
    ax.spines["right"].set_color("none")


def transition_plot(
    ax,
    fixed_amplitude,
    cap_folder,
    tasks_folder="../Data/FPUT_tasks",
    params_to_filter={}
):

    colors = {
        "capacity": get_color("capacity"),
        # 'nonlin. cap. delay 5': '#77CBB9',
        "nonlin. cap. delay 5": get_color("accent", desaturated=True),
        "nonlinear capacity\ndelay 5": "#77CBB9",
        # 'nonlin. cap. delay 10': '#2C0735',
        "nonlin. cap. delay 10": get_color("accent", desaturated=True),
        "nonlinear capacity\ndelay 10": "#2C0735",
        "degrees": get_color("degree"),  # 'wheat',
        "delays": get_color("delay"),  # 'plum',
        "xor": get_color("XOR"),
        "narma5": get_color("accent", desaturated=True),
    }

    # Capacity line
    cap_data_dict = get_heatmap_data(
        y_name="input_duration",
        x_name="input_amplitude",
        capacity_folder=cap_folder,
        params_to_filter=params_to_filter,
        get_max_delays=False,
        get_max_degrees=False,
        mindegree=2,
        maxdelay=2,
        use_cache=use_cache,
    )
    cap_data = cap_data_dict[fixed_amplitude]
    ax.plot(
        cap_data.keys(),
        cap_data.values(),
        color=colors["capacity"],
        linestyle="--",
        label="nonlin. 0 delay",
    )

    # Capacity line
    cap_data_dict = get_heatmap_data(
        y_name="input_duration",
        x_name="input_amplitude",
        capacity_folder=cap_folder,
        params_to_filter=params_to_filter,
        get_max_delays=False,
        get_max_degrees=False,
        mindegree=2,
        mindelay=5,
        use_cache=use_cache,
    )
    cap_data = cap_data_dict[fixed_amplitude]
    ax.plot(
        cap_data.keys(),
        cap_data.values(),
        color=colors["capacity"],
        linestyle=":",
        label=r"nonlin. delay $\geq5$",
    )

    # Capacity line
    cap_data_dict = get_heatmap_data(
        y_name="input_duration",
        x_name="input_amplitude",
        capacity_folder=cap_folder,
        params_to_filter=params_to_filter,
        get_max_delays=False,
        get_max_degrees=False,
        use_cache=use_cache,
        maxdelay=2,
    )
    cap_data = cap_data_dict[fixed_amplitude]
    ax.plot(
        cap_data.keys(), cap_data.values(), color=colors["capacity"], label="capacity"
    )

    ax2 = ax.twinx()
    # Task Lines: Xor
    xor_data = get_task_results(
        Path(tasks_folder) / "xor/",
        y_param_name="input_duration",
        x_param_name="input_amplitude",
        use_cache=use_cache,
    ).to_dict()[fixed_amplitude]
    ax2.plot(xor_data.keys(), xor_data.values(), color=colors["xor"], label="XOR")

    # Task Lines: t_Xor
    t_xor_data = get_task_results(
        Path(tasks_folder) / "t_xor/",
        y_param_name="input_duration",
        x_param_name="input_amplitude",
        use_cache=use_cache,
    ).to_dict()[fixed_amplitude]
    ax2.plot(
        t_xor_data.keys(),
        t_xor_data.values(),
        color=colors["xor"],
        linestyle="--",
        label="tXOR",
    )

    # Task Lines: Narma
    narma_data = get_task_results(
        Path(tasks_folder) / "narma5/",
        y_param_name="input_duration",
        x_param_name="input_amplitude",
        use_cache=use_cache,
        metric="squared_corr_coeff",
    ).to_dict()[fixed_amplitude]
    ax2.plot(
        narma_data.keys(), narma_data.values(), color=colors["narma5"], label="NARMA5"
    )

    ax.set_ylabel("capacity")
    ax.set_xlabel(r"$\Delta s$")
    ax2.set_ylabel("performance")

    legend_elements = [
        Line2D([0], [0], color=colors["capacity"], lw=2, label="capacity"),
        Line2D(
            [0],
            [0],
            color=colors["capacity"],
            lw=2,
            linestyle=":",
            label=r"nonlin. delay $\geq5$",
        ),
        Line2D(
            [0],
            [0],
            color=colors["capacity"],
            lw=2,
            linestyle="--",
            label="nonlin. 0 delay",
        ),
        Line2D([0], [0], color=colors["xor"], lw=2, label="XOR"),
        Line2D([0], [0], color=colors["xor"], lw=2, linestyle="--", label="tXOR"),
        Line2D([0], [0], color=colors["narma5"], lw=2, label="NARMA5"),
    ]
    ax.legend(
        handles=legend_elements,
        ncol=2,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.33),
        fontsize=10,
    ).set_in_layout(False)
    # ax2.legend(ncol=2, loc='upper center', bbox_to_anchor=(0.5, -0.29), fontsize=10)


def memory_plot(
    ax,
    cap_folder,
    input_amplitude,
    input_duration,
):
    dict_path_f = filter_cap_dict_paths(
        cap_folder, input_duration=input_duration, input_amplitude=input_amplitude
    )[0]
    with open(dict_path_f, "rb") as dict_file_f:
        dict_f = pickle.load(dict_file_f)
        cap_bars_single_run.plot_capacity_bars(dict_f, ax)

    ax.set_xlabel("delay")
    ax.set_ylabel("capacity")


def color_heatmap_and_ax(
    heatmap_ax, ax, edge_color, input_amplitude_idx, input_duration_idx
):
    heatmap_ax.add_patch(
        Rectangle(
            (input_duration_idx, input_amplitude_idx),
            1,
            1,
            fill=False,
            edgecolor=edge_color,
            lw=2,
            clip_on=False,
        )
    )
    ax.spines["bottom"].set_color(edge_color)
    ax.spines["bottom"].set_linewidth(3)
    ax.spines["top"].set_color(edge_color)
    ax.spines["top"].set_linewidth(3)
    ax.spines["right"].set_color(edge_color)
    ax.spines["right"].set_linewidth(3)
    ax.spines["left"].set_color(edge_color)
    ax.spines["left"].set_linewidth(3)


fig = plt.figure(constrained_layout=True)
axs = fig.subplot_mosaic(
    [
        ["heatmap", [["memory1"], ["memory2"]]],
        ["transition", "corr"],
    ],
)


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
    plt.rc("font", size=SMALL_SIZE)  # controls default text sizes
    plt.rc("axes", titlesize=BIGGER_SIZE)  # fontsize of the axes title
    plt.rc("axes", labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
    plt.rc("xtick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc("ytick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc("legend", fontsize=SMALL_SIZE)  # legend fontsize
    plt.rc("figure", titlesize=BIGGER_SIZE)  # fontsize of the figure title
    # plt.rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
    # plt.rc('font', family='serif')
    matplotlib.rcParams["figure.dpi"] = 600


setup_pyplot()

fixed_amplitude = 0.033
fixed_duration = 48
lower_fixed_duration = 1
tmp_data = get_heatmap_data(
    x_name="input_duration",
    y_name="input_amplitude",
    capacity_folder="../Data/FPUT_capacities",
    params_to_filter=params_to_filter,
    get_max_delays=False,
    get_max_degrees=False,
    use_cache=use_cache,
)
durations = list(tmp_data.keys())
amplitudes = list(tmp_data[durations[0]].keys())


# Heatmap plot
heatmaps.plot_heatmap(
    y_name="input_amplitude",
    x_name="input_duration",
    capacity_folder="../Data/FPUT_capacities",
    title="",
    params_to_filter=params_to_filter,
    cutoff=0.0,
    figure_path=None,
    plot_max_degrees=False,
    plot_max_delays=False,
    plot_num_trials=False,
    annotate=False,
    plot_degree_delay_product=False,
    ax=axs["heatmap"],
    other_filter_keys=None,
    cmap=None,
    mindegree=0,
    maxdegree=np.inf,
    mindelay=0,
    maxdelay=np.inf,
    use_cache=use_cache,
    max_marker_color=None,
    colorbar_label=None,
    cbar_ticks=None,
)
axs['heatmap'].set_xticks([10, 20, 30, 40])
axs['heatmap'].set_xticklabels([10, 20, 30, 40])
axs['heatmap'].set_yticks([10, 20, 30, 40, 50])
axs['heatmap'].set_yticklabels([0.01, 0.02, 0.03, 0.04, 0.05])
axs["heatmap"].set_xlabel(r"$\Delta s$")
axs["heatmap"].set_ylabel(r"$a_{max}$")

# Correlation plot
plot_single_correlations_plot(
    ax=axs["corr"],
    bar_width=0.25,
    cap_dict={
        "axes_letter": "tasks1",
        "cap_groupname": "FPUT_capacities",
        "tasks": {
            "XOR": "xor",
            "tXOR": "t_xor",
            "NARMA5": "narma5",
        },
        "figname": "delete_me.pdf",
    },
    cap_folder="../Data/FPUT_capacities",
    cap_title="",
    task_base_folder="../Data/FPUT_tasks",
    use_spearmanr=False,
    use_cache=use_cache,
    params_to_filter=params_to_filter,
)
axs["corr"].set_ylabel("correlation")

axs["corr"].set_yticks([-1, 0, 1])
axs["corr"].set_yticklabels([-1, 0, 1])
axs["corr"].legend(ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.3), fontsize=10).set_in_layout(False)

# transition_plot
transition_plot(
    ax=axs["transition"],
    cap_folder="../Data/FPUT_capacities",
    fixed_amplitude=fixed_amplitude,
)

# memory plot
memory_plot(
    ax=axs["memory1"],
    input_amplitude=fixed_amplitude,
    input_duration=fixed_duration,
    cap_folder="../Data/FPUT_capacities",
)
color_heatmap_and_ax(
    ax=axs["memory1"],
    input_amplitude_idx=amplitudes.index(fixed_amplitude),
    input_duration_idx=durations.index(fixed_duration),
    heatmap_ax=axs["heatmap"],
    edge_color=get_color("XOR"),
)
memory_plot(
    ax=axs["memory2"],
    input_amplitude=fixed_amplitude,
    input_duration=lower_fixed_duration,
    cap_folder="../Data/FPUT_capacities",
)
color_heatmap_and_ax(
    ax=axs["memory2"],
    input_amplitude_idx=amplitudes.index(fixed_amplitude),
    input_duration_idx=durations.index(lower_fixed_duration),
    heatmap_ax=axs["heatmap"],
    edge_color=get_color("accent"),
)

legend_elements = [
     Patch(facecolor=get_degree_color(1), label='1'),
     Patch(facecolor=get_degree_color(2), label='2'),
     Patch(facecolor=get_degree_color(3), label='3'),
]
axs['memory2'].legend(
    title='degrees',
    handles=legend_elements,
    ncol=1,
    loc="center right",
    bbox_to_anchor=(1.45, 1.45),
    fontsize=10,
    title_fontsize=10,
).set_in_layout(False)

ax_translation = dict(
    A=axs["heatmap"],
    B=axs["transition"],
    E=axs["corr"],
    D=axs["memory2"],
    C=axs["memory1"],
)
for label, ax in ax_translation.items():
    ax.text(
        -0.3, 1.10, label, transform=ax.transAxes, fontsize=16, va="top", ha="right"
    )
fig.canvas.draw()
for ax in axs.values():
    leg = ax.get_legend()
    if leg is not None:
        leg.set_in_layout(True)
    # we don't want the layout to change at this point.
fig.set_layout_engine(None)
#% start: automatic generated code from pylustrator
plt.figure(1).ax_dict = {ax.get_label(): ax for ax in plt.figure(1).axes}
import matplotlib as mpl
plt.figure(1).ax_dict["<colorbar>"].set_position([0.424495+0.02, 0.602750, 0.013252, 0.353802])
#% end: automatic generated code from pylustrator
plt.savefig("FPUT_heatmap.pdf",bbox_inches='tight')
# plt.show()
