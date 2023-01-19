import argparse
import subprocess
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from utils.general_utils import to_squared_corr
from colors import adjust_color


def parse_cmd():
    parser = argparse.ArgumentParser()

    parser.add_argument('--figure_output', '-o', help="Path to store the resulting figure.")

    return parser.parse_args()


def main():
    args = parse_cmd()
    # plt.rc('font', family='serif')
    plt.rc('text', usetex=True)
    alpha_list = np.arange(0., 1.01, 0.01)

    low_color = 'cyan'
    high_color = 'magenta'
    var_ratio_colors = {
        0.01: adjust_color(low_color, saturation_factor=1),
        0.1: adjust_color(low_color, saturation_factor=0.66),
        0.5: adjust_color(low_color, saturation_factor=0.33),
        1.: 'k',
        2.: adjust_color(high_color, saturation_factor=0.33),
        10.: adjust_color(high_color, saturation_factor=0.66),
        100.: adjust_color(high_color, saturation_factor=1.)
    }

    # fig, ax = plt.subplots(ncols=2, figsize=(8, 3))
    fig, ax = plt.subplots(ncols=2, figsize=(5, 2))

    for var_ratio, color in var_ratio_colors.items():
        if var_ratio >= 1:
            label = int(var_ratio)
        else:
            label = var_ratio

        ax[0].plot(alpha_list, to_squared_corr(alpha_list, var_ratio=var_ratio), color=color)
        ax[1].plot(alpha_list, np.sqrt(to_squared_corr(alpha_list, var_ratio=var_ratio)), color=color, label=label)

    ax[0].plot(alpha_list, alpha_list, '--', color='grey', alpha=0.5, label='linear')
    ax[1].plot(alpha_list, alpha_list, '--', color='grey', alpha=0.5)
    ax[0].set_ylabel('capacity')
    ax[1].set_ylabel('$\sqrt{\mathrm{capacity}}$')
    ax[0].set_xlabel(r'$\alpha$')
    ax[1].set_xlabel(r'$\alpha$')
    ax[0].legend()
    ax[1].legend(title=r'$\frac{\mathrm{var}(b_l)}{\mathrm{var}(y_l)}$')

    text_y = 0.94
    text_size = 16.
    fig.text(0.015, text_y, 'A', size=text_size)
    fig.text(0.505, text_y, 'B', size=text_size)

    plt.tight_layout()

    if args.figure_output is None:
        plt.show()
    else:
        if args.figure_output.endswith('.eps'):
            # rasterization to enable transparency
            figure_path = Path(args.figure_output)
            pdf_path = figure_path.parent / f'{figure_path.stem}.pdf'
            plt.savefig(pdf_path)
            subprocess.run(['pdftops', '-eps', '-r', '1000', f'{pdf_path}', f'{figure_path}'])
            no_transparency_path = figure_path.parent / f'{figure_path.stem}.not-transparent.eps'
            plt.savefig(no_transparency_path)
        else:
            plt.savefig(args.figure_output)


if __name__ == "__main__":
    main()
