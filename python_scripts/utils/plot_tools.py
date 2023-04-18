from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_PRETTY_VAR_NAMES_MATH = {
    "const": r"$\beta_0$",
    "rts": r"$RT$",
    "max_sto": r"$S_{max}$",
    "rel_inf_corr": r"$r(D, I)$",
    "storage_pre": r"$S_{t-1}$",
    "release_pre": r"$D_{t-1}$",
    "inflow": r"$NI_t$",
    "inflow2": r"$NI^2$",
    "release_pre2": r"$R_{t-1}^2$",
    "sto_diff": r"$\Delta S_{t-1}$",
    "release_roll7": r"$\overline{S}_{t-1}^7$",
    "inflow_roll7": r"$\overline{NI}_{t}^7$",
    "storage_x_inflow": r"$S_{t-1} \times NI_t$",
}

_PRETTY_VAR_NAMES_MATH_LOWER = {
    "const": r"$\beta_0$",
    "rts": r"$RT$",
    "max_sto": r"$S_{max}$",
    "rel_inf_corr": r"$r(S, NI)$",
    "storage_pre": r"$s_{t-1}$",
    "release_pre": r"$d_{t-1}$",
    "inflow": r"$ni_t$",
    "inflow2": r"$ni_t^2$",
    "release_pre2": r"$d_{t-1}^2$",
    "sto_diff": r"$\Delta s_{t-1}$",
    "release_roll7": r"$\overline{s}_{t-1}^7$",
    "inflow_roll7": r"$\overline{ni}_{t}^7$",
    "storage_x_inflow": r"$s_{t-1} \times ni_t$",
}

_PRETTY_VAR_NAMES = {
    "rts": "Residence Time",
    "max_sto": "Max. Storage",
    "rel_inf_corr": "Pearson Cor. for Inflow & Release",
    "storage_pre": "Lag-1 Storage",
    "release_pre": "Lag-1 Release",
    "inflow": "Net Inflow",
}

VAR_ORDER = [
    "const",
    "storage_pre",
    "release_pre",
    "inflow",
    "sto_diff",
    "release_roll7",
    "inflow_roll7",
    "storage_x_inflow",
    "inflow2",
    "release_pre2",
]


def get_pretty_var_name(var: str, math=True, lower=False):
    if math:
        if lower:
            return _PRETTY_VAR_NAMES_MATH_LOWER.get(var, var)
        else:
            return _PRETTY_VAR_NAMES_MATH.get(var, var)
    else:
        return _PRETTY_VAR_NAMES.get(var, var)


def mxbline(m: float, b: float, ax=None, **kwargs):
    """Draw a line with slope m and intercept b.
    Additional kwargs are passed to ax.plot

    Args:
        m (float): slope
        b (float): intercept
        ax (Axes, optional): Matplotlib axes to draw on. Defaults to None.
    """
    if not ax:
        ax = plt.gca()
    xmin, xmax = ax.get_xlim()
    x = np.linspace(xmin, xmax, 100)
    y = m * x + b
    ax.plot(x, y, **kwargs)


def determine_grid_size(N: int) -> Tuple[int, int]:
    """Determine the optimal grid size for a given number of plots.

    Args:
        N (int): Number of plots on grid.

    Returns:
        Tuple[int, int]: Optimal grid size
    """
    if N <= 3:
        return (N, 1)
    else:
        poss_1 = [(i, N // i) for i in range(2, int(N**0.5) + 1) if N % i == 0]
        poss_2 = [
            (i, (N + 1) // i)
            for i in range(2, int((N + 1) ** 0.5) + 1)
            if (N + 1) % i == 0
        ]
        poss = poss_1 + poss_2
        min_index = np.argmin([sum(i) for i in poss])
        return poss[min_index]


def get_tick_years(index, ax):
    nticks = len(ax.get_xticks()) - 1
    start_year = index.min().year + 1
    stop_year = index.max().year
    nyears = stop_year - start_year
    tick_years = np.arange(start_year, stop_year, max([nyears // max([nticks, 1]), 1]))

    ticks = [
        np.where(index == pd.Timestamp(year=i, day=1, month=1))[0][0]
        for i in tick_years
    ]
    return tick_years, ticks


def custom_bar_chart(data, width=0.8, ax=None, colors=None, error=None, **kwargs):
    """Create a custom bar chart with multiple groups.

    Args:
        data (pd.DataFrame): dataframe with data to plot. Index is x-axis,
            columns are groups.
        width (float, optional): Proportion of available space to use for each cluster.
            Defaults to 0.8.
        ax (Axes, optional): Axes to plot on. If none, gets current axes.
            Defaults to None.
        colors (dict, optional): Dictionary of colors for each group. Defaults to None.
        error (dict, optional): Dictionary of error bars to plot. Defaults to None.
    """
    show = False
    if not ax:
        show = True
        ax = plt.gca()

    x = data.index
    xticks = np.arange(len(x))

    groups = data.columns
    ngroup = len(groups)
    bar_width = width / ngroup

    if ngroup % 2 == 0:
        offset_multipliers = np.array([i - ngroup // 2 + 0.5 for i in range(ngroup)])
    else:
        offset_multipliers = np.array([i - ngroup // 2 for i in range(ngroup)])

    offsets = bar_width * offset_multipliers

    for group, offset in zip(groups, offsets):
        ax.bar(
            xticks + offset,
            data[group],
            width=bar_width,
            color=colors[group] if colors else None,
            **kwargs,
        )
        if error is not None:
            ax.errorbar(
                xticks + offset,
                data[group],
                yerr=error[group],
                fmt="none",
                capsize=1,
                color="k",
            )

    if show:
        plt.show()
