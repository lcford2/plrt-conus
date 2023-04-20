import calendar
import os
import re
from collections import defaultdict
from multiprocessing import cpu_count

import geopandas as gpd
import matplotlib as mpl  # noqa: F401
import matplotlib.gridspec as mgridspec  # noqa: F401
import matplotlib.patches as mpatch
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd
import scienceplots  # noqa: F401
import seaborn as sns
from fit_plrt_model import get_params_and_groups
from joblib import Parallel, delayed
from matplotlib.cm import ScalarMappable, get_cmap  # noqa: F401
from matplotlib.colors import ListedColormap, Normalize
from palettable import colorbrewer
from parameter_sweep_analysis import (
    get_contiguous_wbds,
    load_model_file,
    load_model_results,
    setup_map,
)
from scipy.stats import pearsonr, zscore
from utils.config import config
from utils.io import load_feather, load_pickle, write_feather, write_pickle
from utils.metrics import (
    get_alpha_nse,
    get_beta_nse,
    get_entropy,
    get_nnse,
    get_nrmse,
)
from utils.plot_tools import (  # VAR_ORDER,
    custom_bar_chart,
    determine_grid_size,
    get_pretty_var_name,
    get_tick_years,
)
from utils.utils import format_equation

# plt.rcParams["svg.fonttype"] = "none"

CPUS = cpu_count()
os.environ["OMP_NUM_THREADS"] = str(CPUS)

OP_GROUPS = {
    "Very Small": [1],
    "Small, Low RT": [2],
    "Small, Mid RT": [3, 4, 5],
    "Small, High RT": [6],
    "Medium, Low RT": [7],
    "Medium, Mid RT": [9, 13, 14, 16, 17],
    "Medium, High RT": [10, 13, 14, 16, 17],
    "Medium-Large": [13, 14, 16, 17],
    "Large, Low RT": [8],
    "Large 1": [12, 13, 14, 16, 17],
    "Large 2": [11, 13, 14, 16, 17],
    "Large 3": [11, 12, 13, 14, 16, 17],
    "Very Large": [11, 12, 13, 15, 16, 17],
}

OP_GROUP_FINAL_KEYS = [
    "Very Small",
    "Small, Low RT",
    "Small, Mid RT",
    "Small, High RT",
    "Medium, Low RT",
    "Medium, Mid RT",
    "Medium, High RT",
    "Large, Low RT",
    "Large",
    "Very Large",
]

OP_GROUP_PLOT_INFO = {
    "Very Small": {"marker": "o"},
    "Small, Low RT": {"marker": "o"},
    "Small, Mid RT": {"marker": "v"},
    "Small, High RT": {"marker": "o"},
    "Medium, Low RT": {"marker": "o"},
    "Medium, Mid RT": {"marker": "v"},
    "Medium, High RT": {"marker": "v"},
    "Large, Low RT": {"marker": "o"},
    "Large": {"marker": "v"},
    "Very Large": {"marker": "v"},
}

TIME_VARYING_GROUPS = [
    "Small, Mid RT",
    "Medium, Mid RT",
    "Medium, High RT",
    "Large",
    "Very Large",
]


def find_operational_groups_for_res(model, model_data):
    _, train_groups = get_params_and_groups(model_data["train"], model)
    _, test_groups = get_params_and_groups(model_data["test"], model)
    # get the unique final leaves
    groups_uniq = train_groups.unique()
    # sorts them in ascending order
    groups_uniq.sort()
    # maps them to their sorted index + 1
    group_map = {j: i + 1 for i, j in enumerate(groups_uniq)}
    train_groups = train_groups.apply(group_map.get)
    test_groups = test_groups.apply(group_map.get)

    groups = pd.concat([train_groups, test_groups])
    groups = groups.groupby("res_id").unique()

    group_names = OP_GROUPS.keys()
    res_op_groups = defaultdict(list)
    for res, rgroups in groups.items():
        found_match = False
        for ogroup, nodes in OP_GROUPS.items():
            if sorted(nodes) == sorted(rgroups):
                res_op_groups[ogroup].append(res)
                found_match = True
                break
        if not found_match:
            print(f"No match for {res}")
            print(f"Nodes = {sorted(rgroups)}")
            for i, ogroup in enumerate(group_names):
                print(f"[{i + 1}]: {ogroup}")
            resp = input("Enter operational group: ")
            resp_group = group_names[int(resp) - 1]
            res_op_groups[resp_group].append(res)
    res_op_groups["Medium, High RT"].extend(res_op_groups["Medium-Large"])
    res_op_groups.pop("Medium-Large")
    write_pickle(
        res_op_groups,
        config.get_dir("agg_results") / "best_model_op_groups.pickle",
    )
    records = []
    for group, resers in res_op_groups.items():
        records.extend([(group, r) for r in resers])

    records = pd.DataFrame.from_records(records, columns=["op_group", "res_id"])
    write_feather(
        records,
        config.get_dir("agg_results") / "best_model_op_groups.feather",
    )


def plot_operational_group_map(model_results):
    res_op_groups = load_pickle(
        config.get_dir("agg_results") / "best_model_op_groups.pickle",
    )
    res_op_groups["Large"] = [
        *res_op_groups["Large 1"],
        *res_op_groups["Large 2"],
        *res_op_groups["Large 3"],
    ]
    res_op_groups.pop("Large 1")
    res_op_groups.pop("Large 2")
    res_op_groups.pop("Large 3")

    # train_res = model_results["train_data"].index.get_level_values("res_id").unique()
    # test_res = model_results["test_data"].index.get_level_values("res_id").unique()

    groups = OP_GROUP_FINAL_KEYS

    norm = Normalize(vmin=0, vmax=len(groups) - 1)
    cmap = get_cmap("Paired")
    group_colors = {g: cmap(norm(i)) for i, g in enumerate(groups)}

    wbds = get_contiguous_wbds()
    other_bounds = [(b, "k", "w") for b in wbds]

    west, south, east, north = (
        -127.441406,
        24.207069,
        -66.093750,
        53.382373,
    )
    m = setup_map(coords=(west, south, east, north), other_bound=other_bounds)

    grand = gpd.read_file(config.get_dir("spatial_data") / "my_grand_info")

    for group in groups:
        resers = res_op_groups[group]
        marker = OP_GROUP_PLOT_INFO[group]["marker"]
        res_coords = [
            (row.LONG_DD, row.LAT_DD)
            for _, row in grand[grand["GRAND_ID"].isin(resers)].iterrows()
        ]
        res_x, res_y = list(zip(*res_coords))
        m.scatter(
            res_x,
            res_y,
            latlon=True,
            marker=marker,
            color=group_colors[group],
            label=group,
            zorder=4,
            sizes=[50],
        )
    ax = plt.gca()
    ax.legend(loc="lower left", ncol=3, prop={"size": 10})
    plt.show()


def get_coefficient_dataframe(model, model_data):
    params, groups = get_params_and_groups(model_data["train"], model)
    feats = [model.feats[i] for i in model.reg_vars]
    # get the unique final leaves
    groups_uniq = groups.unique()
    # sorts them in ascending order
    groups_uniq.sort()
    # maps them to their sorted index + 1
    group_map = {j: i + 1 for i, j in enumerate(groups_uniq)}
    groups = groups.apply(group_map.get)
    param_df = pd.DataFrame.from_records(
        params,
        columns=feats,
        index=groups.index,
    )
    param_df["group"] = groups
    out_param_df = param_df.groupby("group").mean()
    print(out_param_df.to_markdown(floatfmt=".3f"))
    param_vars = {
        "const": r"\beta_0",
        "storage_pre": r"S_{t-1}",
        "release_pre": r"D_{t-1}",
        "inflow": r"NI_{t}",
        "sto_diff": r"S_{t-1} - \bar{S}^7_{t-1}",
        "release_roll7": r"\bar{D}^7_{t-1}",
        "inflow_roll7": r"\bar{NI}^7_{t-1}",
        "storage_x_inflow": r"S_{t-1} \times NI_{t}",
        "inflow2": r"NI_{t}^2",
        "release_pre2": r"D_{t-1}^2",
    }

    equations = []
    for group, row in out_param_df.iterrows():
        coefs = {var: f"{i:.3f}" for var, i in row.items()}
        equations.append(format_equation(param_vars, coefs))

    with open(config.get_dir("agg_results") / "equations.tex", "w") as f:
        f.write(r"\begin{document}" + "\n\n")

        for eq in equations:
            f.write(r"\[{}\]".format(eq) + "\n")

        f.write("\n\n" + r"\end{document}")

    from IPython import embed as II

    II()


def get_basin_op_mode_breakdown():
    res_op_groups = load_feather(
        config.get_dir("agg_results") / "best_model_op_groups.feather",
    )
    res_huc2 = load_feather(
        config.get_dir("spatial_data") / "updated_res_huc2.feather",
    )
    res_huc2["huc2_id"] = [f"{i:02d}" for i in res_huc2["huc2_id"]]
    res_huc2 = res_huc2.set_index("res_id")
    res_huc2 = res_huc2.loc[res_op_groups["res_id"]]

    huc2_op_groups = {}
    for huc2 in sorted(res_huc2["huc2_id"].unique()):
        resers = res_huc2[res_huc2["huc2_id"] == huc2].index
        groups = res_op_groups[res_op_groups["res_id"].isin(resers)]
        huc2_op_groups[huc2] = groups["op_group"].value_counts()

    from IPython import embed as II

    II()


def get_all_res_groups(model, model_data):
    _, train_groups = get_params_and_groups(model_data["train"], model)
    _, test_groups = get_params_and_groups(model_data["test"], model)
    # get the unique final leaves
    groups_uniq = train_groups.unique()
    # sorts them in ascending order
    groups_uniq.sort()
    # maps them to their sorted index + 1
    group_map = {j: i + 1 for i, j in enumerate(groups_uniq)}
    train_groups = train_groups.apply(group_map.get)
    test_groups = test_groups.apply(group_map.get)

    return pd.concat([train_groups, test_groups])


def get_res_seasonal_operations(model, model_data, op_group):
    groups = get_all_res_groups(model, model_data)
    res_op_groups = load_feather(
        config.get_dir("agg_results") / "best_model_op_groups.feather",
    )

    res_op_groups["op_group"] = res_op_groups["op_group"].replace(
        {"Large 1": "Large", "Large 2": "Large", "Large 3": "Large"},
    )

    resers = res_op_groups[res_op_groups["op_group"] == op_group]["res_id"]
    groups = groups.loc[pd.IndexSlice[resers, :]]
    counts = (
        groups.groupby(["res_id", groups.index.get_level_values("date").month])
        .value_counts()
        .unstack()
    )
    counts = counts.fillna(0.0)
    props = counts.divide(counts.sum(axis=1), axis=0)
    props = props.stack().reset_index().rename(columns={"level_2": "group", 0: "prop"})
    return props


def plot_seasonal_operations(model, model_data, polar=False):
    large = get_res_seasonal_operations(model, model_data, "Large")
    very_large = get_res_seasonal_operations(model, model_data, "Very Large")
    med_mid_rt = get_res_seasonal_operations(
        model,
        model_data,
        "Medium, Mid RT",
    )
    med_high_rt = get_res_seasonal_operations(
        model,
        model_data,
        "Medium, High RT",
    )
    sml_mid_rt = get_res_seasonal_operations(model, model_data, "Small, Mid RT")

    fig, axes = plt.subplots(
        2,
        3,
        sharex=False,
        sharey=True,
        figsize=(16, 10),
        subplot_kw={"projection": "polar"} if polar else None,
    )
    axes = axes.flatten()

    plot_args = zip(
        [sml_mid_rt, med_mid_rt, med_high_rt, large, very_large],
        [
            "Small, Mid RT",
            "Medium, Mid RT",
            "Medium, High RT",
            "Large",
            "Very Large",
        ],
        axes[:5],
    )
    all_groups = [3, 4, 5, 9, 10, 11, 12, 13, 14, 15, 16, 17]
    color_palette = colorbrewer.qualitative.Paired_12.mpl_colors
    group_colors = {j: color_palette[i] for i, j in enumerate(all_groups)}

    for df, title, ax in plot_args:
        prop_df = df.groupby(["date", "group"])["prop"]
        prop = prop_df.mean().unstack()
        # q1 = prop_df.quantile(0.25).unstack()
        # q3 = prop_df.quantile(0.75).unstack()
        # error = {
        #     g: np.vstack([q1[g], q3[g]]) for g in prop.columns
        # }
        if not polar:
            custom_bar_chart(
                prop,
                width=0.85,
                ax=ax,
                colors=group_colors,
                edgecolor="k",
            )
            ax.set_title(title)
            ax.set_xticks(range(0, 12))
            ax.set_xticklabels(calendar.month_abbr[1:], rotation=90)
            ax.tick_params(
                axis="both",
                which="minor",
                top=False,
                right=False,
                bottom=False,
                left=False,
            )
            ax.tick_params(
                axis="both",
                which="major",
                top=False,
                right=False,
                bottom=False,
            )
            ax.spines.right.set_visible(False)
            ax.spines.top.set_visible(False)
        else:
            for group in prop.columns:
                ax.plot(
                    np.deg2rad(np.arange(0, 360, 30)),
                    prop[group].values,
                    color=group_colors[group],
                    label=group,
                )

    fig.text(
        0.02,
        0.5,
        "Average Operational Mode Occurence Proportion",
        va="center",
        rotation="vertical",
    )
    handles = [mpatch.Patch(color=group_colors[i]) for i in all_groups]
    axes[-1].axis("off")
    axes[-1].legend(
        handles,
        all_groups,
        ncol=4,
        title="Operational Mode",
        loc="center",
        frameon=True,
    )
    plt.subplots_adjust(
        top=0.90,
        bottom=0.1,
        left=0.068,
        right=0.989,
        hspace=0.318,
        wspace=0.051,
    )
    plt.savefig(
        "/home/lucas/Dropbox/plrt-conus-figures/good_figures/op_group_analysis/"
        "seasonal_bar_charts/all_groups_seasonal_percentages.png",
        dpi=450,
        bbox_inches="tight",
    )
    plt.show()


def plot_basin_specific_seasonal_operations(model, model_data, op_group):
    props = get_res_seasonal_operations(model, model_data, op_group)
    resers = props["res_id"].unique()
    res_huc2 = load_feather(
        config.get_dir("spatial_data") / "updated_res_huc2.feather",
    )
    res_huc2["huc2_id"] = [f"{i:02d}" for i in res_huc2["huc2_id"]]
    res_huc2 = res_huc2.set_index("res_id")
    res_huc2 = res_huc2.loc[resers]

    props["basin"] = res_huc2.loc[props["res_id"]].values
    res_per_basin = props.groupby("basin")["res_id"].unique().apply(lambda x: x.size)

    props = props[props["basin"].isin(res_per_basin[res_per_basin > 1].index)]
    basin_props = props.groupby(["date", "group", "basin"])["prop"].mean().unstack()

    basins = basin_props.columns
    nbasins = len(basins)

    gs = determine_grid_size(nbasins)
    fig, axes = plt.subplots(*gs, sharex=True, sharey=True, figsize=(12, 8))
    huc2_names = pd.read_csv(
        config.get_dir("spatial_data") / "huc2_names.csv",
        index_col=0,
        names=["name"],
    )
    legend = True
    for basin, ax in zip(basins, axes.flatten()):
        pdf = basin_props[basin].unstack()
        pdf.plot.bar(ax=ax, stacked=True, legend=False, width=0.8)
        basin_name = huc2_names.loc[int(basin), "name"]
        res_count = res_per_basin.loc[basin]
        ax.set_title(f"{basin}-{basin_name} (# res: {res_count})")
        ax.set_ylim(0, 1)
        ax.set_ylabel("Occurence Probability")
        ax.set_xlabel("")
        ax.set_xticks(range(0, 12))
        ax.set_xticklabels(calendar.month_abbr[1:], rotation=90)
        if legend:
            ax.legend(loc="lower left", ncol=2, prop={"size": 12})
            legend = False
    fig.suptitle(f"Op. Group: {op_group}")

    left_over = len(axes.flatten()) - nbasins
    if left_over > 0:
        for ax in axes.flatten()[-left_over:]:
            ax.axis("off")
    plt.show()


def plot_reservoir_most_likely_group_maps(model, model_data, op_group):
    props = get_res_seasonal_operations(model, model_data, op_group)
    wbds = get_contiguous_wbds()
    wbd_ids = [re.search(r"WBD_(\d\d)_HU2", i).group(1) for i in wbds]
    wbd_map = {int(i): wbd for i, wbd in zip(wbd_ids, wbds)}

    resers = props["res_id"].unique()
    groups = props["group"].unique()

    res_huc2 = load_feather(
        config.get_dir("spatial_data") / "updated_res_huc2.feather",
    )
    res_huc2["huc2_id"] = [f"{i:02d}" for i in res_huc2["huc2_id"]]
    res_huc2 = res_huc2.set_index("res_id")
    res_huc2 = res_huc2.loc[resers]

    most_likely_groups = []
    for res in resers:
        rdf = props[props["res_id"] == res]
        rdf = rdf.pivot(index="group", columns="date", values="prop")
        rdf = rdf.fillna(0.0)
        most_likely_groups.append([res, *rdf.idxmax(axis=0).values])

    most_likely_groups = pd.DataFrame.from_records(
        most_likely_groups,
        columns=["res", *range(1, 13)],
    )
    most_likely_groups = most_likely_groups.set_index("res")
    most_likely_groups["basin"] = res_huc2

    most_likely_basin_groups = []
    for basin in most_likely_groups["basin"].unique():
        bdf = most_likely_groups[most_likely_groups["basin"] == basin]
        mode = bdf.mode(axis=0)
        most_likely_basin_groups.append([basin, *mode[range(1, 13)].values[0]])

    most_likely_basin_groups = pd.DataFrame.from_records(
        most_likely_basin_groups,
        columns=["basin", *range(1, 13)],
    )

    color_pal = sns.color_palette("Paired")
    norm = Normalize(vmin=0, vmax=len(groups) - 1)
    cmap = ListedColormap(color_pal.as_hex()[: len(groups)])
    group_colors = {g: cmap(norm(i)) for i, g in enumerate(sorted(groups))}

    most_likely_basin_groups = most_likely_basin_groups.set_index("basin")
    basin_month_colors = {}

    for basin in most_likely_basin_groups.index:
        bgroups = most_likely_basin_groups.loc[basin]
        basin_month_colors[basin] = [group_colors[int(i)] for i in bgroups.values]

    other_bounds = []
    for month in range(12):
        temp_bounds = []
        for basin in res_huc2["huc2_id"].unique():
            bwbd = wbd_map[int(basin)]
            color = basin_month_colors[basin][month]
            temp_bounds.append((bwbd, "k", color))
        other_bounds.append(temp_bounds)

    west, south, east, north = (
        -127.441406,
        24.207069,
        -66.093750,
        53.382373,
    )

    fig, axes = plt.subplots(3, 4, sharex=True, sharey=True)
    axes = axes.flatten()

    # l, r, t, b
    label_positions = [
        [1, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [1, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [1, 0, 0, 1],
        [0, 0, 0, 1],
        [0, 0, 0, 1],
        [0, 0, 0, 1],
    ]
    maps = [
        setup_map(
            coords=[west, south, east, north],
            other_bound=ob,
            ax=ax,
            label_positions=lp,
            return_ticks=True,
        )
        for ax, lp, ob in zip(axes, label_positions, other_bounds)
    ]

    # res_coords = [
    #     (row.LONG_DD, row.LAT_DD)
    #     for i, row in grand[grand["GRAND_ID"].isin(resers)].iterrows()
    # ]
    # res_x, res_y = list(zip(*res_coords))

    # for month, map_info in zip(range(1, 13), maps):
    #     m = map_info[0]
    #     m.scatter(
    #         res_x,
    #         res_y,
    #         latlon=True,
    #         marker="v",
    #         color="k",
    #         # label=group,
    #         zorder=4,
    #         sizes=[40],
    #     )

    for i, ax in enumerate(axes):
        ax.set_title(calendar.month_name[i + 1])

    mvals, pvals = maps[8][1:]
    xticks = [i[1][0].get_position()[0] for i in mvals.values() if i[1]]
    yticks = []
    for i in pvals.values():
        try:
            yticks.append(i[1][0].get_position()[1])
        except IndexError:
            pass

    for ax in axes:
        ax.set_xticks(xticks)
        ax.set_yticks(yticks)
        ax.tick_params(
            axis="both",
            direction="in",
            left=True,
            right=True,
            top=True,
            bottom=True,
            labelleft=False,
            labelright=False,
            labeltop=False,
            labelbottom=False,
            zorder=10,
        )
    handles = [mpatch.Patch(edgecolor="k", facecolor=c) for c in group_colors.values()]
    axes[0].legend(
        handles,
        [f"Group {i}" for i in group_colors.keys()],
        loc="upper right",
        prop={"size": 10},
    )
    fig.suptitle(f"Op. Group: {op_group}")
    plt.show()


def plot_basin_group_makeup():
    op_groups = load_feather(
        config.get_dir("agg_results") / "best_model_op_groups.feather",
        index_keys=["index"],
    ).set_index("res_id")
    op_groups["op_group"] = op_groups["op_group"].replace(
        {"Large 1": "Large", "Large 2": "Large", "Large 3": "Large"},
    )
    res_huc2 = load_feather(
        config.get_dir("spatial_data") / "updated_res_huc2.feather",
        index_keys=["res_id"],
    )
    op_groups["basin"] = res_huc2

    counts = op_groups.groupby("basin")["op_group"].value_counts().unstack().fillna(0)
    props = counts.divide(counts.sum(axis=1), axis=0)
    props.index = props.index.astype(int)
    props = props.sort_index()
    props = props[OP_GROUP_FINAL_KEYS]
    ax = props.plot.bar(stacked=True, width=0.8, edgecolor="k")
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax.set_xticklabels([f"{i:02d}" for i in props.index], rotation=0)
    ax.set_xlabel("HUC2")
    ax.set_ylabel("Op. Group Proportion")
    # ax.get_legend().set_title("Op. Group")
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, ncol=5, loc="upper left")
    ax.set_ylim(0, 1.09)
    plt.show()


def plot_res_group_colored_timeseries(results, model, model_data, res=None):
    from fit_plrt_model import load_resopsus_data

    data, meta = load_resopsus_data(min_years=3)
    groups = get_all_res_groups(model, model_data)

    df = data.loc[:, ["storage", "inflow", "release"]]
    df["groups"] = groups
    df["modeled_release"] = pd.concat(
        [results["train_data"]["model"], results["test_data"]["model"]],
    )

    grand = gpd.read_file(config.get_dir("spatial_data") / "my_grand_info").set_index(
        "GRAND_ID",
    )

    res_huc2 = load_feather(
        config.get_dir("spatial_data") / "updated_res_huc2.feather",
        index_keys=["res_id"],
    )
    huc2_names = pd.read_csv(
        config.get_dir("spatial_data") / "huc2_names.csv",
        header=None,
        names=["huc2_id", "huc2_name"],
    ).set_index("huc2_id")

    style_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    op_groups = load_pickle(
        config.get_dir("agg_results") / "best_model_op_groups.pickle",
    )
    op_groups["Large"] = [
        *op_groups["Large 1"],
        *op_groups["Large 2"],
        *op_groups["Large 3"],
    ]
    for i in range(1, 4):
        del op_groups[f"Large {i}"]

    parallel = True

    iterator = []
    for op_group in TIME_VARYING_GROUPS:
        for res in op_groups[op_group]:
            iterator.append((op_group, res))

    if parallel:
        Parallel(n_jobs=-1, verbose=11)(
            delayed(parallel_body_colored_group_plots)(
                df,
                res,
                grand.loc[res, "DAM_NAME"],
                res_huc2.loc[res, "huc2_id"],
                huc2_names.loc[int(res_huc2.loc[res, "huc2_id"]), "huc2_name"],
                style_colors,
                op_group,
                save=True,
            )
            for op_group, res in iterator
        )
    else:
        for op_group, res in iterator:
            huc2 = res_huc2.loc[res, "huc2_id"]
            parallel_body_colored_group_plots(
                df,
                res,
                grand.loc[res, "DAM_NAME"],
                huc2,
                huc2_names.loc[int(huc2), "huc2_name"],
                style_colors,
                op_group,
                show=True,
                save=True,
            )


def parallel_body_colored_group_plots(
    df,
    res,
    print_res,
    basin,
    print_basin,
    style_colors,
    op_group,
    show=False,
    save=True,
):
    pdf = df.loc[pd.IndexSlice[res, :]]
    rgroups = sorted(pdf["groups"].unique())
    if len(rgroups) == 1:
        return

    plot_resid = False
    if plot_resid:
        fig, axes = plt.subplots(4, 1, sharex=True, figsize=(19, 10))
        axes = axes.flatten()
    else:
        fig, axes = plt.subplots(3, 1, sharex=True, figsize=(19, 10))
        axes = axes.flatten()

    pdf = pdf.reset_index()

    for var in ["storage", "inflow"]:
        inf_scores = zscore(pdf[var])
        pdf.loc[inf_scores.abs() > 3, var] = np.nan
        pdf[var] = pdf[var].interpolate()

    pdf["residual"] = pdf["modeled_release"] - pdf["release"]

    colors = [style_colors[rgroups.index(i)] for i in pdf["groups"]]
    axes[0].scatter(pdf.index, pdf["storage"], c=colors, s=10)
    axes[1].scatter(pdf.index, pdf["inflow"], c=colors, s=10)
    axes[2].scatter(pdf.index, pdf["release"], c=colors, s=10)
    if plot_resid:
        axes[3].scatter(
            pdf.index,
            pdf["modeled_release"] - pdf["release"],
            c=colors,
            s=10,
        )

    axes[0].set_ylabel("Storage [TAF]")
    axes[1].set_ylabel("Inflow [TAF/day]")
    axes[2].set_ylabel("Release [TAF/day]")
    bottom_ax = axes[2]
    if plot_resid:
        axes[3].set_ylabel("Residual [TAF/day]")
        bottom_ax = axes[3]

    index = pdf["date"]
    tick_years, ticks = get_tick_years(index, bottom_ax)
    tick_labels = [str(i) for i in tick_years]

    bottom_ax.set_xticks(ticks)
    bottom_ax.set_xticklabels(tick_labels)
    handles = [
        plt.scatter([], [], color=style_colors[i], alpha=1) for i in range(len(rgroups))
    ]
    labels = [int(i) for i in rgroups]
    bottom_ax.legend(handles, labels, loc="upper right")
    fig.align_ylabels()
    fig.suptitle(f"{print_res} - {print_basin}")
    plt.subplots_adjust(
        top=0.945,
        bottom=0.045,
        left=0.045,
        right=0.991,
        hspace=0.126,
        wspace=0.2,
    )

    if save:
        file_res = "_".join(print_res.lower().split())
        odir = os.path.expanduser(
            "~/Dropbox/plrt-conus-figures/good_figures/"
            f"group_colored_timeseries/{op_group}",
        )
        if not os.path.exists(odir):
            os.makedirs(odir)
        plt.savefig(
            os.path.join(odir, f"{basin}_{file_res}.png"),
            dpi=450,
        )
    if show:
        plt.show()
    plt.close()


def load_op_groups():
    op_groups = load_feather(
        config.get_dir("agg_results") / "best_model_op_groups.feather",
        index_keys=["index"],
    )
    op_groups["op_group"] = op_groups["op_group"].replace(
        {"Large 1": "Large", "Large 2": "Large", "Large 3": "Large"},
    )
    return op_groups


def plot_basin_group_entropy(
    model,
    model_data,
    op_group="all",
    scale=True,
    plot_res=False,
):
    groups = get_all_res_groups(model, model_data)
    op_groups = load_op_groups()
    if op_group != "all":
        resers = op_groups.loc[op_groups["op_group"] == op_group, "res_id"]
    else:
        resers = op_groups["res_id"]
    groups = groups.loc[pd.IndexSlice[resers, :]]

    res_huc2 = load_feather(
        config.get_dir("spatial_data") / "updated_res_huc2.feather",
    )
    res_huc2 = res_huc2.set_index("res_id")

    # get name of wbd files
    wbds = get_contiguous_wbds()
    wbd_ids = [re.search(r"WBD_(\d\d)_HU2", i).group(1) for i in wbds]
    wbd_map = {int(i): wbd for i, wbd in zip(wbd_ids, wbds)}

    # get grand database
    grand = gpd.read_file(config.get_dir("spatial_data") / "my_grand_info")

    scores = get_entropy(groups, "res_id", scale=scale)

    resers = scores.index
    res_huc2 = res_huc2.loc[resers]
    res_huc2["entropy"] = scores

    print(
        (
            res_huc2.groupby("huc2_id").max() - res_huc2.groupby("huc2_id").min()
        ).sort_values(by="entropy", ascending=False),
    )
    from IPython import embed as II

    II()
    return
    scores = res_huc2.groupby("huc2_id").mean()

    # max_score = scores["entropy"].max()
    # min_score = scores["entropy"].min()
    # print(min_score, max_score)
    # * doing the above for each group gives the following full bounds
    if scale:
        # * if scaling by log2(k)
        min_score = 0.23073330900603095
        max_score = 0.9210221060603138
    else:
        min_score = 0.4134185912778788
        max_score = 1.6364994773122774

    score_range = max_score - min_score

    norm = Normalize(
        vmin=min_score - score_range * 0.05,
        vmax=max_score + score_range * 0.05,
    )
    cmap = get_cmap("viridis")
    other_bounds = []
    for wbd_id, score in scores["entropy"].items():
        wbd = wbd_map[int(wbd_id)]
        color = cmap(norm(score))
        other_bounds.append((wbd, "k", color))

    west, south, east, north = (
        -127.441406,
        24.207069,
        -66.093750,
        53.382373,
    )

    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    ax.tick_params(
        axis="both",
        which="minor",
        left=False,
        bottom=False,
        top=False,
        right=False,
    )
    m = setup_map(
        coords=[west, south, east, north],
        other_bound=other_bounds,
        ax=ax,
    )
    maps = [m]

    if plot_res:
        res_coords = [
            (row.LONG_DD, row.LAT_DD)
            for i, row in grand[grand["GRAND_ID"].isin(resers)].iterrows()
        ]
        res_x, res_y = list(zip(*res_coords))

        for m in maps:
            m.scatter(
                res_x,
                res_y,
                latlon=True,
                marker="v",
                color="r",
                zorder=4,
                sizes=res_huc2.loc[resers, "entropy"].values * 50,
            )
    x, y = maps[0](-80, 51)
    ax.text(x, y, op_group, fontsize=24, ha="center", va="center")

    # if scale:
    #     label = r"Mean Operational Mode Entropy [%]"
    # else:
    #     label = "Mean Operational Mode Entropy  [bits]"
    # cbar_fig, cbar_ax = plt.subplots(1, 1)
    # plt.colorbar(
    #     ScalarMappable(norm=norm, cmap=cmap),
    #     cax=cbar_ax,
    #     orientation="horizontal",
    #     label=label,
    #     aspect=4,
    #     shrink=0.8,
    #     format=lambda x, _: f"{x:.0%}"
    # )
    # plt.show()
    file_name = {
        "Small, Mid RT": "small_mid_rt.png",
        "Medium, Mid RT": "medium_mid_rt.png",
        "Medium, High RT": "medium_high_rt.png",
        "Large": "large.png",
        "Very Large": "very_large.png",
    }[op_group]
    outdir = os.path.expanduser(
        "~/Dropbox/plrt-conus-figures/good_figures/op_group_analysis/"
        "op_group_entropy_maps",
    )
    if scale:
        file_name = "scaled_" + file_name
        output_file = f"{outdir}/scaled/{file_name}"
    else:
        output_file = f"{outdir}/raw/{file_name}"

    plt.savefig(output_file, dpi=450, bbox_inches="tight")


def plot_training_vs_testing_simul_perf(model_results, ax=None):
    opt_model_results = load_model_results(
        config.get_dir("results")
        / "monthly_merged_data_set_minyr3"
        / "TD6_MSS0.03_SM_basin_0.8",
    )
    opt_simmed = opt_model_results["simmed_data"]

    train_reservoirs = (
        model_results["train_data"].index.get_level_values("res_id").unique()
    )
    test_reservoirs = (
        model_results["test_data"].index.get_level_values("res_id").unique()
    )
    simmed_data = model_results["simmed_data"]

    opt_nnse = get_nnse(opt_simmed, "actual", "model", "res_id")
    opt_nrmse = get_nrmse(opt_simmed, "actual", "model", "res_id", "range")

    nnse = get_nnse(simmed_data, "actual", "model", "res_id")
    nrmse = get_nrmse(simmed_data, "actual", "model", "res_id", "range")

    nnse -= opt_nnse
    nrmse -= opt_nrmse

    dset = pd.Series("train", index=train_reservoirs)
    dset = pd.concat([dset, pd.Series("test", index=test_reservoirs)])
    df = pd.DataFrame.from_dict({"nNSE": nnse, "nRMSE": nrmse, "Dataset": dset})
    df = df.fillna("test")

    if ax:
        show = False
    else:
        show = True
        ax = plt.gca()

    train_df = df[df["Dataset"] == "train"]
    test_df = df[df["Dataset"] == "test"]

    ax.scatter(
        train_df["nNSE"],
        train_df["nRMSE"],
        label="Training Reservoir",
        edgecolor="k",
        linewidths=0.5,
    )
    ax.scatter(
        test_df["nNSE"],
        test_df["nRMSE"],
        label="Testing Reservoir",
        edgecolor="k",
        marker="X",
        linewidths=0.5,
        zorder=10,
    )
    ax.set_xlabel(r"$\Delta$ nNSE")
    ax.set_ylabel(r"$\Delta$ nRMSE")
    # ax.axhline(train_df["nRMSE"].mean())
    # ax.axhline(test_df["nRMSE"].mean())
    # ax.axvline(train_df["nNSE"].mean())
    # ax.axvline(test_df["nNSE"].mean())
    ax.axhline(0, color="k", linestyle="--")
    ax.axvline(0, color="k", linestyle="--")

    # ax = sns.scatterplot(
    #     data=df,
    #     x="nNSE",
    #     y="nRMSE",
    #     hue="Dataset",
    #     style="Dataset",
    #     hue_order=["train", "test"],
    #     style_order=["train", "test"],
    #     legend="full",
    #     edgecolor="k",
    #     **kwargs,
    # )

    ax.legend(loc="best")
    if show:
        plt.show()


def plot_experimental_dset_sim_perf():
    model_paths = [
        "TD6_MSS0.03_SM_meta_rts_0.8",
        "TD6_MSS0.03_SM_meta_rts_-0.2",
        "TD6_MSS0.03_SM_meta_max_sto_0.8",
        "TD6_MSS0.03_SM_meta_max_sto_-0.2",
        "TD6_MSS0.03_SM_meta_rel_inf_corr_0.8",
        "TD6_MSS0.03_SM_meta_rel_inf_corr_-0.2",
    ]

    titles = [
        r"Upper 20\% $RT$",
        r"Lower 20\% $RT$",
        r"Upper 20\% $S_{max}$",
        r"Lower 20\% $S_{max}$",
        r"Upper 20\% $r(D_t, NI_t)$",
        r"Lower 20\% $r(D_t, NI_t)$",
    ]
    width = 12
    height = 10
    fig, axes = plt.subplots(
        3,
        2,
        sharex=True,
        sharey=True,
        figsize=(width, height),
    )
    label_args = [
        {"label_x": False, "label_y": True, "legend": True},
        {"label_x": False, "label_y": False, "legend": False},
        {"label_x": False, "label_y": True, "legend": False},
        {"label_x": False, "label_y": False, "legend": False},
        {"label_x": True, "label_y": True, "legend": False},
        {"label_x": True, "label_y": False, "legend": False},
    ]

    plot_iterator = zip(model_paths, axes.flatten(), titles, label_args)
    for model_path, ax, title, label_arg in plot_iterator:
        full_model_path = (
            config.get_dir("results") / "monthly_merged_data_set_minyr3" / model_path
        )
        model_results = load_model_results(full_model_path)
        plot_training_vs_testing_simul_perf(model_results, ax=ax)
        if not label_arg["label_x"]:
            ax.set_xlabel("")
        if not label_arg["label_y"]:
            ax.set_ylabel("")
        if not label_arg["legend"]:
            ax.get_legend().remove()
        ax.set_title(title)
    fig.align_xlabels()
    fig.align_ylabels()
    plt.subplots_adjust(
        top=0.7,
        bottom=0.11,
        left=0.2,
        right=0.8,
        hspace=0.15,
        wspace=0.05,
    )
    # plt.savefig(
    #     os.path.expanduser(
    #         "~/Dropbox/plrt-conus-figures/good_figures/experimental_result/"
    #         "nnse_vs_nrmse_diff.svg",
    #     ),
    #     format="svg",
    #     dpi=1200,
    #     bbox_inches="tight",
    # )
    plt.show()


def plot_experimental_dset_nse_decomp():
    model_paths = [
        "TD6_MSS0.03_SM_meta_rts_0.8",
        "TD6_MSS0.03_SM_meta_rts_-0.2",
        "TD6_MSS0.03_SM_meta_max_sto_0.8",
        "TD6_MSS0.03_SM_meta_max_sto_-0.2",
        "TD6_MSS0.03_SM_meta_rel_inf_corr_0.8",
        "TD6_MSS0.03_SM_meta_rel_inf_corr_-0.2",
    ]

    titles = [
        r"Upper 20\% $RT$",
        r"Lower 20\% $RT$",
        r"Upper 20\% $S_{max}$",
        r"Lower 20\% $S_{max}$",
        r"Upper 20\% $r(D_t, NI_t)$",
        r"Lower 20\% $r(D_t, NI_t)$",
    ]
    width = 16
    height = 10
    fig, axes = plt.subplots(
        3,
        6,
        sharex="col",
        sharey=True,
        figsize=(width, height),
    )
    label_args = [
        {"label_x": False, "label_y": True, "legend": True},
        {"label_x": False, "label_y": False, "legend": False},
        {"label_x": False, "label_y": True, "legend": False},
        {"label_x": False, "label_y": False, "legend": False},
        {"label_x": True, "label_y": True, "legend": False},
        {"label_x": True, "label_y": False, "legend": False},
    ]
    flat_axes = axes.flatten()
    axes_iterator = [flat_axes[i : i + 3] for i in range(0, len(flat_axes), 3)]

    plot_iterator = zip(model_paths, axes_iterator, titles, label_args)
    for model_path, axes, title, label_arg in plot_iterator:
        full_model_path = (
            config.get_dir("results") / "monthly_merged_data_set_minyr3" / model_path
        )
        model_results = load_model_results(full_model_path)
        plot_nse_decomp_experiment(model_results, axes=axes)
        if not label_arg["label_x"]:
            for ax in axes:
                ax.set_xlabel("")
        if not label_arg["label_y"]:
            for ax in axes:
                ax.set_ylabel("")
        else:
            for ax in axes[1:]:
                ax.set_ylabel("")
        if not label_arg["legend"]:
            for ax in axes:
                try:
                    ax.get_legend().remove()
                except AttributeError:
                    pass
        axes[1].set_title(title, pad=9)
    fig.align_xlabels()
    fig.align_ylabels()
    plt.subplots_adjust(
        top=0.958,
        bottom=0.069,
        left=0.047,
        right=0.985,
        hspace=0.15,
        wspace=0.05,
    )
    plt.savefig(
        os.path.expanduser(
            "~/Dropbox/plrt-conus-figures/good_figures/experimental_result/"
            "delta_nnse_decomp.svg",
        ),
        format="svg",
        dpi=1200,
        bbox_inches="tight",
    )
    plt.show()


def plot_nse_decomp_experiment(model_results, axes=None):
    opt_model_results = load_model_results(
        config.get_dir("results")
        / "monthly_merged_data_set_minyr3"
        / "TD6_MSS0.03_SM_basin_0.8",
    )
    opt_simmed = opt_model_results["simmed_data"]

    train_reservoirs = (
        model_results["train_data"].index.get_level_values("res_id").unique()
    )
    test_reservoirs = (
        model_results["test_data"].index.get_level_values("res_id").unique()
    )
    simmed_data = model_results["simmed_data"]

    opt_nnse = get_nnse(opt_simmed, "actual", "model", "res_id")
    opt_alpha = get_alpha_nse(opt_simmed, "actual", "model", "res_id")
    opt_beta = get_beta_nse(opt_simmed, "actual", "model", "res_id")
    opt_corr = opt_simmed.groupby("res_id").apply(
        lambda x: pearsonr(x["actual"], x["model"]).statistic,
    )

    nnse = get_nnse(simmed_data, "actual", "model", "res_id")
    alpha = get_alpha_nse(simmed_data, "actual", "model", "res_id")
    beta = get_beta_nse(simmed_data, "actual", "model", "res_id")
    corr = simmed_data.groupby("res_id").apply(
        lambda x: pearsonr(x["actual"], x["model"]).statistic,
    )

    dset = pd.Series("train", index=train_reservoirs)
    dset = pd.concat([dset, pd.Series("test", index=test_reservoirs)])
    metric_df_opt = pd.DataFrame.from_dict(
        {
            "nnse": opt_nnse,
            "alpha": opt_alpha,
            "beta": opt_beta,
            "corr": opt_corr,
            "Dataset": dset,
        },
    )
    metric_df = pd.DataFrame.from_dict(
        {
            "nnse": nnse,
            "alpha": alpha,
            "beta": beta,
            "corr": corr,
            "Dataset": dset,
        },
    )
    diff_df = pd.DataFrame.from_dict(
        {
            "nnse": nnse - opt_nnse,
            "alpha": alpha - opt_alpha,
            "beta": beta - opt_beta,
            "corr": corr - opt_corr,
            "Dataset": dset,
        },
    )
    metric_df_opt = metric_df_opt.fillna("test")
    metric_df = metric_df.fillna("test")

    train_df_opt = metric_df_opt[metric_df_opt["Dataset"] == "train"]
    test_df_opt = metric_df_opt[metric_df_opt["Dataset"] == "test"]
    train_df = metric_df[metric_df["Dataset"] == "train"]
    test_df = metric_df[metric_df["Dataset"] == "test"]
    train_diff = diff_df[diff_df["Dataset"] == "train"]
    test_diff = diff_df[diff_df["Dataset"] == "test"]
    style_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    labels = {
        "alpha": r"$\alpha$",
        "beta": r"$\beta$",
        "corr": "Pearson r",
    }

    diff = True

    for ax, variable in zip(axes, ["alpha", "beta", "corr"]):
        if diff:
            ax.scatter(
                train_diff[variable],
                train_diff["nnse"],
                label="Training Reservoir",
                c=style_colors[0],
                edgecolor="k",
                linewidths=0.5,
            )
            ax.scatter(
                test_diff[variable],
                test_diff["nnse"],
                label="Testing Reservoir",
                c=style_colors[1],
                edgecolor="k",
                marker="X",
                linewidths=0.5,
                zorder=10,
            )
            ax.axvline(0, color="k", linestyle="--")
            ax.axhline(0, color="k", linestyle="--")
            ax.set_xlabel(rf"$\Delta$ {labels[variable]}")
            ax.set_ylabel(r"$\Delta$ nNSE")
        else:
            ax.scatter(
                train_df_opt[variable],
                train_df_opt["nnse"],
                label="Training Reservoir (Opt)",
                c=style_colors[0],
                edgecolor="k",
                linewidths=0.5,
            )
            ax.scatter(
                test_df_opt[variable],
                test_df_opt["nnse"],
                label="Testing Reservoir (Opt)",
                c=style_colors[1],
                edgecolor="k",
                marker="X",
                linewidths=0.5,
                zorder=10,
            )
            ax.scatter(
                train_df[variable],
                train_df["nnse"],
                label="Training Reservoir",
                c=style_colors[0],
                edgecolor="k",
                linewidths=0.5,
            )
            ax.scatter(
                test_df[variable],
                test_df["nnse"],
                label="Testing Reservoir",
                c=style_colors[1],
                edgecolor="k",
                marker="X",
                linewidths=0.5,
                zorder=10,
            )

            ax.set_xlabel(labels[variable])
            ax.set_ylabel("nNSE")

            if variable == "alpha":
                ax.axvline(1, color="k", linestyle="--")
            if variable == "beta":
                ax.axvline(0, color="k", linestyle="--")
            ax.axhline(0.5, color="k", linestyle="--")

        if ax == axes[2]:
            ax.legend(
                loc="best",
                frameon=True,
                framealpha=1.0,
                handlelength=1,
                handletextpad=0.2,
                borderpad=0.2,
            )


def transition_probabilities(model, model_data):
    groups = get_all_res_groups(model, model_data)
    res_op_groups = load_feather(
        config.get_dir("agg_results") / "best_model_op_groups.feather",
        index_keys=["res_id"],
    ).drop("index", axis=1)

    res_op_groups = res_op_groups.replace(
        {
            "Large 1": "Large",
            "Large 2": "Large",
            "Large 3": "Large",
        },
    )
    resers = res_op_groups[res_op_groups["op_group"].isin(TIME_VARYING_GROUPS)].index
    groups = groups.loc[pd.IndexSlice[resers, :]]
    groups.name = "group"
    groups = groups.to_frame()
    groups["lagged"] = groups.groupby("res_id")["group"].shift(1)
    next_counts = groups.groupby(["res_id", "group"])["lagged"].value_counts()
    next_props = next_counts.groupby(["res_id", "group"], group_keys=False).apply(
        lambda x: x / x.sum(),
    )
    next_props.name = "t_prob"
    next_props = next_props.to_frame()
    next_props["op_group"] = [
        res_op_groups.loc[i, "op_group"] for i in next_props.index.get_level_values(0)
    ]
    t_probs = next_props.groupby(["op_group", "group", "lagged"]).mean()

    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    axes = axes.flatten()
    for group, ax in zip(TIME_VARYING_GROUPS, axes):
        sns.heatmap(
            t_probs.loc[pd.IndexSlice[group]].unstack().T,
            ax=ax,
            annot=True,
        )
        ax.set_title(group)
        ax.set_xlabel("")
        ax.set_ylabel("")

    plt.show()


def plot_coef_bar(model_path, split=False):
    coef_file = (
        config.get_dir("results")
        / "monthly_merged_data_set_minyr3"
        / model_path
        / "random_effects.csv"
    )

    coefs = pd.read_csv(coef_file, index_col=0).T

    def plot_coefs(coefs, filename=None):
        gs = determine_grid_size(coefs.shape[1])
        coefs.index = [
            get_pretty_var_name(i, math=True, lower=True) for i in coefs.index
        ]
        colors = dict(zip(coefs.index, sns.color_palette("muted")))

        plot_order = [
            get_pretty_var_name(i, math=True, lower=True) for i in coefs.index
        ]

        fig, axes = plt.subplots(*gs, sharex=True, sharey=True, figsize=(6, 8))
        if hasattr(axes, "size"):
            axes = axes.flatten()
        else:
            axes = [axes]

        for mode, ax in zip(coefs.columns, axes):
            pdf = coefs[mode]
            pdf.loc[plot_order[::-1]].plot.barh(
                ax=ax,
                color=[colors[i] for i in pdf.index],
                width=0.8,
                zorder=2,
            )
            ax.set_title(f"Mode: {mode}")
            ax.grid(True, color="k", linewidth=0.5, zorder=0)
            ax.set_xlabel("Fitted Coef.")
            ax.set_xlim((-0.2682, 1.1442))
            ax.set_ylim((-0.65, 9.65))

        for ax in axes:
            ax.tick_params(
                axis="both",
                which="minor",
                left=False,
                bottom=False,
                top=False,
                right=False,
            )
            ax.tick_params(
                axis="both",
                which="major",
                top=False,
                right=False,
                labelbottom=True,
            )

        plt.subplots_adjust(
            top=0.928,
            bottom=0.111,
            left=0.238,
            right=0.951,
            hspace=0.2,
            wspace=0.2,
        )
        show = True
        if filename:
            plt.savefig(filename, dpi=300, bbox_inches="tight")
            show = False
        patches = [mpatch.Patch(color=c) for c in sns.color_palette("muted")]

        leg_fig, leg_ax = plt.subplots(1, 1)
        leg_ax.legend(
            patches[::-1],
            [i.get_text() for i in axes[0].get_yticklabels()[::-1]],
            loc="center",
            frameon=False,
            ncol=5,
        )
        leg_ax.axis("off")
        if show:
            plt.show()
        else:
            plt.close(fig)
            plt.close(leg_fig)

    if split:
        # for op_group, modes in OP_GROUPS.items():
        #     plot_coefs(coefs[modes])
        for mode in coefs.columns:
            file = (
                "/home/lucas/Dropbox/plrt-conus-figures/good_figures/coefs_large/"
                f"mode_{mode}.png"
            )
            plot_coefs(coefs[[mode]], file)
    else:
        plot_coefs(coefs)


def calc_variable_importance(model, model_data):
    groups = get_all_res_groups(model, model_data)
    coef_file = (
        config.get_dir("results")
        / "monthly_merged_data_set_minyr3"
        / model_path
        / "random_effects.csv"
    )

    coefs = pd.read_csv(coef_file, index_col=0).T

    props = groups.value_counts() / groups.shape[0]

    importance = (coefs * props).abs().max(axis=1)
    importance = importance.sort_values(ascending=False)

    from IPython import embed as II

    II()


def plot_experimental_leave_out_ratios():
    rts = {
        "upper": [
            "TD6_MSS0.03_SM_meta_rts_0.2",
            "TD6_MSS0.03_SM_meta_rts_0.4",
            "TD6_MSS0.03_SM_meta_rts_0.6",
            "TD6_MSS0.03_SM_meta_rts_0.8",
        ],
        "lower": [
            "TD6_MSS0.03_SM_meta_rts_-0.2",
            "TD6_MSS0.03_SM_meta_rts_-0.4",
            "TD6_MSS0.03_SM_meta_rts_-0.6",
            "TD6_MSS0.03_SM_meta_rts_-0.8",
        ],
    }
    max_sto = {
        "upper": [
            "TD6_MSS0.03_SM_meta_max_sto_0.2",
            "TD6_MSS0.03_SM_meta_max_sto_0.4",
            "TD6_MSS0.03_SM_meta_max_sto_0.6",
            "TD6_MSS0.03_SM_meta_max_sto_0.8",
        ],
        "lower": [
            "TD6_MSS0.03_SM_meta_max_sto_-0.2",
            "TD6_MSS0.03_SM_meta_max_sto_-0.4",
            "TD6_MSS0.03_SM_meta_max_sto_-0.6",
            "TD6_MSS0.03_SM_meta_max_sto_-0.8",
        ],
    }
    rel_inf_corr = {
        "upper": [
            "TD6_MSS0.03_SM_meta_rel_inf_corr_0.2",
            "TD6_MSS0.03_SM_meta_rel_inf_corr_0.4",
            "TD6_MSS0.03_SM_meta_rel_inf_corr_0.6",
            "TD6_MSS0.03_SM_meta_rel_inf_corr_0.8",
        ],
        "lower": [
            "TD6_MSS0.03_SM_meta_rel_inf_corr_-0.2",
            "TD6_MSS0.03_SM_meta_rel_inf_corr_-0.4",
            "TD6_MSS0.03_SM_meta_rel_inf_corr_-0.6",
            "TD6_MSS0.03_SM_meta_rel_inf_corr_-0.8",
        ],
    }

    graphs = [
        rts["upper"],
        rts["lower"],
        max_sto["upper"],
        max_sto["lower"],
        rel_inf_corr["upper"],
        rel_inf_corr["lower"],
    ]

    titles = [
        r"Lower$RT$",
        r"Upper $RT$",
        r"Lower $S_{max}$",
        r"Upper $S_{max}$",
        r"Lower $r(D_t, NI_t)$",
        r"Upper $r(D_t, NI_t)$",
    ]

    width = 12
    height = 10
    fig, axes = plt.subplots(
        3,
        2,
        sharex="col",
        sharey=True,
        figsize=(width, height),
    )
    label_args = [
        {"label_x": False, "label_y": True, "legend": True},
        {"label_x": False, "label_y": False, "legend": False},
        {"label_x": False, "label_y": True, "legend": False},
        {"label_x": False, "label_y": False, "legend": False},
        {"label_x": True, "label_y": True, "legend": False},
        {"label_x": True, "label_y": False, "legend": False},
    ]

    plot_iterator = zip(graphs, axes.flatten(), titles, label_args)
    for graph_paths, ax, title, label_arg in plot_iterator:
        plot_experimental_leave_out_ratio(graph_paths, ax=ax)
        if not label_arg["label_x"]:
            ax.set_xlabel("")
        if not label_arg["label_y"]:
            ax.set_ylabel("")
        if not label_arg["legend"]:
            ax.get_legend().remove()
        ax.set_title(title, pad=6)
    fig.align_xlabels()
    fig.align_ylabels()
    plt.subplots_adjust(
        top=0.958,
        bottom=0.069,
        left=0.047,
        right=0.985,
        hspace=0.15,
        wspace=0.05,
    )
    plt.savefig(
        os.path.expanduser(
            "~/Dropbox/plrt-conus-figures/good_figures/experimental_result/"
            "leave_out_ratio_test.svg",
        ),
        format="svg",
        dpi=1200,
        bbox_inches="tight",
    )
    plt.show()


def plot_experimental_leave_out_ratio(graph_paths, ax=None):
    """Plot the leaveout ratio for the experimental results."""
    paths = [
        config.get_dir("results") / "monthly_merged_data_set_minyr3" / model_path
        for model_path in graph_paths
    ]
    results = [load_model_results(path) for path in paths]
    # opt_model_results = load_model_results(
    #     config.get_dir("results")
    #     / "monthly_merged_data_set_minyr3"
    #     / "TD6_MSS0.03_SM_basin_0.8",
    # )
    # opt_simmed = opt_model_results["simmed_data"]

    scores = []
    for model_results, ratio in zip(results, [0.2, 0.4, 0.6, 0.8]):
        train_reservoirs = (
            model_results["train_data"].index.get_level_values("res_id").unique()
        )
        test_reservoirs = (
            model_results["test_data"].index.get_level_values("res_id").unique()
        )
        simmed_data = model_results["simmed_data"]
        train_data = simmed_data.loc[
            simmed_data.index.get_level_values("res_id").isin(train_reservoirs)
        ]
        test_data = simmed_data.loc[
            simmed_data.index.get_level_values("res_id").isin(test_reservoirs)
        ]
        train_nnse = get_nnse(
            train_data,
            "actual",
            "model",
            "res_id",
        )
        test_nnse = get_nnse(
            test_data,
            "actual",
            "model",
            "res_id",
        )
        train_nnse = train_nnse.to_frame()
        train_nnse["ratio"] = ratio
        train_nnse["dset"] = "Training"
        test_nnse = test_nnse.to_frame()
        test_nnse["ratio"] = ratio
        test_nnse["dset"] = "Testing"
        rat_scores = pd.concat([train_nnse, test_nnse])
        scores.append(rat_scores)

    scores = pd.concat(scores)
    # sns.lineplot(
    #     data=scores,
    #     y="NNSE",
    #     x="ratio",
    #     hue="dset",
    #     ax=ax,
    # )
    sns.boxplot(
        data=scores,
        y="NNSE",
        x="ratio",
        hue="dset",
        ax=ax,
        showfliers=False,
    )
    # sns.barplot(
    #     data=scores,
    #     x="ratio",
    #     y="NNSE",
    #     hue="dset",
    #     ax=ax,
    # )
    ax.legend(title="")
    ax.set_xlabel("Testing Ratio")
    ax.set_ylabel("nNSE")


if __name__ == "__main__":
    # sns.set_theme(context="notebook", palette="colorblind", font_scale=1.1)
    plt.style.use(["science", "nature"])
    sns.set_context("paper", font_scale=1.2)
    # mpl.rcParams["xtick.major.size"] = 8
    # mpl.rcParams["xtick.major.width"] = 1
    mpl.rcParams["xtick.minor.size"] = 0
    # mpl.rcParams["xtick.minor.width"] = 1
    # mpl.rcParams["ytick.major.size"] = 8
    # mpl.rcParams["ytick.major.width"] = 1
    # mpl.rcParams["ytick.minor.size"] = 4
    # mpl.rcParams["ytick.minor.width"] = 1
    # mpl.rcParams["axes.linewidth"] = 1.5
    # plt.style.use("tableau-colorblind10")
    # args, remaining = parse_args()
    # func_args = parse_unknown_args(remaining)

    # if args.model_path:
    #     model_path = args.model_path
    # else:
    model_path = "TD6_MSS0.03_SM_basin_0.8"
    full_model_path = (
        config.get_dir("results") / "monthly_merged_data_set_minyr3" / model_path
    )

    model_results = load_model_results(full_model_path)
    model = load_model_file(full_model_path)
    model_data = load_pickle(full_model_path / "datasets.pickle")

    # * Find operational groups
    # find_operational_groups_for_res(model, model_data)

    # * Plot operational groups on map
    # plot_operational_group_map(model_results)

    # * Get model equation coeff dataframe
    # get_coefficient_dataframe(model, model_data)

    # * get breakdown of reservoirs per mode per basin
    # get_basin_op_mode_breakdown()

    # * plot seasonal operations for all TV groups
    # plot_seasonal_operations(model, model_data, polar=False)

    # * get seasonal operations for a specific group
    # plot_basin_specific_seasonal_operations(model, model_data, "Medium, High RT")

    # for mode in TIME_VARYING_GROUPS[:1]:
    #    plot_basin_specific_seasonal_operations(model, model_data, mode)

    # * Plot seasonal operation maps for a specific group
    # for op_group in TIME_VARYING_GROUPS:
    #     plot_reservoir_most_likely_group_maps(model, model_data, op_group)

    # * Plot charts for reservoir group breakdown for each basin
    # plot_basin_group_makeup()

    # * Plot multicolored line plots
    # plot_res_group_colored_timeseries(model_results, model, model_data)

    # * Plot basin group variance map
    # for group in TIME_VARYING_GROUPS:
    #     print(f"\n{group}\n")
    #     plot_basin_group_entropy(model, model_data, op_group=group, plot_res=False)

    # * Plot training vs testing simul performance
    # plot_training_vs_testing_simul_perf(model_results)

    # * Transition probabilities
    # transition_probabilities(model, model_data)

    # * Plot bar charts of coefficient values
    # plot_coef_bar(model_path, split=True)

    # * determine variable importance
    # calc_variable_importance(model, model_data)

    # * plot experimental results
    # plot_experimental_dset_sim_perf()

    # * plot experimental nse decomp
    # plot_experimental_dset_nse_decomp()

    # * plot experimental ratio leave outs
    plot_experimental_leave_out_ratios()
