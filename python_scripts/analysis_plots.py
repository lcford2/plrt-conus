import calendar
import os
import re
from collections import defaultdict
from multiprocessing import cpu_count

import geopandas as gpd
import matplotlib.patches as mpatch
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import pandas as pd
import seaborn as sns
from fit_plrt_model import get_params_and_groups
from matplotlib.cm import get_cmap
from matplotlib.colors import ListedColormap, Normalize
from parameter_sweep_analysis import (
    get_contiguous_wbds,
    load_model_file,
    load_model_results,
    setup_map,
)
from utils.config import config
from utils.io import load_feather, load_pickle, write_feather, write_pickle
from utils.plot_tools import determine_grid_size
from utils.utils import format_equation

plt.rcParams["svg.fonttype"] = "none"

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
        records, config.get_dir("agg_results") / "best_model_op_groups.feather"
    )


def plot_operational_group_map(model_results):
    res_op_groups = load_pickle(
        config.get_dir("agg_results") / "best_model_op_groups.pickle"
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
    param_df = pd.DataFrame.from_records(params, columns=feats, index=groups.index)
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
        config.get_dir("agg_results") / "best_model_op_groups.feather"
    )
    res_huc2 = load_feather(config.get_dir("spatial_data") / "updated_res_huc2.feather")
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


def get_res_seasonal_operations(model, model_data, op_group):
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

    res_op_groups = load_feather(
        config.get_dir("agg_results") / "best_model_op_groups.feather"
    )

    res_op_groups["op_group"] = res_op_groups["op_group"].replace(
        {"Large 1": "Large", "Large 2": "Large", "Large 3": "Large"}
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


def plot_basin_specific_seasonal_operations(model, model_data, op_group):
    props = get_res_seasonal_operations(model, model_data, op_group)
    resers = props["res_id"].unique()
    res_huc2 = load_feather(config.get_dir("spatial_data") / "updated_res_huc2.feather")
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

    res_huc2 = load_feather(config.get_dir("spatial_data") / "updated_res_huc2.feather")
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
        most_likely_groups, columns=["res", *range(1, 13)]
    )
    most_likely_groups = most_likely_groups.set_index("res")
    most_likely_groups["basin"] = res_huc2

    most_likely_basin_groups = []
    for basin in most_likely_groups["basin"].unique():
        bdf = most_likely_groups[most_likely_groups["basin"] == basin]
        mode = bdf.mode(axis=0)
        most_likely_basin_groups.append([basin, *mode[range(1, 13)].values[0]])

    most_likely_basin_groups = pd.DataFrame.from_records(
        most_likely_basin_groups, columns=["basin", *range(1, 13)]
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
        {"Large 1": "Large", "Large 2": "Large", "Large 3": "Large"}
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


if __name__ == "__main__":
    # sns.set_theme(context="notebook", palette="colorblind", font_scale=1.1)
    sns.set_context("notebook", font_scale=1.1)
    plt.style.use("tableau-colorblind10")
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

    # * get seasonal operations for a specific group
    # plot_basin_specific_seasonal_operations(model, model_data, "Medium, High RT")

    # for mode in TIME_VARYING_GROUPS[:1]:
    #    plot_basin_specific_seasonal_operations(model, model_data, mode)

    # * Plot seasonal operation maps for a specific group
    # for op_group in TIME_VARYING_GROUPS:
    #     plot_reservoir_most_likely_group_maps(model, model_data, op_group)

    # * Plot charts for reservoir group breakdown for each basin
    plot_basin_group_makeup()
