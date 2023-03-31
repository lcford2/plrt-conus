import calendar
import os
from collections import defaultdict
from multiprocessing import cpu_count

import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from fit_plrt_model import get_params_and_groups
from matplotlib.cm import get_cmap
from matplotlib.colors import Normalize
from parameter_sweep_analysis import (
    get_contiguous_wbds,
    load_model_file,
    load_model_results,
    setup_map,
)
from utils.config import config
from utils.io import load_feather, load_pickle, write_pickle
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
    "Medium-Large",
    "Large, Low RT",
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
    write_pickle(
        res_op_groups,
        config.get_dir("agg_results") / "best_model_op_groups.pickle",
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
        res_coords = [
            (row.LONG_DD, row.LAT_DD)
            for _, row in grand[grand["GRAND_ID"].isin(resers)].iterrows()
        ]
        res_x, res_y = list(zip(*res_coords))
        m.scatter(
            res_x,
            res_y,
            latlon=True,
            marker="v",
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


def plot_basin_specific_seasonal_operations(model, model_data, op_group):
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
    plt.show()


if __name__ == "__main__":
    sns.set_theme(context="notebook", palette="colorblind", font_scale=1.1)
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

    # * Find operational groups
    # model = load_model_file(full_model_path)
    # model_data = load_pickle(full_model_path / "datasets.pickle")
    # find_operational_groups_for_res(model, model_data)

    # * Plot operational groups on map
    # plot_operational_group_map(model_results)

    # * Get model equation coeff dataframe
    model = load_model_file(full_model_path)
    model_data = load_pickle(full_model_path / "datasets.pickle")
    get_coefficient_dataframe(model, model_data)

    # * get breakdown of reservoirs per mode per basin
    # get_basin_op_mode_breakdown()

    # * get seasonal operations for a specific mode
    # model = load_model_file(full_model_path)
    # model_data = load_pickle(full_model_path / "datasets.pickle")
    # modes = [
    #     "Small, Mid RT",
    #     "Medium-Large",
    #     "Medium, Mid RT",
    #     "Medium, High RT",
    #     "Large",
    #     "Very Large",
    # ]

    # for mode in modes[3:4]:
    #     try:
    #         plot_basin_specific_seasonal_operations(model, model_data, mode)
    #     except Exception:
    #         pass
