import argparse
import calendar
import glob
import os
import re
from collections import defaultdict
from itertools import combinations
from multiprocessing import cpu_count

import geopandas as gpd
import matplotlib.gridspec as mgridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scienceplots  # noqa: F401
import seaborn as sns
from fit_plrt_model import load_resopsus_data
from joblib import Parallel, delayed
from matplotlib.cm import ScalarMappable, get_cmap
from matplotlib.colors import ListedColormap, Normalize
from parameter_sweep_analysis import get_contiguous_wbds, load_model_results, setup_map
from scipy.spatial.distance import cosine as cosine_dist
from single_tree_breakdown import get_groups_for_model
from utils.config import config
from utils.io import load_feather, load_huc2_basins, load_huc2_name_map, write_pickle
from utils.metrics import get_nnse, get_nrmse
from utils.plot_tools import get_pretty_var_name, mxbline
from utils.utils import sorted_k_partitions

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
    "Large": [11, 12, 13, 14, 16, 17],
    "Very Large": [11, 12, 13, 15, 16, 17],
}


def parse_args():
    parser = argparse.ArgumentParser(description="Compare basin results")
    parser.add_argument(
        "-b",
        "--basins",
        nargs="+",
        type=int,
        help="Which two basins should  be compared",
        required=False,
    )
    parser.add_argument(
        "-m",
        "--model-path",
        dest="model_path",
        help="The model results that should be plotted",
        required=False,
    )
    return parser.parse_known_args()


def parse_unknown_args(unknown_args):
    pattern = re.compile(r"--(.*)=(.*)")
    output = {}
    for arg in unknown_args:
        search_result = re.search(pattern, arg)
        if search_result:
            key = search_result.group(1)
            value = search_result.group(2)

            if value in ("true", "True"):
                value = True
            elif value in ("false", "False"):
                value = False
            elif value.isdecimal():
                value = int(value)
            else:
                try:
                    value = float(value)
                except ValueError:
                    pass

            output[key] = value
    return output


def plot_basin_tree_breakdown_comparison(basins, results):
    huc2 = load_huc2_basins()
    groups = get_groups_for_model(results)
    groups.name = "group"
    groups = groups.to_frame()
    groups["basin"] = [huc2.loc[i, "huc2_id"] for i in groups.index.get_level_values(0)]
    groups = groups[groups["basin"].isin(basins)]
    counts = groups.groupby(["res_id", "basin"])["group"].value_counts()
    counts.name = "count"
    counts = counts.reset_index()
    res_sums = counts.groupby("res_id")["count"].sum()
    for res, rsum in res_sums.items():
        counts.loc[counts["res_id"] == res, "count"] /= rsum

    huc2_names = load_huc2_name_map()
    counts["basin"] = counts["basin"].replace(huc2_names)
    fg = sns.catplot(
        data=counts,
        x="group",
        y="count",
        hue="basin",
        kind="box",
        whis=(0.1, 0.9),
        showfliers=True,
        legend_out=False,
    )
    fg.set_ylabels("Group Proportion")
    fg.set_xlabels("Tree Group")
    plt.show()


def plot_seasonal_tree_breakdown_basin_comparison(basins, results, plot_type="line"):
    huc2 = load_huc2_basins()
    # huc2_names = load_huc2_name_map()
    groups = get_groups_for_model(results)
    groups.name = "group"
    groups = groups.to_frame()
    groups["basin"] = [huc2.loc[i, "huc2_id"] for i in groups.index.get_level_values(0)]
    groups = groups[groups["basin"].isin(basins)]
    groups["month"] = groups.index.get_level_values(1).month
    groups = groups.reset_index()
    counts = groups.groupby(["res_id", "month"])["group"].value_counts()
    sums = counts.groupby(["res_id", "month"]).sum()
    idx = pd.IndexSlice
    for (res_id, month), rsum in sums.items():
        values = counts.loc[idx[res_id, month, :]]
        values /= rsum
        counts.loc[idx[res_id, month, :]] = values.values

    counts.name = "prop"
    counts = counts.reset_index()
    counts["basin"] = [huc2.loc[i, "name"] for i in counts["res_id"]]
    props = counts.groupby(["basin", "month", "group"])["prop"].mean()
    props *= 100
    if plot_type == "box":
        fg = sns.catplot(
            data=props.reset_index(),
            x="month",
            y="prop",
            # hue="basin",
            col="group",
            col_wrap=5,
            kind="box",
            legend_out=False,
            height=5,
            aspect=0.7,
        )
    else:
        fg = sns.relplot(
            data=props.reset_index(),
            x="month",
            y="prop",
            # hue="basin",
            col="group",
            col_wrap=5,
            kind="line",
            height=5,
            aspect=0.7,
        )

    fg.set_ylabels("Group Occ. [%]")
    fg.set_xlabels("Month")
    fg.set_titles("Tree Node: {col_name}")
    for ax in fg.axes.flatten():
        ax.set_xticks(range(1, 13))
    fg.set_xticklabels(range(1, 13))

    plt.subplots_adjust(
        top=0.963,
        bottom=0.064,
        left=0.039,
        right=0.991,
        hspace=0.106,
        wspace=0.051,
    )
    output_dir = os.path.expanduser("~/Dropbox/plrt-conus-figures/basin_comparison")
    basin_string = "-".join(map(str, basins))
    if plot_type == "box":
        output_file = f"monthly_basin_compare_{basin_string}_box.png"
    else:
        output_file = f"monthly_basin_compare_{basin_string}_line.png"

    plt.savefig("/".join([output_dir, output_file]))
    plt.show()


def plot_basin_comparison_map(basin):
    comp_data = load_feather(
        config.get_dir("agg_results") / "basin_comp_metrics.feather",
    )
    basin_data = comp_data.loc[
        (comp_data["level_0"] == basin) | (comp_data["level_1"] == basin)
    ]
    other_basins = list(set([*basin_data["level_0"], *basin_data["level_1"]]))
    other_basins.remove(basin)
    basin_data = basin_data.set_index(["level_0", "level_1"])
    scores_var = "cosine"
    scores = {}
    for obasin in other_basins:
        try:
            score = basin_data.loc[pd.IndexSlice[basin, obasin], scores_var]
        except KeyError:
            score = basin_data.loc[pd.IndexSlice[obasin, basin], scores_var]
        scores[obasin] = score
    scores = pd.Series(scores)

    # fig, ax = plt.subplots(1, 1)
    fig = plt.figure()
    gs = mgridspec.GridSpec(1, 2, figure=fig, width_ratios=[20, 1])
    ax = fig.add_subplot(gs[0, 0])
    cbar_ax = fig.add_subplot(gs[0, 1])
    wbds = get_contiguous_wbds()
    wbd_ids = [re.search(r"WBD_(\d\d)_HU2", i).group(1) for i in wbds]
    norm = Normalize(vmin=scores.min(), vmax=scores.max())
    cmap = get_cmap("plasma_r")
    color_vars = []
    for wbd in wbd_ids:
        if wbd == "04":
            color_vars.append("k")
            continue
        try:
            color_vars.append(cmap(norm(scores[int(wbd)])))
        except KeyError:
            color_vars.append("w")

    other_bounds = [(b, "k", c) for b, c in zip(wbds, color_vars)]

    west, south, east, north = (
        -127.441406,
        24.207069,
        -66.093750,
        53.382373,
    )
    setup_map(ax=ax, coords=[west, south, east, north], other_bound=other_bounds)
    plt.colorbar(
        ScalarMappable(norm=norm, cmap=cmap),
        cax=cbar_ax,
        orientation="vertical",
        label="Cosine Distance",
        aspect=4,
        shrink=0.8,
    )
    # wbd_gdfs = [gpd.read_file(i+".shp") for i in wbds]
    # for wbd_id, wbd_gdf in zip(wbd_ids, wbd_gdfs):
    #     centroid = wbd_gdf.centroid[0]
    #     x, y = centroid.x, centroid.y
    #     print(x, y)
    #     ax.text(x, y, wbd_id)/comp_data
    plt.show()


def plot_grouped_basin_map():
    basins = {}
    with open(config.get_dir("spatial_data") / "huc2_names.csv", "r") as f:
        for line in f.readlines():
            line = line.strip("\n\r")
            j, i = line.split(",")
            basins[i] = int(j)

    # wbds = get_contiguous_wbds()
    # wbd_ids = [re.search(r"WBD_(\d\d)_HU2", i).group(1) for i in wbds]
    # wbd_map = {int(i): wbd for i, wbd in zip(wbd_ids, wbds)}

    # norm = Normalize(vmin=0, vmax=2)
    # cmap = get_cmap("plasma_r")
    # parts = load_pickle(config.get_dir("agg_results") / "best_partitions_3.pickle")
    # thresh = 3
    # filtered_parts = [i for i in parts if i[-1] == thresh]

    # best_part = filtered_parts[0][0]

    # color_pal = sns.color_palette("Set2")
    # color_dict = {
    #     tuple(item): color_pal[i]
    #     for i, (k, item) in enumerate(BASIN_GROUPS.items())
    #     # for i, item in enumerate(best_part)
    # }

    # color_vars = {}
    # for wbd_id in wbd_ids:
    #     for gbasins, color in color_dict.items():
    #         if int(wbd_id) in gbasins:
    #             color_vars[int(wbd_id)] = color

    # other_bounds = [(wbd_map[i], "k", color_vars[i]) for i in color_vars.keys()]

    west, south, east, north = (
        -127.441406,
        24.207069,
        -66.093750,
        53.382373,
    )
    # setup_map(coords=[west, south, east, north], other_bound=other_bounds)
    setup_map(coords=[west, south, east, north])
    # wbd_gdfs = [gpd.read_file(i+".shp") for i in wbds]
    # for wbd_id, wbd_gdf in zip(wbd_ids, wbd_gdfs):
    #     centroid = wbd_gdf.centroid[0]
    #     x, y = centroid.x, centroid.y
    #     print(x, y)
    #     ax.text(x, y, wbd_id)
    # handles = [mpatch.Patch(edgecolor="k", facecolor=color_pal[i]) for i in range(3)]
    # labels = [str(i) for i in best_part]
    # labels = BASIN_GROUPS.keys()
    # ax = plt.gca()
    # ax.legend(handles, labels, loc="best")
    plt.show()


def find_similar_basins():
    comp_data = load_feather(
        config.get_dir("agg_results") / "basin_comp_metrics.feather",
    )
    comp_data = comp_data.pivot(index="level_0", columns="level_1", values="cosine")
    all_df = pd.DataFrame(index=range(1, 19), columns=range(1, 19))

    for i in range(1, 18):
        if i == 4:
            continue
        for j in range(2, 19):
            if j == 4:
                continue
            try:
                value = comp_data.loc[i, j]
            except KeyError:
                value = comp_data.loc[j, i]

            if np.isnan(value):
                value = comp_data.loc[j, i]
            all_df.loc[i, j] = value

    closests = {}

    for i, row in all_df.iterrows():
        sort_row = row.sort_values()
        row_best = sort_row.head(5).index.values
        closests[i] = (row_best, sort_row[row_best].mean())

    for key, (index, value) in sorted(closests.items(), key=lambda x: x[1][1]):
        resers = [key, *list(index)]
        print(*sorted(resers), value)

    basins = list(range(1, 19))
    basins.remove(4)

    potential_groups = []
    for n in range(2, 16):
        potential_groups.extend(combinations(basins, n))

    scores = []
    for group in potential_groups:
        gscores = []
        checked = []
        for i in group:
            for j in group:
                if i == j:
                    continue
                index = (min((i, j)), max((i, j)))
                if index not in checked:
                    checked.append(index)
                    gscores.append(comp_data.loc[index])
        scores.append(np.mean(gscores))

    scores = [(i, j) for i, j in enumerate(scores)]
    ranked_scores = sorted(scores, key=lambda x: x[1])
    ranked_groups = [(potential_groups[i], j) for i, j in ranked_scores]

    score_dict = {tuple(g): s for g, s in ranked_groups}

    groups_by_size = {}
    for i in range(2, 16):
        sub_groups = [g for g in ranked_groups if len(g[0]) == i]
        groups_by_size[i] = sub_groups

    best_group = groups_by_size[6][0]
    next_best_group, worst_group = "", ""
    for group in groups_by_size[6]:
        is_next_best = True
        for b in group[0]:
            if b in best_group[0]:
                is_next_best = False
                break
        if is_next_best:
            next_best_group = group
            break

    remaining = [i for i in basins if i not in [*best_group[0], *next_best_group[0]]]

    for g in groups_by_size[len(remaining)]:
        if g[0] == remaining:
            worst_group = g

    print(best_group, next_best_group, worst_group)
    # two_group_size = 6
    # third_group = len(basins) - 2 * two_group_size

    poss_partitions = sorted_k_partitions(basins, 3)

    # filtered = [i for i in poss_partitions if all([len(p) > 1 for p in i])]
    similar_filtered = [
        i for i in poss_partitions if filter_partitions_by_similar_size(i, 3)
    ]

    nprocs = CPUS
    nitems = len(similar_filtered)
    chunk_size = nitems // (nprocs - 1)
    chunked_parts = [
        similar_filtered[i * chunk_size : (i + 1) * chunk_size] for i in range(nprocs)
    ]

    results = Parallel(n_jobs=48, verbose=11)(
        delayed(get_part_scores)(parts, score_dict) for parts in chunked_parts
    )

    scores = []
    for i in results:
        scores.extend(i)

    scores = np.array(scores)
    mean = scores.mean(axis=1)
    mean = [tup for tup in enumerate(list(mean))]
    mean.sort(key=lambda x: x[1])

    output = [
        (similar_filtered[i], j, find_partitions_size_diff(similar_filtered[i]))
        for i, j in mean
    ]
    write_pickle(output, config.get_dir("agg_results") / "best_partitions_3.pickle")

    # groups_by_size = defaultdict(list)
    # for i, j in mean:
    #     groups = similar_filtered[i]
    #     min_size = min(len(k) for k in groups)
    #     groups_by_size[min_size].append((groups, j))
    from IPython import embed as II

    II()


def filter_partitions_by_size(part):
    is_valid = True
    for p in part:
        if len(p) < 2:
            is_valid = False
            break
    if is_valid:
        return part
    else:
        return None


def find_partitions_size_diff(part):
    max_diff = 0
    for i, j in combinations(range(len(part)), 2):
        diff = abs(len(part[i]) - len(part[j]))
        if diff > max_diff:
            max_diff = diff
    return max_diff


def filter_partitions_by_similar_size(part, thresh):
    for i, j in combinations(range(len(part)), 2):
        i_size = len(part[i])
        j_size = len(part[j])
        if abs(i_size - j_size) > thresh:
            return False
    return True


def get_part_scores(parts, score_dict):
    pscores = []
    for i, part in enumerate(parts):
        pscores.append([score_dict[tuple(p)] for p in part])
        # if i % 1000 == 0:
        #     print(f"Iteration {i}")
        # for g, s in groups_by_size[len(p)]:
        #     if tuple(p) == tuple(g):
        #         pscores.append(s)
    # return pscores
    return pscores


def find_similar_reservoir_characteristics(model_results):
    groups = model_results["groups"]
    counts = groups.groupby(
        [
            groups.index.get_level_values(0),
            groups.index.get_level_values(1).month,
        ]
    ).value_counts()

    seasonal_percentages = {}
    same_groups = defaultdict(list)
    idx = pd.IndexSlice
    for res in counts.index.get_level_values(0).unique():
        rdf = counts.loc[idx[res, :, :]].unstack()
        rdf = rdf.fillna(0.0)
        rdf = rdf.divide(rdf.sum(axis=1), axis=0)
        seasonal_percentages[res] = rdf
        same_groups[tuple(sorted(rdf.columns))].append(res)

    all_group_res = same_groups[(1, 2, 3, 4, 5, 6, 7, 8, 9, 10)]
    distances = []
    completed_pairs = []
    for res1 in all_group_res:
        rdf1 = seasonal_percentages[res1]
        for res2 in all_group_res:
            if res2 == res1:
                continue
            # if (res2, res1) in completed_pairs:
            #     continue
            rdf2 = seasonal_percentages[res2]
            total_distance = 0
            for column in rdf2.columns:
                total_distance += np.abs(cosine_dist(rdf1[column], rdf2[column]))
            distances.append((res1, res2, total_distance))
            completed_pairs.append((res1, res2))

    distances = pd.DataFrame.from_records(distances, columns=["res1", "res2", "dist"])
    distances[
        [
            "res1_rt",
            "res2_rt",
            "res1_max_sto",
            "res2_max_sto",
            "res1_rel_inf_corr",
            "res2_rel_inf_corr",
        ]
    ] = 0.0
    res_data, res_meta = load_resopsus_data()

    for i, row in distances.iterrows():
        res1, res2 = row.res1, row.res2
        res1_meta = res_meta.loc[res1]
        res2_meta = res_meta.loc[res2]
        row.res1_rt = res1_meta.rts
        row.res1_max_sto = res1_meta.max_sto
        row.res1_rel_inf_corr = res1_meta.rel_inf_corr
        row.res2_rt = res2_meta.rts
        row.res2_max_sto = res2_meta.max_sto
        row.res2_rel_inf_corr = res2_meta.rel_inf_corr
        distances.loc[i] = row

    distances["rt_diff"] = (distances["res1_rt"] - distances["res2_rt"]).abs()
    distances["max_sto_diff"] = (
        distances["res1_max_sto"] - distances["res2_max_sto"]
    ).abs()
    distances["rel_inf_corr_diff"] = (
        distances["res1_rel_inf_corr"] - distances["res2_rel_inf_corr"]
    ).abs()

    diff_columns = ["rt_diff", "max_sto_diff", "rel_inf_corr_diff"]

    closest_reservoirs = {}
    closest_records = []
    for res in all_group_res:
        rdf = distances[distances["res1"] == res]
        rdf = rdf.sort_values(by="dist")
        key = rdf.head(10)["dist"].mean()
        closest_reservoirs[(res, key)] = rdf.head(10)
        closest_records.extend(rdf.head(10).values)

    closest_diffs = []
    for res, rdf in closest_reservoirs.items():
        closest_diffs.append(
            [
                *res,
                *rdf[diff_columns].mean().values,
            ]
        )
    closest_diffs = pd.DataFrame.from_records(
        closest_diffs,
        columns=["res", "mean_dist", *diff_columns],
    )
    closest_df = pd.DataFrame.from_records(closest_records, columns=distances.columns)
    fig, axes = plt.subplots(1, 3, sharey=True)
    axes = axes.flatten()
    xlabels = [f"Absolute Difference {i}" for i in ["RT", "St. Cap.", "r(R, I)"]]
    for column, ax, xlabel in zip(diff_columns, axes, xlabels):
        # y = closest_diffs["mean_dist"]
        # x = closest_diffs[column]
        y = closest_df["dist"]
        x = closest_df[column]
        ax.scatter(x, y)
        if ax == axes[0]:
            ax.set_ylabel("Cosine Distance")
        ax.set_xlabel(xlabel)

    plt.show()


def plot_reservoir_group_access_map():
    basins = {}
    with open(config.get_dir("spatial_data") / "huc2_names.csv", "r") as f:
        for line in f.readlines():
            line = line.strip("\n\r")
            j, i = line.split(",")
            basins[i] = int(j)

    # wbds = get_contiguous_wbds()
    # wbd_ids = [re.search(r"WBD_(\d\d)_HU2", i).group(1) for i in wbds]
    # wbd_map = {int(i): wbd for i, wbd in zip(wbd_ids, wbds)}

    res_access = pd.read_csv(
        "../aggregated_results/res_group_access.csv", index_col=0, dtype=str
    )
    grand = gpd.read_file(config.get_dir("spatial_data") / "my_grand_info")

    groups = res_access.columns
    color_pal = sns.color_palette("Set2")
    group_colors = {group: color_pal[i] for i, group in enumerate(groups)}

    # color_dict = {
    #     tuple(item): color_pal[i] for i, (k, item) in enumerate(BASIN_GROUPS.items())
    # }

    # color_vars = {}
    # for wbd_id in wbd_ids:
    #     for gbasins, color in color_dict.items():
    #         if int(wbd_id) in gbasins:
    #             color_vars[int(wbd_id)] = color

    # other_bounds = [(wbd_map[i], "k", "w") for i in color_vars.keys()]
    other_bounds = []

    west, south, east, north = (
        -127.441406,
        24.207069,
        -66.093750,
        53.382373,
    )
    m = setup_map(coords=[west, south, east, north], other_bound=other_bounds)

    for group in groups:
        resers = res_access[group].dropna().values
        res_coords = [
            (row.LONG_DD, row.LAT_DD)
            for i, row in grand[grand["GRAND_ID"].isin(resers)].iterrows()
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
            sizes=[40],
        )
    ax = plt.gca()
    ax.legend(loc="lower left")
    plt.show()


def plot_reservoir_most_likely_group_maps(model_results):
    basins = {}
    with open(config.get_dir("spatial_data") / "huc2_names.csv", "r") as f:
        for line in f.readlines():
            line = line.strip("\n\r")
            j, i = line.split(",")
            basins[i] = int(j)

    wbds = get_contiguous_wbds()
    wbd_ids = [re.search(r"WBD_(\d\d)_HU2", i).group(1) for i in wbds]
    wbd_map = {int(i): wbd for i, wbd in zip(wbd_ids, wbds)}

    grand = gpd.read_file(config.get_dir("spatial_data") / "my_grand_info")

    groups = model_results["groups"]
    counts = groups.groupby(
        [
            groups.index.get_level_values(0),
            groups.index.get_level_values(1).month,
        ]
    ).value_counts()

    most_likely_groups = []
    idx = pd.IndexSlice
    for res in counts.index.get_level_values(0).unique():
        rdf = counts.loc[idx[res, :, :]].unstack()
        rdf = rdf.fillna(0.0)
        rdf = rdf.divide(rdf.sum(axis=1), axis=0)
        most_likely_groups.append([res, *rdf.idxmax(axis=1).values])

    most_likely_groups = pd.DataFrame.from_records(
        most_likely_groups, columns=["res", *range(1, 13)]
    )
    # norm = Normalize(vmin=0, vmax=11)
    # cmap = get_cmap("Set3")
    color_pal = sns.color_palette("Paired")
    norm = Normalize(vmin=1, vmax=len(groups.unique()))
    cmap = ListedColormap(color_pal.as_hex()[: len(groups.unique())])
    # group_colors = {i + 1: cmap(norm(i)) for i in range(10)}
    # group_colors = {i + 1: color_pal[i] for i in range(10)}
    resers = most_likely_groups["res"].values
    res_coords = [
        (row.LONG_DD, row.LAT_DD)
        for i, row in grand[grand["GRAND_ID"].isin(resers)].iterrows()
    ]
    res_x, res_y = list(zip(*res_coords))

    res_huc2 = load_feather(config.get_dir("spatial_data") / "updated_res_huc2.feather")
    res_huc2["huc2_id"] = [f"{i:02d}" for i in res_huc2["huc2_id"]]
    res_huc2 = res_huc2.set_index("res_id")
    res_huc2 = res_huc2.loc[resers]
    most_likely_groups = most_likely_groups.set_index("res")
    basin_month_colors = {}

    for basin in res_huc2["huc2_id"].unique():
        bres = res_huc2[res_huc2["huc2_id"] == basin].index
        bgroups = most_likely_groups.loc[bres]
        basin_month_colors[basin] = [cmap(norm(i)) for i in bgroups.mean().values]

    # color_dict = {
    #     tuple(item): "w" for i, (k, item) in enumerate(BASIN_GROUPS.items())
    # }

    # color_vars = {}
    # for wbd_id in wbd_ids:
    #     for gbasins, color in color_dict.items():
    #         if int(wbd_id) in gbasins:
    #             color_vars[int(wbd_id)] = color

    # other_bounds = [(wbd_map[i], "k", color_vars[i]) for i in color_vars.keys()]

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

    # fig, axes = plt.subplots(3, 4, sharex=True, sharey=True)
    # axes = axes.flatten()
    fig = plt.figure()
    gs = mgridspec.GridSpec(3, 5, figure=fig, width_ratios=[10, 10, 10, 10, 1])
    axes = []
    for i in range(3):
        for j in range(4):
            axes.append(fig.add_subplot(gs[i, j]))

    cbar_ax = fig.add_subplot(gs[:, 4])

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

    # resers = most_likely_groups["res"].values
    # res_coords = [
    #     (row.LONG_DD, row.LAT_DD)
    #     for i, row in grand[grand["GRAND_ID"].isin(resers)].iterrows()
    # ]
    # res_x, res_y = list(zip(*res_coords))
    # for month, map_info in zip(range(1, 13), maps):
    #     m = map_info[0]
    #     groups = most_likely_groups[month]
    #     colors = [group_colors[g] for g in list(groups)]
    #     m.scatter(
    #         res_x,
    #         res_y,
    #         latlon=True,
    #         marker="v",
    #         color=colors,
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
    plt.colorbar(
        ScalarMappable(norm=norm, cmap=cmap),
        # ScalarMappable(
        #     # norm=Normalize(vmin=1, vmax=10),
        #     # cmap=ListedColormap(color_pal.as_hex()[:10]),
        #     cmap=cmap,
        # ),
        cax=cbar_ax,
        orientation="vertical",
        label="Most Likely Operational Mode",
        aspect=4,
        shrink=0.8,
    )
    plt.show()


def rank_reservoirs_by_performance(
    model_results, monthly_group=False, plot_pairs=False
):
    train_data = model_results["train_data"]
    test_data = model_results["test_data"]

    if monthly_group:
        train_grouper = [
            train_data.index.get_level_values(0),
            train_data.index.get_level_values(1).month,
        ]
        test_grouper = [
            test_data.index.get_level_values(0),
            test_data.index.get_level_values(1).month,
        ]
    else:
        train_grouper, test_grouper = "res_id", "res_id"

    train_nnse = get_nnse(train_data, "actual", "model", train_grouper)
    train_nrmse = get_nrmse(train_data, "actual", "model", train_grouper)
    test_nnse = get_nnse(test_data, "actual", "model", test_grouper)
    test_nrmse = get_nrmse(test_data, "actual", "model", test_grouper)

    train_metrics = pd.DataFrame.from_dict({"NNSE": train_nnse, "NRMSE": train_nrmse})
    test_metrics = pd.DataFrame.from_dict({"NNSE": test_nnse, "NRMSE": test_nrmse})

    if plot_pairs:
        train_fg = sns.pairplot(train_metrics)
        train_fg.figure.suptitle("Training Reservoirs")
        test_fg = sns.pairplot(test_metrics)
        test_fg.figure.suptitle("Testing Reservoirs")
        plt.show()

    return train_metrics.sort_values(by="NNSE"), test_metrics.sort_values(by="NNSE")


def plot_basin_mean_performance(
    model_results,
    metric="NNSE",
    data_set="train",
    plot_res=False,
    monthly=False,
):
    res_huc2 = load_feather(config.get_dir("spatial_data") / "updated_res_huc2.feather")
    res_huc2 = res_huc2.set_index("res_id")

    # get name of wbd files
    wbds = get_contiguous_wbds()
    wbd_ids = [re.search(r"WBD_(\d\d)_HU2", i).group(1) for i in wbds]
    wbd_map = {int(i): wbd for i, wbd in zip(wbd_ids, wbds)}

    # get grand database
    grand = gpd.read_file(config.get_dir("spatial_data") / "my_grand_info")

    df = model_results[f"{data_set}_data"]
    if metric == "NNSE":
        metric_func = get_nnse
    else:
        metric_func = get_nrmse

    if monthly:
        scores = metric_func(
            df,
            "actual",
            "model",
            [
                df.index.get_level_values(0),
                df.index.get_level_values(1).month,
            ],
        )
        scores = scores.unstack()
    else:
        scores = metric_func(df, "actual", "model", "res_id")

    resers = scores.index
    res_huc2 = res_huc2.loc[resers]
    if monthly:
        res_huc2 = res_huc2.join(scores)
    else:
        res_huc2[metric] = scores

    scores = res_huc2.groupby("huc2_id").mean()

    if monthly:
        max_score = scores.max().max()
        min_score = scores.min().min()
    else:
        scores = scores[metric]
        max_score = scores.max()
        min_score = scores.min()

    score_range = max_score - min_score

    norm = Normalize(
        vmin=max([min_score - score_range * 0.05, 0]),
        vmax=min([max_score + score_range * 0.05, 1]),
    )
    cmap = get_cmap("viridis")
    other_bounds = []
    if monthly:
        for month in range(1, 13):
            temp_bounds = []
            for basin in scores.index:
                wbd = wbd_map[int(basin)]
                color = cmap(norm(scores.loc[basin, month]))
                temp_bounds.append((wbd, "k", color))
            other_bounds.append(temp_bounds)
    else:
        for wbd_id, score in scores.items():
            wbd = wbd_map[int(wbd_id)]
            color = cmap(norm(score))
            other_bounds.append((wbd, "k", color))

    west, south, east, north = (
        -127.441406,
        24.207069,
        -66.093750,
        53.382373,
    )

    if monthly:
        fig = plt.figure()
        gs = mgridspec.GridSpec(3, 5, figure=fig, width_ratios=[10, 10, 10, 10, 1])
        axes = []
        for i in range(3):
            for j in range(4):
                axes.append(fig.add_subplot(gs[i, j]))
        cbar_ax = fig.add_subplot(gs[:, 4])
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
    else:
        fig = plt.figure()
        gs = mgridspec.GridSpec(1, 2, figure=fig, width_ratios=[20, 1])
        ax = fig.add_subplot(gs[0, 0])
        cbar_ax = fig.add_subplot(gs[:, 1])

        m = setup_map(
            coords=[west, south, east, north], other_bound=other_bounds, ax=ax
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
                sizes=res_huc2.loc[resers, metric].values * 50,
            )

    plt.colorbar(
        ScalarMappable(norm=norm, cmap=cmap),
        cax=cbar_ax,
        orientation="vertical",
        label=f"Basin Average {metric}",
        aspect=4,
        shrink=0.8,
    )
    if data_set == "simmed":
        fig.suptitle(f"{data_set.title()} Reservoirs")
    else:
        fig.suptitle(f"{data_set.title()}ing Reservoirs")
    # plt.subplots_adjust(
    #     top=0.894,
    #     bottom=0.049,
    #     left=0.058,
    #     right=0.937,
    #     hspace=0.2,
    #     wspace=0.142,
    # )
    plt.show()
    # if monthly:
    #     file_name = f"{data_set}_{metric}_monthly.png"
    # else:
    #     file_name = f"{data_set}_{metric}.png"

    # plt.savefig(
    #     os.path.expanduser(
    #         f"~/Dropbox/plrt-conus-figures/basin_performance/{file_name}"
    #     )
    # )


def plot_basin_group_variance(
    model_results,
    metric="NNSE",
    data_set="train",
    plot_res=False,
    monthly=False,
):
    res_huc2 = load_feather(config.get_dir("spatial_data") / "updated_res_huc2.feather")
    res_huc2 = res_huc2.set_index("res_id")

    # get name of wbd files
    wbds = get_contiguous_wbds()
    wbd_ids = [re.search(r"WBD_(\d\d)_HU2", i).group(1) for i in wbds]
    wbd_map = {int(i): wbd for i, wbd in zip(wbd_ids, wbds)}

    # get grand database
    grand = gpd.read_file(config.get_dir("spatial_data") / "my_grand_info")

    df = model_results[f"{data_set}_data"]
    if metric == "NNSE":
        metric_func = get_nnse
    else:
        metric_func = get_nrmse

    if monthly:
        scores = metric_func(
            df,
            "actual",
            "model",
            [
                df.index.get_level_values(0),
                df.index.get_level_values(1).month,
            ],
        )
        scores = scores.unstack()
    else:
        scores = metric_func(df, "actual", "model", "res_id")

    resers = scores.index
    res_huc2 = res_huc2.loc[resers]
    if monthly:
        res_huc2 = res_huc2.join(scores)
    else:
        res_huc2[metric] = scores

    scores = res_huc2.groupby("huc2_id").mean()

    if monthly:
        max_score = scores.max().max()
        min_score = scores.min().min()
    else:
        scores = scores[metric]
        max_score = scores.max()
        min_score = scores.min()

    score_range = max_score - min_score

    norm = Normalize(
        vmin=max([min_score - score_range * 0.05, 0]),
        vmax=min([max_score + score_range * 0.05, 1]),
    )
    cmap = get_cmap("viridis")
    other_bounds = []
    if monthly:
        for month in range(1, 13):
            temp_bounds = []
            for basin in scores.index:
                wbd = wbd_map[int(basin)]
                color = cmap(norm(scores.loc[basin, month]))
                temp_bounds.append((wbd, "k", color))
            other_bounds.append(temp_bounds)
    else:
        for wbd_id, score in scores.items():
            wbd = wbd_map[int(wbd_id)]
            color = cmap(norm(score))
            other_bounds.append((wbd, "k", color))

    west, south, east, north = (
        -127.441406,
        24.207069,
        -66.093750,
        53.382373,
    )

    if monthly:
        fig = plt.figure()
        gs = mgridspec.GridSpec(3, 5, figure=fig, width_ratios=[10, 10, 10, 10, 1])
        axes = []
        for i in range(3):
            for j in range(4):
                axes.append(fig.add_subplot(gs[i, j]))
        cbar_ax = fig.add_subplot(gs[:, 4])
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
    else:
        fig = plt.figure()
        gs = mgridspec.GridSpec(1, 2, figure=fig, width_ratios=[20, 1])
        ax = fig.add_subplot(gs[0, 0])
        cbar_ax = fig.add_subplot(gs[:, 1])

        m = setup_map(
            coords=[west, south, east, north], other_bound=other_bounds, ax=ax
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
                sizes=res_huc2.loc[resers, metric].values * 50,
            )

    plt.colorbar(
        ScalarMappable(norm=norm, cmap=cmap),
        cax=cbar_ax,
        orientation="vertical",
        label=f"Basin Average {metric}",
        aspect=4,
        shrink=0.8,
    )
    if data_set == "simmed":
        fig.suptitle(f"{data_set.title()} Reservoirs")
    else:
        fig.suptitle(f"{data_set.title()}ing Reservoirs")
    # plt.subplots_adjust(
    #     top=0.894,
    #     bottom=0.049,
    #     left=0.058,
    #     right=0.937,
    #     hspace=0.2,
    #     wspace=0.142,
    # )
    plt.show()
    # if monthly:
    #     file_name = f"{data_set}_{metric}_monthly.png"
    # else:
    #     file_name = f"{data_set}_{metric}.png"

    # plt.savefig(
    #     os.path.expanduser(
    #         f"~/Dropbox/plrt-conus-figures/basin_performance/{file_name}"
    #     )
    # )


def plot_basin_mean_performance_dset_tile(
    model_results,
    metric="NNSE",
):
    res_huc2 = load_feather(config.get_dir("spatial_data") / "updated_res_huc2.feather")
    res_huc2 = res_huc2.set_index("res_id")

    # get name of wbd files
    wbds = get_contiguous_wbds()
    wbd_ids = [re.search(r"WBD_(\d\d)_HU2", i).group(1) for i in wbds]
    wbd_map = {int(i): wbd for i, wbd in zip(wbd_ids, wbds)}

    if metric == "NNSE":
        metric_func = get_nnse
    else:
        metric_func = get_nrmse

    train_scores = metric_func(model_results["train_data"], "actual", "model", "res_id")
    test_scores = metric_func(model_results["test_data"], "actual", "model", "res_id")
    simmed_scores = metric_func(
        model_results["simmed_data"], "actual", "model", "res_id"
    )

    train_resers = train_scores.index
    test_resers = test_scores.index
    train_res_huc2 = res_huc2.loc[train_resers]
    test_res_huc2 = res_huc2.loc[test_resers]

    train_res_huc2[metric] = train_scores
    test_res_huc2[metric] = test_scores

    train_scores = train_res_huc2.groupby("huc2_id").mean()
    test_scores = test_res_huc2.groupby("huc2_id").mean()
    test_res_huc2[metric] = simmed_scores
    simmed_scores = test_res_huc2.groupby("huc2_id").mean()

    train_scores = train_scores[metric]
    test_scores = test_scores[metric]
    simmed_scores = simmed_scores[metric]

    max_score = max([train_scores.max(), test_scores.max(), simmed_scores.max()])
    min_score = min([train_scores.min(), test_scores.min(), simmed_scores.min()])

    score_range = max_score - min_score

    norm = Normalize(
        vmin=max([min_score - score_range * 0.05, 0]),
        vmax=min([max_score + score_range * 0.05, 1]),
    )
    cmap = get_cmap("viridis")
    other_bounds = []
    for scores in [train_scores, test_scores, simmed_scores]:
        temp_bounds = []
        for wbd_id, score in scores.items():
            wbd = wbd_map[int(wbd_id)]
            color = cmap(norm(score))
            temp_bounds.append((wbd, "k", color))
        other_bounds.append(temp_bounds)

    west, south, east, north = (
        -127.441406,
        24.207069,
        -66.093750,
        53.382373,
    )

    fig = plt.figure()
    # gs = mgridspec.GridSpec(
    #     3, 2, figure=fig, width_ratios=[20, 1], height_ratios=[1, 1, 1]
    # )
    # train_ax = fig.add_subplot(gs[0, 0])
    # test_ax = fig.add_subplot(gs[1, 0])
    # simmed_ax = fig.add_subplot(gs[2, 0])
    # cbar_ax = fig.add_subplot(gs[:, 1])
    gs = mgridspec.GridSpec(2, 2, figure=fig, width_ratios=[1, 1], height_ratios=[1, 1])
    train_ax = fig.add_subplot(gs[0, 0])
    test_ax = fig.add_subplot(gs[0, 1])
    simmed_ax = fig.add_subplot(gs[1, 0])
    cbar_ax = fig.add_subplot(gs[1, 1])
    axes = [train_ax, test_ax, simmed_ax]
    maps = [
        setup_map(
            coords=[west, south, east, north],
            other_bound=ob,
            ax=ax,
            # label_positions=lp,
            return_ticks=True,
        )
        for ax, ob in zip(axes, other_bounds)
    ]
    for ax, title in zip(axes, ["Train", "Test", "Simmed"]):
        ax.set_title(title)

    mvals, pvals = maps[0][1:]
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

    plt.colorbar(
        ScalarMappable(norm=norm, cmap=cmap),
        cax=cbar_ax,
        orientation="horizontal",
        label=f"Basin Average {metric}",
        aspect=100,
        shrink=0.8,
        fraction=0.05,
    )
    # fig.suptitle(f"{data_set.title()}ing Reservoirs")
    plt.subplots_adjust(
        top=0.96, bottom=0.068, left=0.02, right=0.98, hspace=0.15, wspace=0.066
    )
    plt.show()
    # if monthly:
    #     file_name = f"{data_set}_{metric}_monthly.png"
    # else:
    #     file_name = f"{data_set}_{metric}.png"

    # plt.savefig(
    #     os.path.expanduser(
    #         f"~/Dropbox/plrt-conus-figures/basin_performance/{file_name}"
    #     )
    # )


def new_plot_leave_out_performance_comparisons(
    model_path, axes=None, label_x=True, label_y="left", legend="left"
):
    """Plot the experimental NNSE for Training and Testing reservoirs against
    the base model NNSE for Training and Testing reservoirs.

    Args:
        model_path (str | pathlib.Path): Path to model results.
        axes (list, optional): Axes to plot on. Defaults to None.
        label_x (bool, optional): Draw X labels. Defaults to True.
        label_y (str | bool, optional): How to label y axis. Defaults to "left".
        legend (str | bool, optional): How to draw legend. Defaults to "left".

    Raises:
        ValueError: Model path does not have a meta variable
    """
    pattern = re.compile(r"meta_(.*)_-?\d.\d+")
    search_result = re.search(pattern, model_path)
    if not search_result:
        raise ValueError("Model path does not have a meta variable")
    meta_var = search_result.group(1)
    models_dir = config.get_dir("results") / "monthly_merged_data_set_minyr3"
    model_paths = glob.glob(f"{models_dir.as_posix()}/*{meta_var}*")

    lower_20, upper_20 = "", ""
    for model_path in model_paths:
        if model_path[-4:] == "-0.2":
            lower_20 = model_path
        elif model_path[-3:] == "0.8":
            upper_20 = model_path

    lower_20 = load_model_results(lower_20)
    upper_20 = load_model_results(upper_20)
    base_results = load_model_results(
        config.get_dir("results")
        / "monthly_merged_data_set_minyr3"
        / "TD6_MSS0.03_SM_basin_0.8"
    )

    lower_20_train = lower_20["train_data"]
    lower_20_test = lower_20["test_data"]
    lower_20_simmed = lower_20["simmed_data"]
    upper_20_train = upper_20["train_data"]
    upper_20_test = upper_20["test_data"]
    upper_20_simmed = upper_20["simmed_data"]
    base_simul = base_results["simmed_data"]

    lower_20_test_resers = lower_20_test.index.get_level_values(0).unique()
    upper_20_test_resers = upper_20_test.index.get_level_values(0).unique()
    lower_20_train_resers = lower_20_train.index.get_level_values(0).unique()
    upper_20_train_resers = upper_20_train.index.get_level_values(0).unique()

    # select simulation records from experimental sets that correspond
    # to training and testing reservoirs.
    upper_train_df = upper_20_simmed.loc[
        pd.IndexSlice[upper_20_train_resers, :], :
    ].copy()
    upper_test_df = upper_20_simmed.loc[
        pd.IndexSlice[upper_20_test_resers, :], :
    ].copy()
    lower_train_df = lower_20_simmed.loc[
        pd.IndexSlice[lower_20_train_resers, :], :
    ].copy()
    lower_test_df = lower_20_simmed.loc[
        pd.IndexSlice[lower_20_test_resers, :], :
    ].copy()

    # calculate the base model simul NNSE
    base_simul_scores = get_nnse(base_simul, "actual", "model", "res_id")
    base_simul_scores.name = "base"
    base_simul_scores = base_simul_scores.to_frame()

    # calculate the experimental simul NNSE and add base nnse to the frame
    experiment_dfs = []
    for df in [upper_train_df, upper_test_df, lower_train_df, lower_test_df]:
        scores = get_nnse(df, "actual", "model", "res_id")
        scores.name = "experimental"
        scores = scores.to_frame()
        scores["base"] = base_simul_scores.loc[scores.index, "base"]
        experiment_dfs.append(scores)

    show = False
    if axes is None:
        _, axes = plt.subplots(1, 2, sharex=True, sharey=True)
        axes = axes.flatten()
        show = True

    pretty_var = get_pretty_var_name(meta_var)
    titles = [f"Upper 20% {pretty_var}", f"Lower 20% {pretty_var}"]
    for i, (ax, dfs, title) in enumerate(
        zip(axes, [experiment_dfs[:2], experiment_dfs[2:]], titles)
    ):
        train_df, test_df = dfs
        ax.scatter(
            train_df["base"],
            train_df["experimental"],
            label="Training Reservoirs",
            s=30,
        )
        ax.scatter(
            test_df["base"],
            test_df["experimental"],
            label="Testing Reservoirs",
            s=30,
        )
        legend_prop = {"size": 12}
        label_size = 14
        if i == 0 and legend == "left":
            ax.legend(loc="upper left", prop=legend_prop)
        if i == 1 and legend == "right":
            ax.legend(loc="upper left", prop=legend_prop)
        if legend is True:
            ax.legend(loc="upper left", prop=legend_prop)
        if label_x:
            ax.set_xlabel("NNSE (Base Model)", fontsize=label_size)

        if i == 0 and label_y == "left":
            ax.set_ylabel("NNSE (Excluded)", fontsize=label_size)
        if i == 1 and label_y == "right":
            ax.set_ylabel("NNSE (Excluded)", fontsize=label_size)
        if label_y is True:
            ax.set_ylabel("NNSE (Excluded)", fontsize=label_size)

        ax.set_title(title, fontsize=label_size)
        ax.set_xlim((-0.05, 1.05))
        ax.set_ylim((-0.05, 1.05))
        ax.tick_params(axis="both", which="major", labelsize=12)
        mxbline(m=1, b=0, ax=ax, color="k", ls="--", lw=1)

    if show:
        plt.show()


def plot_all_leave_out_performance_comparisons():
    model_paths = [
        "TD6_MSS0.03_SM_meta_rts_0.8",
        "TD6_MSS0.03_SM_meta_max_sto_0.8",
        "TD6_MSS0.03_SM_meta_rel_inf_corr_0.8",
    ]

    fig, axes = plt.subplots(3, 2)
    label_args = [
        {"label_x": False, "label_y": "left", "legend": "left"},
        {"label_x": False, "label_y": "left", "legend": False},
        {"label_x": True, "label_y": "left", "legend": False},
    ]

    for model_path, ax_row, kwargs in zip(model_paths, axes, label_args):
        new_plot_leave_out_performance_comparisons(model_path, ax_row, **kwargs)

    plt.show()


if __name__ == "__main__":
    # sns.set_theme(context="notebook", palette="Set2", font_scale=1.1)
    # sns.set_context("poster", font_scale=1.1)
    plt.style.use(["science", "nature"])
    args, remaining = parse_args()
    func_args = parse_unknown_args(remaining)

    if args.model_path:
        model_path = args.model_path
    else:
        model_path = "TD6_MSS0.03_SM_basin_0.8"

    model_results = load_model_results(
        config.get_dir("results") / "monthly_merged_data_set_minyr3" / model_path
    )
    # plot_basin_tree_breakdown_comparison(args.basins, model_results)
    # from itertools import combinations
    # basins = range(1, 19)
    # basin_pairs = combinations(basins, 2)
    # for b1, b2 in basin_pairs:
    # plot_seasonal_tree_breakdown_basin_comparison([b1, b2], model_results)
    # plot_seasonal_tree_breakdown_basin_comparison(args.basins, model_results, "box")
    # plot_basin_comparison_map(args.basins[0])
    # plot_grouped_basin_map()
    # find_similar_basins()

    # find_similar_reservoir_characteristics(model_results)

    # plot_reservoir_group_access_map()
    # plot_reservoir_most_likely_group_maps(model_results)

    # train_metrics, test_metrics = rank_reservoirs_by_performance(
    #     model_results,
    #     monthly_group=False,
    #     plot_pairs=False
    # )
    # plot_basin_mean_performance(model_results, **func_args)
    # plot_basin_mean_performance_dset_tile(model_results, **func_args)
    # plot_leave_out_performance_comparisons(model_path)
    plot_all_leave_out_performance_comparisons()
