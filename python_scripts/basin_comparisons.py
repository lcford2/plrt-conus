import argparse
import calendar
import os
import re
from collections import defaultdict
from itertools import combinations
from multiprocessing import cpu_count

import geopandas as gpd
import matplotlib.gridspec as mgridspec
import matplotlib.patches as mpatch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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
from utils.utils import sorted_k_partitions

CPUS = cpu_count()
os.environ["OMP_NUM_THREADS"] = str(CPUS)

BASIN_GROUPS_ORIG = {
    "Most Similar": [10, 11, 14, 16, 17, 18],
    "Sort-of Similar": [3, 5, 6, 7, 8],
    "Least Similar": [1, 2, 9, 12, 13, 15],
}

BASIN_GROUPS = {
    "Most Similar": [10, 14, 16, 17, 18],
    "Sort-of Similar": [3, 5, 6, 7, 11],
    "Least Similar": [1, 2, 8, 9, 12, 13, 15],
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
    return parser.parse_args()


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

    wbds = get_contiguous_wbds()
    wbd_ids = [re.search(r"WBD_(\d\d)_HU2", i).group(1) for i in wbds]
    wbd_map = {int(i): wbd for i, wbd in zip(wbd_ids, wbds)}

    # norm = Normalize(vmin=0, vmax=2)
    # cmap = get_cmap("plasma_r")
    # parts = load_pickle(config.get_dir("agg_results") / "best_partitions_3.pickle")
    # thresh = 3
    # filtered_parts = [i for i in parts if i[-1] == thresh]

    # best_part = filtered_parts[0][0]

    color_pal = sns.color_palette("Set2")
    color_dict = {
        tuple(item): color_pal[i]
        for i, (k, item) in enumerate(BASIN_GROUPS.items())
        # for i, item in enumerate(best_part)
    }

    color_vars = {}
    for wbd_id in wbd_ids:
        for gbasins, color in color_dict.items():
            if int(wbd_id) in gbasins:
                color_vars[int(wbd_id)] = color

    other_bounds = [(wbd_map[i], "k", color_vars[i]) for i in color_vars.keys()]

    west, south, east, north = (
        -127.441406,
        24.207069,
        -66.093750,
        53.382373,
    )
    setup_map(coords=[west, south, east, north], other_bound=other_bounds)
    # wbd_gdfs = [gpd.read_file(i+".shp") for i in wbds]
    # for wbd_id, wbd_gdf in zip(wbd_ids, wbd_gdfs):
    #     centroid = wbd_gdf.centroid[0]
    #     x, y = centroid.x, centroid.y
    #     print(x, y)
    #     ax.text(x, y, wbd_id)
    handles = [mpatch.Patch(edgecolor="k", facecolor=color_pal[i]) for i in range(3)]
    # labels = [str(i) for i in best_part]
    labels = BASIN_GROUPS.keys()
    ax = plt.gca()
    ax.legend(handles, labels, loc="best")
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

    wbds = get_contiguous_wbds()
    wbd_ids = [re.search(r"WBD_(\d\d)_HU2", i).group(1) for i in wbds]
    wbd_map = {int(i): wbd for i, wbd in zip(wbd_ids, wbds)}

    res_access = pd.read_csv(
        "../aggregated_results/res_group_access.csv", index_col=0, dtype=str
    )
    grand = gpd.read_file(config.get_dir("spatial_data") / "my_grand_info")

    groups = res_access.columns
    color_pal = sns.color_palette("Set2")
    group_colors = {group: color_pal[i] for i, group in enumerate(groups)}

    color_dict = {
        tuple(item): color_pal[i] for i, (k, item) in enumerate(BASIN_GROUPS.items())
    }

    color_vars = {}
    for wbd_id in wbd_ids:
        for gbasins, color in color_dict.items():
            if int(wbd_id) in gbasins:
                color_vars[int(wbd_id)] = color

    other_bounds = [(wbd_map[i], "k", "w") for i in color_vars.keys()]

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
    color_pal = sns.color_palette("Set3")
    norm = Normalize(vmin=1, vmax=10)
    cmap = ListedColormap(color_pal.as_hex()[:10])
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
        # ScalarMappable(norm=norm, cmap=cmap),
        ScalarMappable(
            norm=Normalize(vmin=1, vmax=10),
            # cmap=ListedColormap(color_pal.as_hex()[:10]),
            cmap=cmap,
        ),
        cax=cbar_ax,
        orientation="vertical",
        label="Most Likely Operational Mode",
        aspect=4,
        shrink=0.8,
    )
    plt.show()


if __name__ == "__main__":
    sns.set_theme(context="notebook", palette="Set2", font_scale=1.2)
    args = parse_args()

    if args.model_path:
        model_path = args.model_path
    else:
        model_path = "TD4_MSS0.09"

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
    plot_reservoir_most_likely_group_maps(model_results)
