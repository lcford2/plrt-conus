import argparse
import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from parameter_sweep_analysis import load_model_results
from single_tree_breakdown import get_groups_for_model
from utils.config import config
from utils.io import load_huc2_basins, load_huc2_name_map


def parse_args():
    parser = argparse.ArgumentParser(description="Compare basin results")
    parser.add_argument(
        "-b",
        "--basins",
        nargs=2,
        type=int,
        help="Which two basins should  be compared",
        required=True,
    )
    parser.add_argument(
        "-m",
        "--model-path",
        dest="model_path",
        help="The model results that should be plotted",
        required=True,
    )
    return parser.parse_args()


def plot_basin_tree_breakdown_comparison(basins, results):
    huc2 = load_huc2_basins()
    groups = get_groups_for_model(results)
    groups.name = "group"
    groups = groups.to_frame()
    groups["basin"] = [
        huc2.loc[i, "huc2_id"] for i in groups.index.get_level_values(0)
    ]
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


def plot_seasonal_tree_breakdown_basin_comparison(basins, results):
    huc2 = load_huc2_basins()
    huc2_names = load_huc2_name_map()
    groups = get_groups_for_model(results)
    groups.name = "group"
    groups = groups.to_frame()
    # groups["basin"] = [
    #     huc2.loc[i, "huc2_id"] for i in groups.index.get_level_values(0)
    # ]
    # groups = groups[groups["basin"].isin(basins)]
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
    from itertools import combinations

    basins = range(1, 19)
    basin_pairs = combinations(basins, 2)
    for b1, b2 in basin_pairs:
        print(f"Making plot for basins: {b1}, {b2}")
        bprops = props.loc[idx[[huc2_names[b1], huc2_names[b2]], :, :]]

        fg = sns.catplot(
            data=bprops.reset_index(),
            x="month",
            y="prop",
            hue="basin",
            col="group",
            col_wrap=5,
            kind="bar",
            legend_out=False,
            height=5,
            aspect=0.7,
        )

        fg.set_ylabels("Group Occ. [%]")
        fg.set_xlabels("Month")
        fg.set_titles("Tree Node: {col_name}")

        plt.subplots_adjust(
            top=0.963,
            bottom=0.064,
            left=0.039,
            right=0.991,
            hspace=0.106,
            wspace=0.051,
        )
        output_dir = os.path.expanduser(
            "~/Dropbox/plrt-conus-figures/basin_comparison"
        )
        output_file = f"monthly_basin_compare_{'-'.join(map(str, basins))}.png"
        plt.savefig("/".join([output_dir, output_file]))


if __name__ == "__main__":
    sns.set_theme(context="notebook", palette="Set2")
    args = parse_args()

    model_results = load_model_results(
        config.get_dir("results")
        / "monthly_merged_data_set_minyr3"
        / args.model_path
    )
    # plot_basin_tree_breakdown_comparison(args.basins, model_results)
    # from itertools import combinations
    # basins = range(1, 19)
    # basin_pairs = combinations(basins, 2)
    # for b1, b2 in basin_pairs:
    # plot_seasonal_tree_breakdown_basin_comparison([b1, b2], model_results)
    plot_seasonal_tree_breakdown_basin_comparison(args.basins, model_results)
