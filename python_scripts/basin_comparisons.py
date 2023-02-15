import argparse

import matplotlib.pyplot as plt
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
    )
    parser.add_argument(
        "-m",
        "--model-path",
        dest="model_path",
        help="The model results that should be plotted",
    )
    return parser.parse_args()


def plot_basin_tree_breakdown_comparison(basins, results, bar_plot=False):
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


if __name__ == "__main__":
    sns.set_theme(context="notebook", palette="Set2")
    args = parse_args()

    model_results = load_model_results(
        config.get_dir("results")
        / "monthly_merged_data_set_minyr3"
        / args.model_path
    )
    plot_basin_tree_breakdown_comparison(args.basins, model_results)
