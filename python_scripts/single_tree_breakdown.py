# import matplotlib.gridspec as GS
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from parameter_sweep_analysis import load_model_results_from_list
from utils.config import config
from utils.io import load_feather


def get_groups_for_model(model: dict) -> pd.Series:
    """Get tree groups for model

    Args:
        model (dict): Results dictionary for model

    Returns:
        pd.Series: Pandas Series of tree groups
    """
    return model["groups"]


def plot_tree_breakdown_ecdf(groups: pd.Series) -> None:
    """Plot Empirical CDF for each the percentage of time each reservoir
    spends in each node of a tree

    Args:
        groups (pd.Series): Series of groups for each time step for each
            reservoir
    """
    groups.name = "tree_group"
    groups = groups.reset_index()

    counts = groups.groupby("tree_group")["res_id"].value_counts().unstack()
    counts = counts.fillna(0.0)

    percents = counts.divide(counts.sum(axis=0), axis=1) * 100
    percents = percents.stack()
    percents.name = "percent"

    sns.displot(
        data=percents,
        x="percent",
        col="tree_group",
        col_wrap=4,
        kind="ecdf",
    )
    plt.show()


def plot_tree_basin_breakdown(groups: pd.Series) -> None:
    """Plot a bar chart indicating the average percent of time each reservoir
    spends in a particular tree node for each basin.

    Args:
        groups (pd.Series): Series of groups for each timestep for each
            reservoir
    """
    huc2 = load_feather(
        config.get_dir("spatial_data") / "updated_res_huc2.feather",
        index_keys=("res_id",),
    )
    huc2_names = {}
    with open(config.get_dir("spatial_data") / "huc2_names.csv", "r") as f:
        for line in f.readlines():
            split = line.strip().split(",")
            huc2_names[int(split[0])] = f"{split[0]}: {split[1]}"

    groups.name = "tree_group"
    groups = groups.reset_index()
    groups["basin"] = [huc2.loc[res, "huc2_id"] for res in groups["res_id"]]

    counts = groups.groupby("tree_group")["basin"].value_counts().unstack()
    counts = counts.fillna(0.0)
    counts = counts.rename(columns=huc2_names)

    percents = counts.divide(counts.sum(axis=0), axis=1) * 100
    percents = percents.stack()
    percents.name = "percent"
    percents = percents.reset_index()

    fg = sns.catplot(
        data=percents,
        y="percent",
        x="tree_group",
        col="basin",
        # x="basin",
        # col="tree_group",
        col_wrap=6,
        kind="bar",
        legend_out=False,
    )
    fg.set_titles("{col_name}")
    fg.set_xlabels("Tree Node")
    fg.set_ylabels("Percent")
    plt.show()


def plot_tree_basin_monthly_breakdown(groups: pd.Series) -> None:
    """Plot monthly percentages for each basin and each tree group

    Args:
        groups (pd.Series): _description_
    """
    huc2 = load_feather(
        config.get_dir("spatial_data") / "updated_res_huc2.feather",
        index_keys=("res_id",),
    )
    huc2_names = {}
    with open(config.get_dir("spatial_data") / "huc2_names.csv", "r") as f:
        for line in f.readlines():
            split = line.strip().split(",")
            huc2_names[int(split[0])] = f"{split[0]}: {split[1]}"

    groups.name = "tree_group"
    groups = groups.reset_index()
    groups["basin"] = [huc2.loc[res, "huc2_id"] for res in groups["res_id"]]
    groups["month"] = groups["date"].dt.month

    counts = groups.groupby("tree_group")[["month", "basin"]].value_counts()
    counts = counts.fillna(0.0).reset_index()
    counts["basin"] = counts["basin"].replace(huc2_names)
    counts = counts.pivot(index=["tree_group", "month"], columns="basin")
    counts.columns = counts.columns.droplevel(0)

    sums = counts.groupby("month").sum()
    percents = pd.DataFrame(index=counts.index, columns=counts.columns)

    idx = pd.IndexSlice
    for month in range(1, 13):
        percents.loc[idx[:, month], :] = (
            counts.loc[idx[:, month], :] / sums.loc[month] * 100
        )

    percents = percents.fillna(0.0)
    percents = percents.stack()
    percents.name = "percent"
    percents = percents.reset_index()
    # from IPython import embed as II
    # II()

    fig, axes = plt.subplots(3, 5, sharex=True, sharey=False)
    axes = axes.flatten()
    # fig = plt.figure(constrained_layout=True)
    # gs = GS.GridSpec(
    #     3, 7, figure=fig, width_ratios=[*[10 for _ in range(6)], 2]
    # )
    # axes = []
    # for i in range(3):
    #     for j in range(6):
    #         axes.append(
    #             fig.add_subplot(gs[i, j])
    #         )
    # cbar_ax = fig.add_subplot(gs[:, 6])
    # for ax, basin in zip(axes, huc2_names.values()):
    #     try:
    #         data = percents.loc[:, basin].unstack()
    #         sns.heatmap(data, ax=ax, cbar_ax=cbar_ax, vmin=0, vmax=100)
    #     except KeyError:
    #         pass
    #     ax.set_title(basin)

    # plt.show()
    percents = percents.pivot(index=["month", "basin"], columns=["tree_group"])
    percents.columns = percents.columns.droplevel(0)

    for ax, tg in zip(axes, percents.columns):
        data = percents[tg].unstack()
        for basin in huc2_names.values():
            try:
                basin_data = data[basin]
                ax.plot(basin_data.index, basin_data, label=basin)
            except KeyError:
                pass
        ax.set_title(f"Tree Group: {tg}")
    plt.show()


if __name__ == "__main__":
    sns.set_theme(context="notebook", palette="Set2")
    model_list = ["TD5_MSS0.02", "TD4_MSS0.09"]
    model_dirs = [
        config.get_dir("results") / "monthly_merged_data_set_minyr3" / model
        for model in model_list
    ]
    model_results = load_model_results_from_list(model_dirs)
    model_groups = {
        m: get_groups_for_model(model) for m, model in model_results.items()
    }

    for model in model_list[1:]:
        # plot_tree_breakdown_ecdf(model_groups[model])
        # plot_tree_basin_breakdown(model_groups[model])
        plot_tree_basin_monthly_breakdown(model_groups[model])
