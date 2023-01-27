import pathlib

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import utils.metrics as metric_funcs
from fit_plrt_model import load_resopsus_data
from mpl_toolkits.basemap import Basemap
from utils.config import config
from utils.io import load_feather, load_pickle, load_results, write_pickle
from utils.plot_tools import get_pretty_var_name

PSWEEP_RESULTS_DIR = config.get_dir("results") / "parameter_sweep"
GIS_DIR = config.get_dir("general_data") / "GIS"


def load_grand_db() -> gpd.GeoDataFrame:
    """Load the GRanD database

    Returns:
        gpd.GeoDataFrame: GeoDataFrame of GRanD DB
    """
    df = gpd.read_file(
        (config.get_dir("spatial_data") / "my_grand_info").as_posix()
    )
    return df.set_index("GRAND_ID")


def load_model_results(model_dir: str | pathlib.Path) -> dict:
    """Load PLRT Model results

    Args:
        model_dir (str | pathlib.Path): Directory containing model results.

    Returns:
        _type_: _description_
    """
    if isinstance(model_dir, pathlib.Path):
        model_dir = model_dir.as_posix()

    return {model_dir.name: load_results(model_dir.as_posix())}


def load_model_results_from_list(model_dirs: list) -> dict:
    """Load PLRT results for each item in model_dirs

    Args:
        model_dirs (list): List of directories to load results for

    Returns:
        dict: Keys are the name of the directory for each directory model_dirs.
            Values are the dictionaries returned from `load_model_results`
    """
    return {pathlib.Path(d).name: load_model_results(d) for d in model_dirs}


def get_data_from_results(results: dict, dataset="simmed") -> pd.DataFrame:
    """Extract a particular dataset from results

    Args:
        results (dict): results dict (e.g. output from
            `load_model_results_from_list`)
        dataset (str, optional): Dataset to extract. Defaults to "simmed".

    Raises:
        ValueError: If dataset is not in [train, test, simmed]

    Returns:
        pd.DataFrame: Dataset requested, where the "model" column is renamed
            to the name of the model (keys in the results dict)
    """
    available_data = ["train", "test", "simmed"]
    if dataset not in available_data:
        raise ValueError(f"{dataset} must be in {available_data}")

    output = pd.DataFrame()
    for model, mresults in results.items():
        data = mresults[f"{dataset}_data"]
        data = data.rename(columns={"model": model})
        if output.empty:
            output = data
        else:
            output[model] = data[model]
    return output


def calculate_metrics(
    data: pd.DataFrame, data_set: str, metrics=("nnse", "rmse"), recalc=False
) -> dict:
    """Calculate model performance metrics on the data provided

    Args:
        data (pd.DataFrame): Data with an `actual` column and other
            model columns
        data_set (str): name of data set that data refers to
        metrics (tuple, optional): Metrics to calculate.
            Defaults to ("nnse", "rmse").
        recalc (bool, optional): Force recalculation even if cached version
            exists. Defaults to False.

    Returns:
        dict: keys are `metrics`, values are metric data frames where each
            column refers to a model
    """
    file_name = "_".join([data_set, *metrics])
    metrics_file = (
        config.get_dir("agg_results")
        / "parameter_sweep"
        / f"{file_name}.pickle"
    )
    if not recalc and metrics_file.exists():
        return load_pickle(metrics_file.as_posix())
    models = list(data.drop("actual", axis=1).columns)
    models = sorted(
        models,
        key=lambda x: (int(x.split("_")[0][2:]), float(x.split("_")[1][3:])),
    )

    output = {}
    for metric in metrics:
        f_metric = getattr(metric_funcs, f"get_{metric.lower()}")
        metric_df = pd.DataFrame()
        for model in models:
            model_df = f_metric(data, "actual", model, grouper="res_id")
            model_df.name = model

            if metric_df.empty:
                metric_df = model_df.to_frame()
            else:
                metric_df[model] = model_df
        output[metric] = metric_df

    write_pickle(output, metrics_file.as_posix())
    return output


def metric_wide_to_long(metric_df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Convert a dataframe of metrics from wide to long

    Args:
        metric_df (pd.DataFrame): Wide dataframe to convert
        metric (str): Metric name, will become name of column holding
            metric values

    Returns:
        pd.DataFrame: Long metric dataframe
    """
    df = metric_df.melt(var_name="model", value_name=metric)
    df[["TD", "MSS"]] = df["model"].str.split("_", expand=True)
    df["TD"] = df["TD"].str.slice(2)
    df["MSS"] = df["MSS"].str.slice(3)
    return df.drop("model", axis=1)


def plot_metric_box_plot(metric_df: pd.DataFrame, metric: str) -> None:
    """Box plot where X is tree depth, y is the metric, boxes are colored by
        MSS, and whiskers are set to 10% and 90%

    Args:
        metric_df (pd.DataFrame): Metric dataframe to plot
        metric (str): Name of metric to plot
    """
    df = metric_wide_to_long(metric_df, metric)
    fg = sns.catplot(
        data=df,
        x="TD",
        y=metric,
        hue="MSS",
        kind="box",
        whis=(10, 90),
        legend_out=False,
        showfliers=True,
        palette="Set2",
    )
    ax = fg.ax
    ax.legend(title="MSS", loc="lower left", ncol=5)
    plt.show()


def plot_single_model_metrics(df: pd.DataFrame) -> None:
    """Model metrics (nnse, nrmse) for testing reservoirs.

    Args:
        df (pd.DataFrame): Dataframe containing model data.
    """
    grand_names = load_grand_db()
    get_nnse = getattr(metric_funcs, "get_nnse")
    get_nrmse = getattr(metric_funcs, "get_nrmse")

    test_nnse = get_nnse(df, "actual", "test", grouper="res_id")
    test_nrmse = get_nrmse(df, "actual", "test", grouper="res_id")
    simmed_nnse = get_nnse(df, "actual", "simmed", grouper="res_id")
    simmed_nrmse = get_nrmse(df, "actual", "simmed", grouper="res_id")

    nnse = pd.DataFrame.from_dict({"test": test_nnse, "simmed": simmed_nnse})
    nrmse = pd.DataFrame.from_dict({"test": test_nrmse, "simmed": simmed_nrmse})

    nnse = nnse.reset_index().melt(id_vars="res_id")
    nrmse = nrmse.reset_index().melt(id_vars="res_id")
    nnse["metric"] = "nnse"
    nrmse["metric"] = "nrmse"

    metrics = pd.concat([nnse, nrmse])
    metrics["res_name"] = metrics["res_id"].apply(
        lambda x: grand_names.loc[x, "DAM_NAME"]
    )
    metrics = metrics.sort_values(by=["metric", "value"])

    metrics["variable"] = metrics["variable"].replace(
        {"test": "Testing", "simmed": "Simulation"}
    )

    fg = sns.catplot(
        data=metrics,
        x="res_name",
        y="value",
        row="metric",
        hue="variable",
        kind="bar",
        legend=False,
        sharey=False,
    )
    axes = fg.axes.flatten()
    axes[1].legend(loc="best")
    axes[1].tick_params(axis="x", labelrotation=90)

    axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=60, ha="right")
    fg.set_titles("")
    axes[0].set_ylabel("NNSE")
    axes[1].set_ylabel("NRMSE")
    axes[1].set_xlabel("")

    fg.figure.align_ylabels()

    plt.show()


def compare_training_testing_data(results: dict, min_years: int) -> None:
    """Plot ECDF for residence time, maximum storage, release CV
    release_pre, inflow, and storage_pre for both the training and
    testing sets.

    Args:
        results (dict): Dictionary containing results (from load_model_results)
        min_years (int): minimum number of years that was used to create
            the data set of interest
    """
    mr_data, meta = load_resopsus_data(min_years)

    test_df = get_data_from_results(results, dataset="test")

    test_res = test_df.index.get_level_values("res_id").unique()

    meta["Data Set"] = "Train"
    meta.loc[test_res, "Data Set"] = "Test"
    meta_melt = meta.melt(id_vars=["Data Set"])

    mr_data["Data Set"] = "Train"
    mr_data.loc[pd.IndexSlice[test_res, :], "Data Set"] = "Test"
    mr_data = mr_data[["release_pre", "inflow", "storage_pre", "Data Set"]]
    mr_data = mr_data.melt(id_vars=["Data Set"])

    meta_melt = pd.concat([meta_melt, mr_data])

    meta_melt["variable"] = meta_melt["variable"].apply(
        lambda x: get_pretty_var_name(x, math=True)
    )
    fg = sns.displot(
        data=meta_melt,
        x="value",
        col="variable",
        hue="Data Set",
        kind="ecdf",
        col_wrap=3,
        facet_kws={"sharex": False, "sharey": False, "legend_out": False},
    )
    fg.set_titles("{col_name}")
    plt.show()


def setup_map(ax=None, coords=None, other_bound=None) -> Basemap:
    """Generate a map with many common elements

    Args:
        ax (plt.axes, optional): Axes to plot on, if none, the current
            axis is used. Defaults to None.
        coords (tuple, optional): Bounding box for the map, if None will
            use (-127.4414, 24.2071, -66.0938, 49.3824). Defaults to None.
        other_bound (list, optional): Other bounds to add.
            [(path to shapefile, color)...]. Defaults to None.

    Returns:
        Basemap: basemap instance that can be used to modify the map
    """
    if not ax:
        ax = plt.gca()

    if coords:
        west, south, east, north = coords
    else:
        west, south, east, north = (
            -127.441406,
            24.207069,
            -66.093750,
            49.382373,
        )
    m = Basemap(
        # projection="merc",
        epsg=3857,
        resolution="c",
        llcrnrlon=west,
        llcrnrlat=south,
        urcrnrlon=east,
        urcrnrlat=north,
        ax=ax,
    )

    m.drawmapboundary(fill_color="white")
    # states_path = GIS_DIR / "cb_2017_us_state_500k"
    # states = m.readshapefile(states_path.as_posix(), "states")
    # rivers = m.readshapefile(
    #     (GENERAL_DATA_DIR / "rivers" / "rivers_subset").as_posix(),
    #     "rivers",
    #     color="b",
    #     linewidth=0.5,
    #     zorder=3
    # )

    parallels = np.arange(0.0, 81, 10.0)
    meridians = np.arange(10.0, 351.0, 20.0)
    pvals = m.drawparallels(
        parallels,
        linewidth=1.0,
        dashes=[1, 0],
        labels=[1, 1, 1, 1],
        zorder=-1,
    )
    mvals = m.drawmeridians(
        meridians, linewidth=1.0, dashes=[1, 0], labels=[1, 1, 1, 1], zorder=-1
    )
    xticks = [i[1][0].get_position()[0] for i in mvals.values()]
    yticks = []
    for i in pvals.values():
        try:
            yticks.append(i[1][0].get_position()[1])
        except IndexError:
            pass

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

    if other_bound:
        for b, c in other_bound:
            bound = m.readshapefile(
                b,
                "bound",
                # color="#FF3BC6"
                color=c,
            )
            # bound[4].set_facecolor("#FF3BC6")
            bound[4].set_facecolor("w")
            bound[4].set_alpha(1)
            bound[4].set_zorder(2)
    return m


def get_contiguous_wbds():
    """Get path to shapefiles to all contiguous watershed boundaries

    Returns:
        list: Paths to shapefiles
    """
    WBD_DIR = GIS_DIR / "WBD"
    file = "WBD_{:02}_HU2_Shape/Shape/WBDHU2"
    bounds_files = [(WBD_DIR / file.format(i)).as_posix() for i in range(1, 19)]
    return bounds_files


def setup_wbd_map():
    """Setup map with watershed boundaries

    Returns:
        tuple: figure, axes, Basemap
    """
    fig, ax = plt.subplots(1, 1)
    wbds = get_contiguous_wbds()

    other_bounds = [(b, "k") for b in wbds]

    west, south, east, north = (
        -127.441406,
        24.207069,
        -66.093750,
        53.382373,
    )
    m = setup_map(
        ax=ax, coords=[west, south, east, north], other_bound=other_bounds
    )
    return fig, ax, m


def plot_training_testing_map(results, min_years):
    fig, ax, m = setup_wbd_map()
    grand = gpd.read_file(config.get_dir("spatial_data") / "my_grand_info")
    # big_grand = gpd.read_file(
    #     config.get_dir("general_data")
    #     / "GRanD_Version_1_3"
    #     / "GRanD_reservoirs_v1_3.shp"
    # )
    big_grand = gpd.read_file(config.get_file("grand_file"))

    big_grand["GRAND_ID"] = big_grand["GRAND_ID"].astype(str)

    test_df = get_data_from_results(results, dataset="test")
    train_df = get_data_from_results(results, dataset="train")
    all_resops = load_feather(config.get_file("resops_agg"))

    test_res = test_df.index.get_level_values("res_id").unique()
    train_res = train_df.index.get_level_values("res_id").unique()
    all_res = all_resops["res_id"].unique().astype(str)
    merged_data, merged_meta = load_resopsus_data(min_years)
    all_res = list(set([*all_res, *merged_meta.index]))

    left_out_res = [
        i for i in all_res if i not in test_res and i not in train_res
    ]

    test_coords = [
        (row.LONG_DD, row.LAT_DD)
        for i, row in grand[grand["GRAND_ID"].isin(test_res)].iterrows()
    ]
    train_coords = [
        (row.LONG_DD, row.LAT_DD)
        for i, row in grand[grand["GRAND_ID"].isin(train_res)].iterrows()
    ]
    left_out_coords = [
        (row.LONG_DD, row.LAT_DD)
        for i, row in big_grand[
            big_grand["GRAND_ID"].isin(left_out_res)
        ].iterrows()
    ]

    train_x, train_y = list(zip(*train_coords))
    test_x, test_y = list(zip(*test_coords))
    left_out_x, left_out_y = list(zip(*left_out_coords))
    print("Train #:", len(train_x))
    print("Test #:", len(test_x))
    print("Left Out #:", len(left_out_x))

    m.scatter(
        train_x, train_y, latlon=True, label="Training", marker="v", zorder=4
    )
    m.scatter(
        test_x, test_y, latlon=True, label="Testing", marker="v", zorder=4
    )
    # m.scatter(
    #     left_out_x,
    #     left_out_y,
    #     latlon=True,
    #     label="Excluded",
    #     marker="v",
    #     zorder=3,
    #     alpha=0.8,
    # )

    ax.legend(loc="lower left")

    plt.show()


def plot_data_diff_map(results, year1, year2):
    fig, ax, m = setup_wbd_map()
    grand = gpd.read_file(config.get_dir("spatial_data") / "my_grand_info")
    big_grand = gpd.read_file(config.get_file("grand_file"))
    big_grand["GRAND_ID"] = big_grand["GRAND_ID"].astype(str)

    yr1_data, yr1_meta = load_resopsus_data(year1)
    yr2_data, yr2_meta = load_resopsus_data(year2)

    grand = grand.set_index("GRAND_ID")

    yr1_coords = grand.loc[
        yr1_meta.index, ["LONG_DD", "LAT_DD"]
    ].values.tolist()
    yr2_coords = grand.loc[
        yr2_meta.index, ["LONG_DD", "LAT_DD"]
    ].values.tolist()

    yr1_x, yr1_y = list(zip(*yr1_coords))
    yr2_x, yr2_y = list(zip(*yr2_coords))

    yr1_z = 4
    yr2_z = 5
    if len(yr1_x) < len(yr2_x):
        yr1_z = 4
        yr2_z = 5

    m.scatter(
        yr1_x,
        yr1_y,
        latlon=True,
        label=f"Min Years={year1}",
        marker="v",
        zorder=yr1_z,
    )
    m.scatter(
        yr2_x,
        yr2_y,
        latlon=True,
        label=f"Min Years={year2}",
        marker="v",
        zorder=yr2_z,
    )

    ax.legend(loc="lower left")

    plt.show()


if __name__ == "__main__":
    # plt.style.use("ggplot")
    sns.set_theme(context="talk", palette="Set2")
    # results = load_model_results()
    # simmed_data = get_data_from_results(results, dataset="simmed")
    # metrics = calculate_metrics(simmed_data, data_set="simmed", recalc=False)
    # nse, rmse = metrics["nse"], metrics["nrmse"]
    # plot_metric_box_plot(nse, "NSE")

    import sys

    if len(sys.argv) > 1:
        min_years = sys.argv[1]
    else:
        min_years = 3

    model = "TD3_MSS0.04"
    results = load_model_results(
        config.get_dir("results")
        / f"monthly_merged_data_set_minyr{min_years}"
        / model
    )
    df = get_data_from_results(results, dataset="simmed")
    df = df.rename(columns={model: "simmed"})
    df["test"] = get_data_from_results(results, dataset="test")[model]
    plot_single_model_metrics(df)

    # compare_training_testing_data(results, int(min_years))
    # plot_training_testing_map(results, min_years)
