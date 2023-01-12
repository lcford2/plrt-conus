import glob
import pathlib

import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import pandas as pd
import seaborn as sns
import numpy as np
import geopandas as gpd
from IPython import embed as II

from utils.config import PDIRS, FILES, GENERAL_DATA_DIR
from utils.metrics import get_nrmse, get_nse
from utils.io import load_pickle, load_results, write_pickle, load_feather
from utils.plot_tools import get_pretty_var_name


PSWEEP_RESULTS_DIR = PDIRS["PROJECT_RESULTS"] / "parameter_sweep"
GIS_DIR = GENERAL_DATA_DIR / "GIS"


def load_grand_names():
    df = load_feather(
        (PDIRS["PROJECT_DATA"] / "grand_names.feather").as_posix(),
    )
    return df.set_index("GRAND_ID").drop("index", axis=1)


def load_parameter_sweep_results(model_dir=None):
    if model_dir:
        model_dir = pathlib.Path(model_dir)
        return {model_dir.name: load_results(model_dir.as_posix())}
    else:
        directories = glob.glob(f"{PSWEEP_RESULTS_DIR.as_posix()}/*")
        return {pathlib.Path(d).name: load_results(d) for d in directories}


def get_parameter_sweep_data(results, dataset="simmed"):
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


def calculate_metrics(data, data_set, recalc=False):
    metrics_file = (
        PDIRS["PROJECT_AGG_RESULTS"]
        / "parameter_sweep"
        / f"{data_set}_metrics.pickle"
    )
    if not recalc and metrics_file.exists():
        return load_pickle(metrics_file.as_posix())
    models = list(data.drop("actual", axis=1).columns)
    models = sorted(
        models,
        key=lambda x: (int(x.split("_")[0][2:]), float(x.split("_")[1][3:])),
    )

    nse = pd.DataFrame()
    nrmse = pd.DataFrame()
    for model in models:
        m_nse = get_nse(data, "actual", model, grouper="res_id")
        m_nrmse = get_nrmse(data, "actual", model, grouper="res_id")
        m_nse.name = model
        m_nrmse.name = model

        if nse.empty:
            nse = m_nse.to_frame()
        else:
            nse[model] = m_nse

        if nrmse.empty:
            nrmse = m_nrmse.to_frame()
        else:
            nrmse[model] = m_nrmse

    write_pickle({"nse": nse, "nrmse": nrmse}, metrics_file.as_posix())
    return {"nse": nse, "nrmse": nrmse}


def metric_wide_to_long(metric_df, metric):
    df = metric_df.melt(var_name="model", value_name=metric)
    df[["TD", "MSS"]] = df["model"].str.split("_", expand=True)
    df["TD"] = df["TD"].str.slice(2)
    df["MSS"] = df["MSS"].str.slice(3)
    return df.drop("model", axis=1)


def plot_metric_box_plot(metric_df, metric):
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


def plot_single_model_metrics(df):
    grand_names = load_grand_names()
    test_nse = get_nse(df, "actual", "test", grouper="res_id")
    test_nrmse = get_nrmse(df, "actual", "test", grouper="res_id")
    simmed_nse = get_nse(df, "actual", "simmed", grouper="res_id")
    simmed_nrmse = get_nrmse(df, "actual", "simmed", grouper="res_id")

    nse = pd.DataFrame.from_dict({"test": test_nse, "simmed": simmed_nse})
    nrmse = pd.DataFrame.from_dict({"test": test_nrmse, "simmed": simmed_nrmse})

    nse = nse.reset_index().melt(id_vars="res_id")
    nrmse = nrmse.reset_index().melt(id_vars="res_id")
    nse["metric"] = "nse"
    nrmse["metric"] = "nrmse"

    metrics = pd.concat([nse, nrmse])
    metrics["res_name"] = metrics["res_id"].apply(
        lambda x: grand_names.loc[int(x), "RES_NAME"]
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
    axes[0].legend(loc="best")
    axes[1].tick_params(axis="x", labelrotation=90)

    axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=60, ha="right")
    fg.set_titles("")
    axes[0].set_ylabel("NRMSE")
    axes[1].set_ylabel("NSE")
    axes[1].set_xlabel("")

    fg.figure.align_ylabels()

    plt.show()


def compare_training_testing_data(results):
    meta = load_feather(FILES["MODEL_READY_META"])
    meta = meta.set_index("res_id")

    mr_data = load_feather(FILES["MODEL_READY_DATA"]).set_index(
        ["res_id", "date"]
    )

    test_df = get_parameter_sweep_data(results, dataset="test")

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


def setup_map(ax=None, coords=None, other_bound=None):
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

    states_path = GIS_DIR / "cb_2017_us_state_500k"

    mbound = m.drawmapboundary(fill_color="white")
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
    WBD_DIR = GIS_DIR / "WBD"
    file = "WBD_{:02}_HU2_Shape/Shape/WBDHU2"
    bounds_files = [
        (WBD_DIR / file.format(i)).as_posix() for i in range(1, 19)
    ]
    return bounds_files


def plot_training_testing_map(results):
    fig, ax = plt.subplots(1, 1)
    wbds = get_contiguous_wbds()

    other_bounds = [
        (b, "k") for b in wbds 
    ]

    west, south, east, north = (
        -127.441406,
        24.207069,
        -66.093750,
        53.382373,
    )
    m = setup_map(ax=ax, coords=[west, south, east, north], other_bound=other_bounds)
    grand = gpd.read_file(
        GENERAL_DATA_DIR / "GRanD Databasev1.3" / "GRanD_reservoirs_v1_3.shp"
    )

    test_df = get_parameter_sweep_data(results, dataset="test")
    train_df = get_parameter_sweep_data(results, dataset="train")

    test_res = test_df.index.get_level_values("res_id").unique().astype(int)
    train_res = train_df.index.get_level_values("res_id").unique().astype(int)

    test_coords = [
        (row.LONG_DD, row.LAT_DD)
        for i, row in grand[grand["GRAND_ID"].isin(test_res)].iterrows()
    ]
    train_coords = [
        (row.LONG_DD, row.LAT_DD)
        for i, row in grand[grand["GRAND_ID"].isin(train_res)].iterrows()
    ]

    train_x, train_y = list(zip(*train_coords))
    test_x, test_y = list(zip(*test_coords))

    m.scatter(
        train_x, train_y, latlon=True, label="Training", marker="v", zorder=4
    )
    m.scatter(
        test_x, test_y, latlon=True, label="Testing", marker="v", zorder=4
    )

    ax.legend(loc="lower left")

    plt.show()


if __name__ == "__main__":
    # plt.style.use("ggplot")
    sns.set_theme(context="talk", palette="Set2")
    # results = load_parameter_sweep_results()
    # simmed_data = get_parameter_sweep_data(results, dataset="simmed")
    # metrics = calculate_metrics(simmed_data, data_set="simmed", recalc=False)
    # nse, rmse = metrics["nse"], metrics["nrmse"]
    # plot_metric_box_plot(nse, "NSE")

    model = "TD3_MSS0.04"
    results = load_parameter_sweep_results(PSWEEP_RESULTS_DIR / model)
    df = get_parameter_sweep_data(results, dataset="simmed")
    df = df.rename(columns={model: "simmed"})
    df["test"] = get_parameter_sweep_data(results, dataset="test")[model]
    # plot_single_model_metrics(df)

    # compare_training_testing_data(results)
    plot_training_testing_map(results)
