import glob
import pathlib

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from IPython import embed as II
from utils.config import PDIRS
from utils.metrics import get_nrmse, get_nse
from utils.io import load_pickle, load_results, write_pickle, load_feather
from utils.plot_tools import get_pretty_var_name


PSWEEP_RESULTS_DIR = PDIRS["PROJECT_RESULTS"] / "parameter_sweep"


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

    nse = pd.DataFrame.from_dict({
        "test": test_nse,
        "simmed": simmed_nse
    })
    nrmse = pd.DataFrame.from_dict({
        "test": test_nrmse,
        "simmed": simmed_nrmse
    })

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
        sharey=False
    )
    axes = fg.axes.flatten()
    axes[0].legend(loc="best")
    axes[1].tick_params(
        axis="x",
        labelrotation=90
    )

    axes[1].set_xticklabels(
        axes[1].get_xticklabels(),
        rotation=60,
        ha="right"
    )
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
        facet_kws={
            "sharex": False, 
            "sharey": False, 
            "legend_out": False
        },
    )
    fg.set_titles("{col_name}")
    plt.show()




if __name__ == "__main__":
    # plt.style.use("ggplot")
    sns.set_theme(context="notebook", palette="Set2")
    # results = load_parameter_sweep_results()
    # simmed_data = get_parameter_sweep_data(results, dataset="simmed")
    # metrics = calculate_metrics(simmed_data, data_set="simmed", recalc=False)
    # nse, rmse = metrics["nse"], metrics["nrmse"]
    # plot_metric_box_plot(nse, "NSE")

    model = "TD3_MSS0.04"
    results = load_parameter_sweep_results(
        PSWEEP_RESULTS_DIR / model
    )
    df = get_parameter_sweep_data(results, dataset="simmed")
    df = df.rename(columns={model: "simmed"})
    df["test"] = get_parameter_sweep_data(results, dataset="test")[model]
    plot_single_model_metrics(df)
