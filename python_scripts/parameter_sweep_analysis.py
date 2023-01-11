import glob
import pathlib

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from IPython import embed as II
from utils import (
    PDIRS,
    get_nrmse,
    get_nse,
    load_pickle,
    load_results,
    write_pickle,
)

PSWEEP_RESULTS_DIR = PDIRS["PROJECT_RESULTS"] / "parameter_sweep"


def load_parameter_sweep_results():
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


if __name__ == "__main__":
    # plt.style.use("ggplot")
    sns.set_theme(context="notebook", palette="Set2")
    results = load_parameter_sweep_results()
    simmed_data = get_parameter_sweep_data(results, dataset="simmed")
    metrics = calculate_metrics(simmed_data, data_set="simmed", recalc=False)
    nse, rmse = metrics["nse"], metrics["nrmse"]
    II()
    plot_metric_box_plot(nse, "NSE")
