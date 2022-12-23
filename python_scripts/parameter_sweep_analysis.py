import glob
import pathlib

import pandas as pd
from IPython import embed as II
from utils.config import PDIRS
from utils.io import load_results
from utils.metrics import get_nrmse, get_nse

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


def calculate_metrics(data):
    models = list(data.drop("actual", axis=1).columns)
    models = sorted(
        models, key=lambda x: (int(x.split("_")[0][2:]), float(x.split("_")[1][3:]))
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
    return nse, nrmse


def plot_metric_box_plot(metric_df):
    II()


if __name__ == "__main__":
    results = load_parameter_sweep_results()
    simmed_data = get_parameter_sweep_data(results, dataset="simmed")
    nse, nrmse = calculate_metrics(simmed_data)
    plot_metric_box_plot(nse)
