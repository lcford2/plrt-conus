import pathlib

import pandas as pd

RESOPS_PATH = pathlib.Path("~/data/ResOpsUS")
PROJECT_ROOT = pathlib.Path("~/projects/plrt-conus")
AGG_FILE = PROJECT_ROOT / "data" / "resopsus_agg" / "sri_metric.pickle"


def combine_single_variable_tables():
    table_dir = RESOPS_PATH / "time_series_single_variable_table"
    inflow = pd.read_csv(
        table_dir / "DAILY_AV_INFLOW_CUMECS.csv",
        index_col=0,
    )
    release = pd.read_csv(
        table_dir / "DAILY_AV_OUTFLOW_CUMECS.csv",
        index_col=0,
    )
    storage = pd.read_csv(
        table_dir / "DAILY_AV_STORAGE_MCM.csv",
        index_col=0,
    )

    inflow = (
        inflow.melt(ignore_index=False, var_name="res_id", value_name="inflow_cms")
        .reset_index()
        .set_index(["res_id", "date"])
    )
    release = (
        release.melt(ignore_index=False, var_name="res_id", value_name="release_cms")
        .reset_index()
        .set_index(["res_id", "date"])
    )
    storage = (
        storage.melt(ignore_index=False, var_name="res_id", value_name="storage_mcm")
        .reset_index()
        .set_index(["res_id", "date"])
    )

    return pd.concat([inflow, release, storage], axis=1)


def convert_units(records):
    # flow variables: from m3/s to m3/day
    # storage: from mcm to m
    records["inflow_cms"] *= 3600 * 24
    records["release_cms"] *= 3600 * 24
    records["storage_mcm"] *= 1e6

    records = records.rename(
        columns={
            "inflow_cms": "inflow",
            "release_cms": "release",
            "storage_mcm": "storage",
        }
    )
    return records


def save_records(records):
    records.to_pickle(AGG_FILE)
    print(f"Records stored in {AGG_FILE}")


if __name__ == "__main__":
    records = combine_single_variable_tables()
    records = convert_units(records)
    save_records(records)
