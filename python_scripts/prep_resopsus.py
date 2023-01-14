import pandas as pd
from utils.config import config
from utils.io import write_feather

RESOPS_PATH = config.get_dir("resops")
AGG_FILE = config.get_file("resops_agg")


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
    inflow.index = pd.to_datetime(inflow.index, format="%Y-%m-%d")
    release.index = pd.to_datetime(release.index, format="%Y-%m-%d")
    storage.index = pd.to_datetime(storage.index, format="%Y-%m-%d")

    inflow = (
        inflow.melt(
            ignore_index=False, var_name="res_id", value_name="inflow_cms"
        )
        .reset_index()
        .set_index(["res_id", "date"])
    )
    release = (
        release.melt(
            ignore_index=False, var_name="res_id", value_name="release_cms"
        )
        .reset_index()
        .set_index(["res_id", "date"])
    )
    storage = (
        storage.melt(
            ignore_index=False, var_name="res_id", value_name="storage_mcm"
        )
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


if __name__ == "__main__":
    records = combine_single_variable_tables()
    records = convert_units(records)
    write_feather(records, AGG_FILE)
    print(f"Records stored in {AGG_FILE}")
