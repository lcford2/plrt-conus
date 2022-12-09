import datetime

import pandas as pd
from prep_resopsus import AGG_FILE, PROJECT_ROOT

MODEL_READY_FILE = PROJECT_ROOT / "data" / "model_ready" / "resopsus.pickle"


def load_agg_data():
    return pd.read_pickle(AGG_FILE)


def get_max_date_span(in_df):
    df = pd.DataFrame()
    dates = pd.to_datetime(in_df.index.get_level_values(1))
    df["date"] = dates
    df["mask"] = 1
    df.loc[df["date"] - datetime.timedelta(days=1) == df["date"].shift(), "mask"] = 0
    df["mask"] = df["mask"].cumsum()
    spans = df.loc[df["mask"] == df["mask"].value_counts().idxmax(), "date"]
    return (spans.min(), spans.max())


def get_max_res_date_spans(df):
    reservoirs = df.index.get_level_values(0).unique()
    spans = {r: {} for r in reservoirs}
    idx = pd.IndexSlice
    for res in reservoirs:
        span = get_max_date_span(df.loc[idx[res, :], :])
        spans[res]["min"] = span[0]
        spans[res]["max"] = span[1]
    spans = pd.DataFrame.from_dict(spans).T
    spans["delta"] = spans["max"] - spans["min"]
    return spans.sort_values(by="delta")


def filter_short_spans(spans, min_yrs=5):
    cut_off = min_yrs * 365.25
    trimmed_spans = spans[spans["delta"].dt.days >= cut_off]
    return trimmed_spans


def trim_data_to_span(df, spans):
    out_dfs = []
    idx = pd.IndexSlice
    for res, row in spans.iterrows():
        min_date = row["min"]
        max_date = row["max"]
        res_df = df.loc[idx[res, :], :]
        res_df = res_df.loc[
            (res_df.index.get_level_values(1) >= min_date)
            & (res_df.index.get_level_values(1) <= max_date)
        ]
        out_dfs.append(res_df)
    return pd.concat(out_dfs, axis=0, ignore_index=False)


def make_model_ready_data(df):
    # get rows that have atleast storage and release
    notna = pd.notna(df)
    df["good_row"] = notna["storage"] & notna["release"]
    df["all_vars"] = df["good_row"] & notna["inflow"]
    df = df.loc[df["good_row"], :].copy()
    df.index = pd.MultiIndex.from_tuples(
        zip(df.index.get_level_values(0), pd.to_datetime(df.index.get_level_values(1))),
        names=df.index.names,
    )

    # get pre variables
    df[["storage_pre", "release_pre"]] = df.groupby("res_id")[
        ["storage", "release"]
    ].shift(1)

    # calculate net inflow
    df["net_inflow"] = df["storage"] - df["storage_pre"] + df["release"]

    # calculate rolling variables
    rolling_means = (
        df.groupby("res_id")[["storage_pre", "release_pre", "net_inflow"]]
        .rolling(7)
        .mean()
    )
    rolling_means.index = rolling_means.index.droplevel(0)
    df[["storage_roll7", "release_roll7", "inflow_roll7"]] = rolling_means

    # calculate S x I
    df["storage_x_inflow"] = df["storage_pre"] * df["net_inflow"]

    # calculate squared terms
    df["release2"] = df["release_pre"] ** 2
    df["inflow2"] = df["net_inflow"] ** 2

    spans = get_max_res_date_spans(df)
    trimmed_spans = filter_short_spans(spans, 5)
    trimmed_df = trim_data_to_span(df, trimmed_spans)
    print(
        f"Trimming process removed {1 - trimmed_df.shape[0] / df.shape[0]:.1%} of records."
    )
    return trimmed_df


if __name__ == "__main__":
    df = load_agg_data()
    mdf = make_model_ready_data(df)
    mdf.to_pickle(MODEL_READY_FILE)
