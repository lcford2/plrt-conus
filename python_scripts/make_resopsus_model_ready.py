import datetime

import pandas as pd
from utils.config import FILES
from utils.io import load_feather, write_feather

AGG_FILE = FILES["RESOPS_AGG"]
MODEL_READY_FILE = FILES["MODEL_READY_DATA"]


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
    df["release_pre2"] = df["release_pre"] ** 2
    df["inflow2"] = df["net_inflow"] ** 2

    df = df.dropna(how="any", axis=0)

    spans = get_max_res_date_spans(df)
    trimmed_spans = filter_short_spans(spans, 5)
    trimmed_df = trim_data_to_span(df, trimmed_spans)
    print(
        f"Trimming process removed {1 - trimmed_df.shape[0] / df.shape[0]:.1%} of records."
    )
    return trimmed_df


def find_equal_not_equal_indices(df, occ, index):
    equal = True
    ddf = df.loc[index]
    ddf = ddf.dropna(how="all", axis=1)
    for col in ddf.columns:
        if not (ddf[col] == ddf[col][0]).all():
            equal = False
    if equal:
        return (index, 0)
    else:
        return (index, 1)


def remove_duplicate_dates(df):
    # get indices that occur more than once
    occ = pd.Series(index=df.index, data=1)
    occ = occ.groupby(["res_id", "date"]).sum()
    occ = occ[occ > 1]

    from joblib import Parallel, delayed

    index_map = Parallel(n_jobs=-1, verbose=1)(
        delayed(find_equal_not_equal_indices)(
            df,
            occ,
            index,
        )
        for index in occ.index
    )

    all_equal = [x[0] for x in filter(lambda x: x[1] == 0, index_map)]
    not_all_equal = [x[0] for x in filter(lambda x: x[1] == 1, index_map)]

    #
    all_equal_rows = df.loc[all_equal]
    all_equal_new_rows = all_equal_rows.groupby(["res_id", "date"]).mean()
    df = df.drop(all_equal)
    df = pd.concat([df, all_equal_new_rows])

    df = df[~df.index.get_level_values(0).isin([i[0] for i in not_all_equal])]
    return df


if __name__ == "__main__":
    df = load_feather(AGG_FILE, index_keys=("res_id", "date"))
    df = remove_duplicate_dates(df)
    df = make_model_ready_data(df)
    write_feather(df, MODEL_READY_FILE)
    print(f"Model ready data stored in {MODEL_READY_FILE}")
