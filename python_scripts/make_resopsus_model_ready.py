import datetime

import numpy as np
import pandas as pd
from utils.config import config
from utils.io import load_feather, write_feather, write_pickle

AGG_FILE = config.get_file("resops_agg")
MODEL_READY_FILE = config.get_file("model_ready_data")


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
        zip(
            df.index.get_level_values(0),
            pd.to_datetime(df.index.get_level_values(1)),
        ),
        names=df.index.names,
    )

    outlier_resers = {
        "1020": ["release"],
        "1042": ["release"],
        "1170": ["release"],
        "1777": ["storage"],
        "572": ["release"],
        "616": ["storage"],
        "629": ["storage"],
        "7214": ["release"],
        "870": ["storage"],
        "929": ["release"],
    }
    for res, fix_vars in outlier_resers.items():
        for var in fix_vars:
            df.loc[pd.IndexSlice[res, :], var] = fix_outliers(
                df.loc[pd.IndexSlice[res, :], var],
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
    jdf = df.copy()
    jdf["julian_day"] = jdf.index.get_level_values(1).dayofyear
    julian_counts = jdf.groupby("res_id")["julian_day"].value_counts()
    julian_counts = julian_counts.loc[pd.IndexSlice[:, list(range(1, 366))]]

    trimmed_resers = {}
    for i in range(1, 6):
        julian_mask = julian_counts > i
        res_mask = julian_mask.groupby("res_id").all()
        trimmed_resers[i] = res_mask[res_mask].index
        res_mask = res_mask[res_mask]

    write_pickle(
        trimmed_resers,
        config.get_dir("data_to_sync") / "noncon_trim_res.pickle",
    )

    spans = get_max_res_date_spans(df)

    def get_trimmed_df(spans, min_yrs, df):
        trimmed_spans = filter_short_spans(spans, min_yrs)
        trimmed_df = trim_data_to_span(df, trimmed_spans)
        return trimmed_df

    trimmed_dfs = {i: get_trimmed_df(spans, i, df) for i in range(1, 6)}

    trimmed_resers = {
        i: tdf.index.get_level_values("res_id").unique for i, tdf in trimmed_dfs.items()
    }

    write_pickle(
        trimmed_resers,
        config.get_dir("data_to_sync") / "trimmed_resers.pickle",
    )

    for i, tdf in trimmed_dfs.items():
        write_feather(
            tdf,
            config.get_dir("data") / "model_ready" / f"resops_{i}yr.feather",
        )

    trimmed_df = get_trimmed_df(spans, 5, df)

    trimmed_percent = 1 - trimmed_df.shape[0] / df.shape[0]
    print(f"Trimming process removed {trimmed_percent:.1%} of records.")
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


def fix_outliers(series):
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    lb = q1 - 1.5 * iqr
    ub = q3 + 1.5 * iqr
    series[(series < lb) | (series > ub)] = np.nan
    return series.interpolate()


if __name__ == "__main__":
    df = load_feather(AGG_FILE, index_keys=("res_id", "date"))
    df = remove_duplicate_dates(df)
    df = make_model_ready_data(df)
    write_feather(df, MODEL_READY_FILE)
    print(f"Model ready data stored in {MODEL_READY_FILE}")
