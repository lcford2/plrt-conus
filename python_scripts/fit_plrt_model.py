import argparse
import copy
import os
import pickle
from datetime import datetime
from multiprocessing import cpu_count
from time import perf_counter as timer

CPUS = cpu_count()
nprocs_per_job = 2
njobs = int(CPUS / 2)
os.environ["OMP_NUM_THREADS"] = str(nprocs_per_job)

import numpy as np
import pandas as pd
from IPython import embed as II
from make_resopsus_meta_data import make_meta_data
from make_resopsus_model_ready import get_max_res_date_spans, trim_data_to_span
from plrt import PieceWiseLinearRegressionTree, load_model
from sklearn.metrics import mean_squared_error, r2_score
from utils.config import config
from utils.io import load_feather, load_pickle, write_pickle
from utils.timing_function import time_function
from utils.utils import my_groupby

START_TIME = timer()


def log(string, start_time=None):
    ts = timer()
    if start_time:
        ts -= start_time
    else:
        ts -= START_TIME
    print(f"[{ts:.1f} sec elapsed] {string}")


@time_function
def load_resopsus_data(min_years=3):
    # if min_years == 5:
    #     data_file = config.get_file("merged_data")
    #     meta_file = config.get_file("merged_meta")
    #     meta = load_feather(meta_file, index_keys=("res_id",))
    #     data = load_feather(data_file, index_keys=("res_id", "date"))
    #     data["inflow"] = data["net_inflow"]
    # else:
    data_file = config.get_dir("model_ready_data") / f"resops_{min_years}yr.feather"
    data = load_feather(data_file, index_keys=("res_id", "date"))
    data = merge_mb_and_resops(data)
    data["inflow"] = data["net_inflow"]
    spans = get_max_res_date_spans(data)
    data = trim_data_to_span(data, spans)
    meta = make_meta_data(data)
    return data, meta


@time_function
def prep_data(df, monthly=False):
    if monthly:
        grouper = [
            df.index.get_level_values(0),
            df.index.get_level_values(1).month,
        ]
    else:
        grouper = df.index.get_level_values(0)
    std_data = my_groupby(df, grouper).apply(
        lambda x: (x - x.mean()) / x.std().replace({0.0: 1.0}),
    )
    means = my_groupby(df, grouper).mean()
    std = my_groupby(df, grouper).std().replace({0.0: 1.0})
    columns = [
        "release_pre",
        "storage",
        "storage_pre",
        "inflow",
        "release_roll7",
        "inflow_roll7",
        "storage_roll7",
        "storage_x_inflow",
        "inflow2",
        "release_pre2",
        # "sto_diff",
    ]
    X = std_data.loc[:, columns]
    y = std_data["release"]
    return X, y, means, std


def split_train_test_res(index, test_res):
    train = index[~index.get_level_values(1).isin(test_res)]
    test = index[index.get_level_values(1).isin(test_res)]
    return train, test


def get_params_and_groups(X, tree):
    # use the tree to get what leaves correpond with each entry
    # in the X matrix
    params, leaves, paths = tree.apply(X)
    # make those leaves into a pandas series for the ml model
    groups = pd.Series(leaves, index=X.index)
    return params, groups


def split_train_test_index_by_res(df, prop=0.8):
    resers = df.index.get_level_values(0).unique()
    train_index = []
    test_index = []
    idx = pd.IndexSlice
    for res in resers:
        rdf = df.loc[idx[res, :], :].index
        size = len(rdf)
        train_size = int(prop * size)
        train_index.extend(rdf.values[:train_size])
        test_index.extend(rdf.values[train_size:])
    return train_index, test_index


def split_train_test_by_basin(resers, train_prop):
    res_huc2 = load_feather(config.get_dir("spatial_data") / "updated_res_huc2.feather")
    res_huc2 = res_huc2[res_huc2["res_id"].isin(resers)]
    hucs = res_huc2["huc2_id"].unique()
    huc_resers = {i: res_huc2[res_huc2["huc2_id"] == i]["res_id"].values for i in hucs}
    basin_train, basin_test = {}, {}
    import ipdb

    for huc_id, huc_res in huc_resers.items():
        ipdb.set_trace()
        n_train_res = int(np.floor(len(huc_res) * train_prop))
        train_res = np.random.choice(huc_res, n_train_res, replace=False)
        test_res = [i for i in huc_res if i not in train_res]
        basin_train[huc_id] = train_res
        basin_test[huc_id] = test_res

    return basin_train, basin_test


def split_train_test_by_basin_meta_var(resers, meta, meta_var, threshold):
    res_huc2 = load_feather(config.get_dir("spatial_data") / "updated_res_huc2.feather")
    res_huc2 = res_huc2[res_huc2["res_id"].isin(resers)]
    hucs = res_huc2["huc2_id"].unique()
    huc_resers = {i: res_huc2[res_huc2["huc2_id"] == i]["res_id"].values for i in hucs}
    basin_train, basin_test = {}, {}
    if threshold < 0:
        threshold = abs(threshold)
        flip_test_train = True
    else:
        flip_test_train = False

    for huc_id, huc_res in huc_resers.items():
        huc_meta = meta.loc[huc_res]
        cut_off = huc_meta[meta_var].quantile(threshold)
        train_res = huc_meta[huc_meta[meta_var] < cut_off].index
        test_res = huc_meta[huc_meta[meta_var] >= cut_off].index
        if flip_test_train:
            basin_train[huc_id] = test_res
            basin_test[huc_id] = train_res
        else:
            basin_train[huc_id] = train_res
            basin_test[huc_id] = test_res
    return basin_train, basin_test


def split_train_test_by_meta_var(resers, meta, meta_var, threshold):
    res_huc2 = load_feather(config.get_dir("spatial_data") / "updated_res_huc2.feather")
    res_huc2 = res_huc2[res_huc2["res_id"].isin(resers)]
    hucs = res_huc2["huc2_id"].unique()
    {i: res_huc2[res_huc2["huc2_id"] == i]["res_id"].values for i in hucs}
    if threshold < 0:
        threshold = abs(threshold)
        flip_test_train = True
    else:
        flip_test_train = False

    cut_off = meta[meta_var].quantile(threshold)
    train_res = meta[meta[meta_var] < cut_off].index
    test_res = meta[meta[meta_var] >= cut_off].index
    if flip_test_train:
        temp_train_res = train_res.copy()
        train_res = test_res
        test_res = temp_train_res
    return train_res, test_res


def merge_mb_and_resops(df):
    mb_df = load_feather(config.get_dir("data") / "model_ready" / "mb_data.feather")
    mb_df = mb_df.rename(
        columns={
            "GRAND_ID": "res_id",
            "datetime": "date",
            "inflow": "net_inflow",
        },
    )
    mb_df = mb_df.set_index(["res_id", "date"])

    taf_to_m3 = 1233.48 * 1000
    base_columns = [
        "release",
        "release_pre",
        "storage",
        "storage_pre",
        "net_inflow",
        "release_roll7",
        "inflow_roll7",
        "storage_roll7",
    ]
    mb_df.loc[:, base_columns] *= taf_to_m3
    mb_df["storage_x_inflow"] = mb_df["storage_pre"] * mb_df["net_inflow"]
    mb_df["release_pre2"] = mb_df["release_pre"] ** 2
    mb_df["inflow2"] = mb_df["net_inflow"] ** 2
    expected_columns = [
        *base_columns,
        "storage_x_inflow",
        "release_pre2",
        "inflow2",
    ]
    mb_df = mb_df.loc[:, expected_columns]
    existing_keys = []
    for index in mb_df.index:
        if index in df.index:
            existing_keys.append(index)
    df.loc[existing_keys, expected_columns] = mb_df.loc[existing_keys, expected_columns]
    left_out = mb_df.drop(existing_keys)
    df = pd.concat([df, left_out])
    df["good_row"] = df["good_row"].astype(bool)
    df["all_vars"] = df["all_vars"].astype(bool)
    return df


def unstandardize(series, mean, std):
    monthly = hasattr(mean.index, "nlevels") and mean.index.nlevels > 1
    if monthly:
        idx = pd.IndexSlice
        umean = mean.unstack()
        ustd = std.unstack()
        series_act = series.copy()
        for month in range(1, 13):
            mseries = (
                series.loc[idx[:, series.index.get_level_values(1).month == month]]
                .unstack()
                .T
            )
            mmean = umean[month]
            mstd = ustd[month]
            mseries_act = (mseries * mstd + mmean).T.stack()
            series_act.loc[mseries_act.index] = mseries_act
    else:
        series_act = (series.unstack().T * std + mean).T.stack()
    return series_act


def pipeline(args):
    model_set, foldername = make_output_paths(args)
    folderpath = config.get_dir("results") / model_set / foldername
    start_time = timer()
    # month_intercepts = args.month_ints
    max_depth = args.max_depth

    df, meta = load_resopsus_data(args.min_years)

    lower_bounds = my_groupby(df, df.index.get_level_values(0)).min()
    upper_bounds = my_groupby(df, df.index.get_level_values(0)).max()

    df = df.sort_index()
    reservoirs = meta.index

    X, y, means, std = prep_data(df, monthly=args.monthly)
    df["sto_diff"] = X["storage_pre"] - X["storage_roll7"]
    X["sto_diff"] = X["storage_pre"] - X["storage_roll7"]

    # Setup monthly intercepts if using
    # if month_intercepts:
    #     month_arrays = {i: [] for i in calendar.month_abbr[1:]}
    #     for date in X.index.get_level_values(1):
    #         for key in month_arrays.keys():
    #             if calendar.month_abbr[date.month] == key:
    #                 month_arrays[key].append(1)
    #             else:
    #                 month_arrays[key].append(0)
    #     for key, array in month_arrays.items():
    #         X[key] = array

    # set exogenous variables
    X_vars = [
        "storage_pre",
        "release_pre",
        "inflow",
        # "storage_roll7",
        "sto_diff",
        "release_roll7",
        "inflow_roll7",
        "storage_x_inflow",
        "inflow2",
        "release_pre2",
    ]

    if args.previous_model:
        X_vars.remove("inflow2")
        X_vars.remove("release_pre2")

    X_vars_tree = copy.copy(X_vars)

    # no constants for tree to split on
    # if month_intercepts:
    #     X_vars.extend(calendar.month_abbr[1:])
    # else:
    #     X_vars.insert(0, "const")

    # train_index = X[X.index.get_level_values(1) < datetime(2010, 1, 1)].index
    # test_index = X[X.index.get_level_values(1) >= datetime(2010, 1, 1)].index
    # train_index, test_index = split_train_test_index_by_res(X, prop=0.8)

    log("Splitting testing and training reservoirs", start_time)

    np.random.seed(44)
    splitting_method = args.splitting_method
    all_conus = args.conus_split
    conus_mod = "conus_" if all_conus else ""
    cache_file = (
        f"./model_setup_files/test_train_{conus_mod}{'_'.join(splitting_method)}.pickle"
    )
    if os.path.exists(cache_file):
        basin_train, basin_test = load_pickle(cache_file)
    else:
        if splitting_method[0] == "basin":
            basin_train, basin_test = split_train_test_by_basin(
                reservoirs,
                float(splitting_method[-1]),
            )
        else:
            if all_conus:
                basin_train, basin_test = split_train_test_by_meta_var(
                    reservoirs,
                    meta,
                    splitting_method[1],
                    float(splitting_method[-1]),
                )
            else:
                basin_train, basin_test = split_train_test_by_basin_meta_var(
                    reservoirs,
                    meta,
                    splitting_method[1],
                    float(splitting_method[-1]),
                )
        write_pickle((basin_train, basin_test), cache_file)

    if not all_conus:
        train_res, test_res = [], []
        for resers in basin_train.values():
            train_res.extend(resers)
        for resers in basin_test.values():
            test_res.extend(resers)
    else:
        train_res = basin_train
        test_res = basin_test

    print(f"Number of training reservoirs: {len(train_res)}")
    print(f"Number of testing reservoirs: {len(test_res)}")
    testing = True
    if len(test_res) == 0:
        testing = False

    # n_train_res = int(np.floor(len(reservoirs) * train_fraction))
    # train_res = np.random.choice(reservoirs, n_train_res, replace=False)
    # test_res = [i for i in reservoirs if i not in train_res]

    log("Getting training and testing data set", start_time)
    X_train = X.loc[X.index.get_level_values(0).isin(train_res), X_vars].sort_index()
    X_train["const"] = 1
    train_index = X_train.index

    if testing:
        X_test = X.loc[X.index.get_level_values(0).isin(test_res), X_vars].sort_index()
        X_test["const"] = 1
        test_index = X_test.index

    X_vars.insert(0, "const")

    y_train = y.loc[train_index]
    if testing:
        y_test = y.loc[test_index]

    df["const"] = 1
    if testing:
        X_test_act = df.loc[test_index, X_vars]
        y_test_act = df.loc[test_index, "release"]

    add_tree_vars = ["rts", "max_sto"]

    for tree_var in add_tree_vars:
        train_values = []
        for res, _ in X_train.index:
            train_values.append(meta.loc[res, tree_var])
        X_train[tree_var] = train_values

        if testing:
            test_values = []
            for res, _ in X_test.index:
                test_values.append(meta.loc[res, tree_var])
            X_test[tree_var] = test_values
            X_test_act[tree_var] = test_values

    X_vars_tree = [*X_vars, *add_tree_vars]

    if args.sim_all or not testing:
        X_simul = df.loc[:, X_vars].sort_index()
        for tree_var in add_tree_vars:
            values = []
            for res, _ in X_simul.index:
                values.append(meta.loc[res, tree_var])
            X_simul[tree_var] = values
    else:
        X_simul = X_test_act

    X_vars_tree.remove("const")

    min_samples_split = args.mss
    make_dot = False

    if args.data_init:
        II()
        import sys

        sys.exit()

    if max_depth > 0:
        make_dot = True
        if args.load_model:
            model = load_model((folderpath / "model.pickle").as_posix())
        elif args.previous_model:
            model = load_model(
                os.path.expanduser(
                    "~/projects/predict-release/results/tclr_model_testing/all/TD4_MSS0.10_RT_MS_exhaustive/model.pickle",
                ),
            )
        else:
            model = PieceWiseLinearRegressionTree(
                X_train,
                y_train,
                max_depth=max_depth,
                # feature_names=feat_names,
                response_name="release",
                tree_vars=X_vars_tree,
                reg_vars=X_vars,
                njobs=njobs,
                method=args.method,
                n_disc_steps=1000,
                min_samples_split=min_samples_split,
            )

            log("Fitting model", start_time)

            time_function(model.fit)()

        params, groups = get_params_and_groups(X_train, model)
        # get the unique final leaves
        groups_uniq = groups.unique()
        # sorts them in ascending order
        groups_uniq.sort()
        # maps them to their sorted index + 1
        group_map = {j: i + 1 for i, j in enumerate(groups_uniq)}
        groups = groups.apply(group_map.get)
        groups_list = list(groups)
        coefs = pd.DataFrame(
            {i: params[groups_list.index(i)] for i in group_map.values()},
            index=group_map.values(),
            columns=X_vars,
        )

        if args.previous_model:
            fitted = time_function(model.predict)(X_train)
        else:
            fitted = time_function(model.predict)()
        if testing:
            preds = time_function(model.predict)(X_test)

        log("Simulating model", start_time)

        simuled = simulate_plrt_model(
            model,
            "model",
            # X_test_act,
            X_simul,
            means,
            std,
            X_vars_tree,
            X_vars,
            lower_bounds,
            upper_bounds,
            args.assim,
            args.parallel_verbosity,
            parallel=args.parallel,
        )
        simuled = simuled[["release", "storage"]].dropna()
    else:
        X_train = X_train[X_vars]
        X_test = X_test[X_vars]
        X_test_act = X_test_act[X_vars]
        beta = np.linalg.inv(X_train.T @ X_train) @ (X_train.T @ y_train)
        coefs = pd.Series(beta, index=X_vars)
        fitted = X_train @ beta
        if testing:
            preds = X_test @ beta
        simuled = simulate_plrt_model(
            beta,
            "beta",
            # X_test_act,
            X_simul,
            means,
            std,
            X_vars,
            X_vars,
            lower_bounds,
            upper_bounds,
            args.assim,
            args.parallel_verbosity,
            parallel=args.parallel,
        )
        simuled = simuled[["release", "storage"]].dropna()

    log("Preparing Output", start_time)

    fitted = pd.Series(fitted, index=X_train.index)
    if testing:
        preds = pd.Series(preds, index=X_test.index)
    simmed = simuled["release"]

    idx = pd.IndexSlice
    if args.monthly:
        train_locator = idx[train_res, :]
    else:
        train_locator = train_res

    if testing:
        if args.monthly:
            test_locator = idx[test_res, :]
        else:
            test_locator = test_res

    y_train_act = unstandardize(
        y_train,
        means.loc[train_locator, "release"],
        std.loc[train_locator, "release"],
    )
    if testing:
        y_test_act = unstandardize(
            y_test,
            means.loc[test_locator, "release"],
            std.loc[test_locator, "release"],
        )
    fitted_act = unstandardize(
        fitted,
        means.loc[train_locator, "release"],
        std.loc[train_locator, "release"],
    )
    if testing:
        preds_act = unstandardize(
            preds,
            means.loc[test_locator, "release"],
            std.loc[test_locator, "release"],
        )

    y_test_sim = df.loc[simmed.index, "release"]

    f_act_score = r2_score(y_train_act, fitted_act)
    np.sqrt(mean_squared_error(y_train_act, fitted_act))
    if testing:
        r2_score(y_test_act, preds_act)
    s_act_score = r2_score(y_test_sim, simmed)

    train_data = pd.DataFrame({"actual": y_train_act, "model": fitted_act})
    if testing:
        test_data = pd.DataFrame({"actual": y_test_act, "model": preds_act})
    else:
        test_data = pd.DataFrame(columns=["actual", "model"])
    simmed_data = pd.DataFrame({"actual": y_test_sim, "model": simmed})

    # setup output parameters
    # check if the directory exists and handle it
    if folderpath.is_dir():
        folderpath = (
            config.get_dir("results")
            / model_set
            / "_".join([foldername, datetime.today().strfime("%Y%m%d_%H%M")])
        )
        print(f"Saving at {folderpath} instead.")
        folderpath.mkdir()
    else:
        folderpath.mkdir(parents=True)

    # export tree to graphviz file so it can be converted nicely
    # rotate_tree = True if max_depth > 3 else e
    if make_dot:
        model.to_graphviz(folderpath / "tree.dot")
    if max_depth > 0:
        model.save_model((folderpath / "model.pickle").as_posix())

    # setup output container for modeling information
    X_train["storage_pre"] = X["storage_pre"]
    output = {
        "f_act_score": f_act_score,
        # "p_act_score": p_act_score,
        "s_act_score": s_act_score,
        # "f_res_scores": train_res_scores,
        # "p_res_scores": test_res_scores,
        # "s_res_scores": simmed_res_scores,
        "train_data": train_data,
        "test_data": test_data,
        "simmed_data": simmed_data,
    }
    if max_depth > 0:
        output["groups"] = groups

    # write the output dict to a pickle file
    with open((folderpath / "results.pickle").as_posix(), "wb") as f:
        pickle.dump(output, f, protocol=4)

    # write the random effects to a csv file for easy access
    coefs.to_csv((folderpath / "random_effects.csv").as_posix())
    log("Done", start_time)


def make_output_paths(args):
    model_set = f"merged_data_set_minyr{args.min_years}"
    if args.monthly:
        model_set = f"monthly_{model_set}"
    if args.previous_model:
        foldername = "previous_model"
    else:
        assim_mod = f"_{args.assim}" if args.assim else ""
        mss_mod = f"_MSS{args.mss:0.2f}"
        conus_mod = "_conus" if args.conus_split else ""
        sm_mod = f"_SM_{conus_mod}{'_'.join(args.splitting_method)}"
        from_store_mod = "_FROM_STORED" if args.load_model else ""
        foldername = f"TD{args.max_depth}{assim_mod}{mss_mod}{sm_mod}{from_store_mod}"

    return model_set, foldername


def simulate_plrt_model(
    model,
    model_or_beta,
    X_act,
    means,
    std,
    X_vars,
    reg_vars,
    lower_bounds,
    upper_bounds,
    assim,
    verbosity,
    parallel=True,
):
    # I need to keep track of actual storage and release outputs
    # as well as rolling weekly mean storage and release outputs
    track_df = pd.DataFrame(
        columns=["release", "storage"] + list(X_act.columns),
        index=X_act.index,
    )
    track_df[["const", "inflow", "inflow_roll7", "inflow2"]] = X_act[
        ["const", "inflow", "inflow_roll7", "inflow2"]
    ]

    # setup initial tracking information as well as find the start dates
    resers = track_df.index.get_level_values(0).unique()
    start_dates = {}
    idx = pd.IndexSlice
    for res in resers:
        rdf = X_act.loc[idx[res, :], :]
        # rolling 7 day means so we need 7 days of actual values
        first_seven = rdf.index.get_level_values(1).values[:7]
        track_df.loc[idx[res, first_seven], ["release_pre", "storage_pre"]] = X_act.loc[
            idx[res, first_seven],
            ["release_pre", "storage_pre"],
        ]
        start_dates[res] = first_seven[-1]

    # find the initial rolling release and storage values
    init_rolling = (
        my_groupby(track_df, track_df.index.get_level_values(0))[
            ["storage_pre", "release_pre"]
        ]
        .rolling(7)
        .mean()
        .dropna()
    )
    # get a duplicate reservoir index after the above command
    init_rolling.index = init_rolling.index.droplevel(0)

    # add init rolling values to track df
    track_df.loc[
        init_rolling.index,
        ["storage_roll7", "release_roll7"],
    ] = init_rolling.values

    X_act["storage_roll7"] = track_df["storage_roll7"]

    # since all reservoirs have different temporal spans
    # we have to iterate through each reservoir independently
    from joblib import Parallel, delayed

    if parallel:
        outputdfs = Parallel(n_jobs=-1, verbose=verbosity)(
            delayed(simul_reservoir)(
                res,
                model,
                model_or_beta,
                X_act,
                means,
                std,
                X_vars,
                reg_vars,
                lower_bounds,
                upper_bounds,
                assim,
            )
            for res in resers
        )
    else:
        outputdfs = []
        for res in resers:
            outputdfs.append(
                simul_reservoir(
                    res,
                    model,
                    model_or_beta,
                    X_act,
                    means,
                    std,
                    X_vars,
                    reg_vars,
                    lower_bounds,
                    upper_bounds,
                    assim,
                ),
            )
    return pd.concat(outputdfs)


def simul_reservoir(
    res,
    model,
    model_or_beta,
    track_df,
    means,
    std,
    X_loc_vars,
    reg_vars,
    lower_bounds,
    upper_bounds,
    assim=None,
):
    idx = pd.IndexSlice
    rdf = track_df.loc[idx[res, :], :].copy(deep=True)
    # cut off initial 6 values as we do not have a roling release for those
    dates = list(rdf.index.get_level_values(1).values)[6:]
    # need to identify the end date so we can know to stop adding
    # values to track_df
    end_date = dates[-1]

    if assim == "weekly":
        assim_shift = 7
    elif assim == "monthly":
        assim_shift = 30
    elif assim == "seasonally":
        assim_shift = 90
    elif assim == "semi-annually":
        assim_shift = 180
    elif assim == "yearly":
        assim_shift = 365
    elif assim == "daily":
        assim_shift = 1

    start_date = dates[0]
    if "const" in reg_vars:
        cindex = reg_vars.index("const")
        index_after_const = cindex + 1
        reg_vars = reg_vars[:cindex] + reg_vars[index_after_const:]

    roll_storage = pd.Series(index=rdf.index, dtype=np.float64)
    roll_storage.loc[idx[res, dates[0]]] = rdf.loc[idx[res, dates[0]], "storage_roll7"]

    # monthly = False
    # if hasattr(means.index, "levels"):
    #     if len(means.index.nlevels) > 1:
    #         monthly = True

    monthly = hasattr(means.index, "nlevels") and means.index.nlevels > 1

    for date in dates:
        loc = idx[res, date]
        month = date.astype("datetime64[M]").astype(int) % 12 + 1
        # get values for today
        X_r = rdf.loc[loc, X_loc_vars]
        # add the interaction term
        X_r["storage_x_inflow"] = X_r["storage_pre"] * X_r["inflow"]
        X_r["release_pre2"] = X_r["release_pre"] * X_r["release_pre"]
        # grab actual rt and max_sto values if they exist
        rts, max_sto = None, None
        res_vars = False
        try:
            rts, max_sto = X_r[["rts", "max_sto"]]
            res_vars = True
        except KeyError:
            res_vars = False
        # add the difference term
        # X_r["sto_diff"] = X_r["storage_pre"] - roll_storage.loc[loc]

        # standardize the values
        reg_vars_nsd = reg_vars.copy()
        reg_vars_nsd.remove("sto_diff")

        if monthly:
            X_r = (X_r - means.loc[pd.IndexSlice[res, month], reg_vars_nsd]) / std.loc[
                pd.IndexSlice[res, month],
                reg_vars_nsd,
            ]

            X_r["sto_diff"] = X_r["storage_pre"] - (
                (
                    rdf.loc[loc, "storage_roll7"]
                    - means.loc[pd.IndexSlice[res, month], "storage_roll7"]
                )
                / std.loc[pd.IndexSlice[res, month], "storage_roll7"]
            )
        else:
            X_r = (X_r - means.loc[res, reg_vars_nsd]) / std.loc[res, reg_vars_nsd]
            X_r["sto_diff"] = X_r["storage_pre"] - (
                (rdf.loc[loc, "storage_roll7"] - means.loc[res, "storage_roll7"])
                / std.loc[res, "storage_roll7"]
            )

        # and the constant
        X_r["const"] = 1
        if res_vars:
            X_r["max_sto"] = max_sto
            X_r["rts"] = rts

        # reshape to a 2 d row vector
        if model_or_beta == "model":
            X_r_val = X_r[model.feats].values
            X_r_val = X_r_val.reshape(1, X_r_val.size)
            release = model.predict(X_r_val)[0]
            # release = model.predict(X_r[model.feats].values.reshape(
            #     1, X_r.size
            # ))[0]
        else:
            release = X_r[X_loc_vars] @ model

        if np.isnan(release):
            import ipdb

            ipdb.set_trace()

        # get release back to actual space
        if monthly:
            release_act = (
                release * std.loc[pd.IndexSlice[res, month], "release"]
                + means.loc[pd.IndexSlice[res, month], "release"]
            )
        else:
            release_act = release * std.loc[res, "release"] + means.loc[res, "release"]
        # calculate storage from mass balance
        storage = rdf.loc[loc, "storage_pre"] + rdf.loc[loc, "inflow"] - release_act
        # keep storage and release within bounds
        if storage > upper_bounds.loc[res, "storage"]:
            storage = upper_bounds.loc[res, "storage"]
        elif storage < lower_bounds.loc[res, "storage"]:
            storage = lower_bounds.loc[res, "storage"]

        if release_act > upper_bounds.loc[res, "release"]:
            release_act = upper_bounds.loc[res, "release"]
        elif release_act < lower_bounds.loc[res, "release"]:
            release_act = lower_bounds.loc[res, "release"]

        # store calculated values
        rdf.loc[loc, "storage"] = storage
        rdf.loc[loc, "release"] = release_act

        # if we are not at the last day, store values needed for tomorrow
        if date != end_date:
            tomorrow = date + np.timedelta64(1, "D")
            prev_seven = pd.date_range(tomorrow - np.timedelta64(6, "D"), tomorrow)

            if assim:
                offset = ((date - start_date) / np.timedelta64(1, "D")) % assim_shift
                if offset == 0:
                    rdf.loc[idx[res, tomorrow], "storage_pre"] = track_df.loc[
                        idx[res, tomorrow],
                        "storage_pre",
                    ]
                    rdf.loc[idx[res, tomorrow], "release_pre"] = track_df.loc[
                        idx[res, tomorrow],
                        "release_pre",
                    ]
                else:
                    rdf.loc[idx[res, tomorrow], "storage_pre"] = storage
                    rdf.loc[idx[res, tomorrow], "release_pre"] = release_act
            else:
                rdf.loc[idx[res, tomorrow], "storage_pre"] = storage
                rdf.loc[idx[res, tomorrow], "release_pre"] = release_act
            # here we already updated the _pre values we can use the
            # same logic to get rolling values
            rdf.loc[idx[res, tomorrow], "storage_roll7"] = rdf.loc[
                idx[res, prev_seven],
                "storage_pre",
            ].mean()
            roll_storage.loc[idx[res, tomorrow]] = rdf.loc[
                idx[res, prev_seven],
                "storage_pre",
            ].mean()
            rdf.loc[idx[res, tomorrow], "release_roll7"] = rdf.loc[
                idx[res, prev_seven],
                "release_pre",
            ].mean()
            rdf.loc[idx[res, tomorrow], :] = rdf.loc[idx[res, tomorrow], :]
        rdf.loc[loc, :] = rdf.loc[loc, :]
    return rdf


def parse_args(arg_list=None):
    parser = argparse.ArgumentParser(
        description="Fit and test a PLRT model to specified data",
    )
    parser.add_argument(
        "-d",
        "--max_depth",
        dest="max_depth",
        default=3,
        type=int,
        help="How deep should the tree be allowed to go? (default=3)",
    )
    parser.add_argument(
        "--min-years",
        default=3,
        choices=range(1, 6),
        type=int,
        help="How many years of data is required for a reservoir to be valid?",
    )
    parser.add_argument(
        "--assim",
        default=None,
        choices=(
            "daily",
            "weekly",
            "monthly",
            "seasonally",
            "semi-annually",
            "yearly",
        ),
        help="Frequency at which to assimilate observed" " storage and release values",
    )
    parser.add_argument(
        "-M",
        "--method",
        # default="Nelder-Mead",
        default="exhaustive",
        help="Optimization algorithm to use for fitting the TCLR model.",
    )
    parser.add_argument(
        "--mss",
        type=float,
        default=0.05,
        help="Fraction of samples required to be in a child" " node to perform a split",
    )
    parser.add_argument(
        "--data-init",
        dest="data_init",
        action="store_true",
        default=False,
        help="Just prepare the training and testing data then"
        " launch an IPython session.",
    )
    parser.add_argument(
        "-p",
        "--parallel",
        action="store_true",
        default=False,
        help="Perform simulation in parallel to speed it up.",
    )
    parser.add_argument(
        "-m",
        "--monthly",
        action="store_true",
        default=False,
        help="Standardize variables using monthly mean and std values.",
    )
    parser.add_argument(
        "-S",
        "--splitting_method",
        type=str,
        nargs="+",
        default=["basin", "0.8"],
        help="How should the training and the testing split be split? "
        + "If len = 2, should be 'basin' 'proportion'. "
        + "If len = 3, should be 'meta' 'meta var' 'threshold'",
    )
    parser.add_argument(
        "--conus-split",
        action="store_true",
        help="Indicate that splitting should occur for conus instead of per basin",
    )
    parser.add_argument(
        "--sim-all",
        action="store_true",
        default=False,
        help="Simulate all records, not just testing records",
    )
    parser.add_argument(
        "--load-model",
        action="store_true",
        default=False,
        help="Load stored model, rather than fitting new one",
    )
    parser.add_argument(
        "-pv",
        "--parallel-verbosity",
        default=10,
        type=int,
        help="Verbosity level for `joblib` Parallel()",
    )
    parser.add_argument(
        "--previous-model",
        dest="previous_model",
        action="store_true",
        default=False,
        help="Run model from PLRT MB Paper on CONUS Data",
    )
    if arg_list:
        return parser.parse_args(arg_list)
    else:
        return parser.parse_args()


if __name__ == "__main__":
    print("CPUS           = ", CPUS)
    print("NPROCS_PER_JOB = ", nprocs_per_job)
    print("NJOBS          = ", njobs)
    args = parse_args()
    pipeline(args)
