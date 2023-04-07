from typing import Union

import numpy as np
import pandas as pd
from scipy.stats import entropy
from sklearn.metrics import mean_squared_error, r2_score

calc_type = Union[np.array, pd.Series]


def nrmse(
    actual: calc_type, model: calc_type, divisor="mean", handle_zero=True
) -> float:
    """Calculate normalized root mean square error

    MSE / E(actual)

    Args:
        actual (np.array | pd.Series): Array of actual values
        model (np.array | pd.Series): Array of model values
        divisor (str, optional): How to calculate divisor for NRMSE. Options are:
            ["mean", "median", "range", "std"]
        handle_zero (bool, optional): If True, will not divide by zero.

    Returns:
        float: NRMSE
    """
    if divisor == "mean":
        div = actual.mean()
    elif divisor == "median":
        div = actual.median()
    elif divisor == "range":
        div = actual.max() - actual.min()
    elif divisor == "std":
        div = actual.std()
    else:
        raise ValueError("Invalid divisor")

    if handle_zero:
        div = max([div, 0.01])

    return mean_squared_error(actual, model, squared=False) / div


def nnse(actual: calc_type, model: calc_type) -> float:
    """Calculate normalized NSE

    1 / (2 - NSE)

    Args:
        actual (np.array | pd.Series): Array of actual values
        model (np.array | pd.Series): Array of model values

    Returns:
        float: NNSE
    """
    return 1 / (2 - r2_score(actual, model))


def get_nse(df: pd.DataFrame, actual: str, model: str, grouper=None) -> pd.Series:
    """Get NSE for df.

    If grouper is not None, will get NSE for each unique item in grouper

    Args:
        df (pd.DataFrame): Dataframe containing model records
        actual (str): Name of column containing actual records
        model (str): Name of column containing model records
        grouper (_type_, optional): mapping, function, label, or list of labels.
            Defaults to None.

    Returns:
        pd.Series: NSE Values
    """
    if grouper is not None:
        scores = df.groupby(grouper).apply(lambda x: r2_score(x[actual], x[model]))
        scores.name = "NSE"
    else:
        scores = r2_score(df[actual], df[model])
    return scores


def get_nnse(df: pd.DataFrame, actual: str, model: str, grouper=None) -> pd.Series:
    """Get NNSE for df.

    If grouper is not None, will get NNSE for each unique item in grouper

    Args:
        df (pd.DataFrame): Dataframe containing model records
        actual (str): Name of column containing actual records
        model (str): Name of column containing model records
        grouper (_type_, optional): mapping, function, label, or list of labels.
            Defaults to None.

    Returns:
        pd.Series: NNSE Values
    """
    if grouper is not None:
        scores = df.groupby(grouper).apply(lambda x: nnse(x[actual], x[model]))
        scores.name = "NNSE"
    else:
        scores = nnse(df[actual], df[model])
    return scores


def get_rmse(df: pd.DataFrame, actual: str, model: str, grouper=None) -> pd.Series:
    """Get RMSE for df.

    If grouper is not None, will get RMSE for each unique item in grouper

    Args:
        df (pd.DataFrame): Dataframe containing model records
        actual (str): Name of column containing actual records
        model (str): Name of column containing model records
        grouper (_type_, optional): mapping, function, label, or list of labels.
            Defaults to None.

    Returns:
        pd.Series: RMSE Values
    """
    if grouper is not None:
        scores = df.groupby(grouper).apply(
            lambda x: mean_squared_error(x[actual], x[model], squared=False)
        )
        scores.name = "RMSE"
    else:
        scores = mean_squared_error(df[actual], df[model], squared=False)
    return scores


def get_nrmse(
    df: pd.DataFrame, actual: str, model: str, grouper=None, divisor="mean"
) -> pd.Series:
    """Get NRMSE for df.

    If grouper is not None, will get NRMSE for each unique item in grouper

    Args:
        df (pd.DataFrame): Dataframe containing model records
        actual (str): Name of column containing actual records
        model (str): Name of column containing model records
        divisor (str): How to calculate divisor for NRMSE. Options are:
            ["mean", "median", "range", "std"]
        grouper (_type_, optional): mapping, function, label, or list of labels.
            Defaults to None.

    Returns:
        pd.Series: NRMSE Values
    """
    if grouper is not None:
        scores = df.groupby(grouper).apply(
            lambda x: nrmse(x[actual], x[model], divisor=divisor)
        )
        scores.name = "RMSE"
    else:
        scores = nrmse(df[actual], df[model], divisor=divisor)
    return scores


def get_variance(values: pd.Series, grouper=None) -> pd.Series:
    """Get Variance of values.

    If grouper is not None, will get Variance for each unique item in grouper

    Args:
        values (pd.Series): Series containing values to get variance for
        grouper (any, optional): mapping, function, label, or list of labels.
            Defaults to None.

    Returns:
        pd.Series: NRMSE Values
    """
    if grouper is not None:
        scores = values.groupby(grouper).var()
        scores.name = "variance"
    else:
        scores = values.var()
    return scores


def get_entropy(values: pd.Series, grouper=None, scale=False) -> pd.Series:
    """Get entropy of values.

    If grouper is not None, will get entropy for each unique item in grouper

    Args:
        values (pd.Series): Series containing values to get entropy for
        grouper (any, optional): mapping, function, label, or list of labels.
            Defaults to None.

    Returns:
        pd.Series: entropy values
    """
    if grouper is not None:
        probs = values.groupby("res_id").apply(lambda x: x.value_counts() / x.count())
        scores = probs.groupby(grouper).apply(entropy, base=2)
        if scale:
            ngroups = probs.index.get_level_values(1).unique().size
            scores = scores / np.log2(ngroups)
        scores.name = "entropy"
    else:
        probs = values.value_counts() / values.count()
        scores = entropy(values, base=2)
        if scale:
            scores = scores / np.log2(probs.shape[0])
    return scores


def jaccard(a: list, b: list) -> float:
    """Calculate Jaccard index

    Args:
        a (list): Array of values
        b (list): Array of values

    Returns:
        float: Jaccard index
    """
    intersection = len(list(set(a) & set(b)))
    union = (len(a) + len(b)) - intersection
    return float(intersection / union)
