from typing import Union

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score

calc_type = Union[np.array, pd.Series]


def nrmse(actual: calc_type, model: calc_type, handle_zero=True) -> float:
    """Calculate normalized root mean square error

    MSE / E(actual)

    Args:
        actual (np.array | pd.Series): Array of actual values
        model (np.array | pd.Series): Array of model values
        handle_zero (bool, optional): If True, will not divide by zero.

    Returns:
        float: NRMSE
    """
    if handle_zero:
        divisor = max([actual.mean(), 0.01])
    else:
        divisor = actual.mean()
    return mean_squared_error(actual, model, squared=False) / divisor


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
    else:
        scores = r2_score(df[actual], df[model])
    scores.name = "NSE"
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
    else:
        scores = nnse(df[actual], df[model])
    scores.name = "NNSE"
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
    else:
        scores = mean_squared_error(df[actual], df[model], squared=False)
    scores.name = "RMSE"
    return scores


def get_nrmse(df: pd.DataFrame, actual: str, model: str, grouper=None) -> pd.Series:
    """Get NRMSE for df.

    If grouper is not None, will get NRMSE for each unique item in grouper

    Args:
        df (pd.DataFrame): Dataframe containing model records
        actual (str): Name of column containing actual records
        model (str): Name of column containing model records
        grouper (_type_, optional): mapping, function, label, or list of labels.
            Defaults to None.

    Returns:
        pd.Series: NRMSE Values
    """
    if grouper is not None:
        scores = df.groupby(grouper).apply(lambda x: nrmse(x[actual], x[model]))
    else:
        scores = nrmse(df[actual], df[model])
    scores.name = "RMSE"
    return scores
