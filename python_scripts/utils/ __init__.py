# flake8: noqa: F401
from .config import FILES, PDIRS, PROJECT_ROOT, RESOPSUS_UNTS
from .io_utils import (
    load_feather,
    load_pickle,
    load_results,
    write_feather,
    write_pickle,
)
from .metrics import get_nrmse, get_nse, get_rmse, nrmse
from .timing_function import time_function


def my_groupby(df, keys):
    return df.groupby(keys, group_keys=False)
