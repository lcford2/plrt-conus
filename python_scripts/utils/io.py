import pathlib
import pickle
from typing import Any

import pandas as pd
from utils.config import config


def load_feather(file: str | pathlib.Path, index_keys=()) -> pd.DataFrame:
    """Load a pandas dataframe from a feather file.

    Args:
        file (str | path.Pathlib): feather file to load
        index_keys (tuple, optional): Columns to set index to. Defaults to ().

    Returns:
        pd.DataFrame: DataFrame from feather file
    """

    df = pd.read_feather(file)
    if index_keys:
        df = df.set_index(list(index_keys))
    return df


def write_feather(df: pd.DataFrame, file: str | pathlib.Path) -> None:
    """Write a pandas dataframe to a feather file

    Args:
        df (pd.DataFrame): dataframe to write
        file (str | pathlib.Path): file path to store dataframe
    """
    df.reset_index().to_feather(file)


def load_pickle(file: str | pathlib.Path) -> Any:
    """Read a pickle file

    Args:
        file (str | pathlib.Path): File to read.

    Returns:
        Any: Python object from pickle file
    """
    with open(file, "rb") as f:
        data = pickle.load(f)
    return data


def write_pickle(data: Any, file: str | pathlib.Path) -> None:
    """Write data to file using pickle serialization

    Args:
        data (Any): Data to write
        file (str | pathlib.Path): Filename to write to.
    """
    with open(file, "wb") as f:
        pickle.dump(data, f, protocol=4)


def load_results(path: str | pathlib.Path) -> dict:
    """Load `results.pickle` file from the model path

    Args:
        path (str | pathlib.Path): Path to the model directory that contains
        `results.pickle`

    Returns:
        dict: Results dictionary from model
    """
    return load_pickle((pathlib.Path(path) / "results.pickle").as_posix())


def load_huc2_name_map() -> dict:
    huc2_names = {}
    with open(config.get_dir("spatial_data") / "huc2_names.csv", "r") as f:
        for line in f.readlines():
            split = line.strip().split(",")
            huc2_names[int(split[0])] = f"{split[0]}: {split[1]}"
    return huc2_names


def load_huc2_basins() -> dict:
    huc2 = load_feather(
        config.get_dir("spatial_data") / "updated_res_huc2.feather",
        index_keys=("res_id",),
    )
    huc2_names = load_huc2_name_map()
    huc2["name"] = huc2["huc2_id"].replace(huc2_names)
    return huc2
