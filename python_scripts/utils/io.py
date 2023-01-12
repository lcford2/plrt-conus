import pathlib
import pickle

import pandas as pd


def load_feather(file, index_keys=()):
    df = pd.read_feather(file)
    if index_keys:
        df = df.set_index(list(index_keys))
    return df


def write_feather(df, file):
    df.reset_index().to_feather(file)


def load_pickle(file):
    with open(file, "rb") as f:
        data = pickle.load(f)
    return data


def write_pickle(data, file):
    with open(file, "wb") as f:
        pickle.dump(data, f, protocol=4)


def load_results(path):
    return load_pickle((pathlib.Path(path) / "results.pickle").as_posix())
