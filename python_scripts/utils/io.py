import pandas as pd


def load_feather(file, index_keys=()):
    df = pd.read_feather(file)
    if index_keys:
        df = df.set_index(list(index_keys))
    return df


def write_feather(df, file):
    df.reset_index().to_feather(file)
