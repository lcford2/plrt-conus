import pandas as pd
from utils.config import FILES

MODEL_READY_FILE = FILES["MODEL_READY_DATA"]
MODEL_READY_META_FILE = FILES["MODEL_READY_META"]


def load_model_ready_data():
    return pd.read_pickle(MODEL_READY_FILE)


def make_meta_data(df):
    means = df.groupby("res_id").mean()
    meta = pd.DataFrame(index=means.index, columns=["rts", "max_sto", "rel_inf_corr"])
    meta["rts"] = means["storage"] / means["release"]
    meta["max_sto"] = df.groupby("res_id")["storage"].max()
    meta["rel_inf_corr"] = (
        df.groupby("res_id")[["net_inflow", "release"]]
        .corr()["release"]
        .unstack()["net_inflow"]
    )
    return meta


if __name__ == "__main__":
    df = load_model_ready_data()
    meta = make_meta_data(df)
    meta.to_pickle(MODEL_READY_META_FILE)
