import pandas as pd
from utils.config import config
from utils.io import load_feather, write_feather

MODEL_READY_FILE = config.get_file("merged_data")
MODEL_READY_META_FILE = config.get_file("merged_meta")


def make_meta_data(df):
    means = df.groupby("res_id").mean()
    meta = pd.DataFrame(
        index=means.index, columns=["rts", "max_sto", "rel_inf_corr"]
    )
    meta["rts"] = means["storage"] / means["release"]
    meta["max_sto"] = df.groupby("res_id")["storage"].max()
    meta["rel_inf_corr"] = (
        df.groupby("res_id")[["net_inflow", "release"]]
        .corr()["release"]
        .unstack()["net_inflow"]
    )
    return meta


if __name__ == "__main__":
    df = load_feather(MODEL_READY_FILE, index_keys=("res_id", "date"))
    meta = make_meta_data(df)
    write_feather(meta, MODEL_READY_META_FILE)
    print(f"Meta data stored in {MODEL_READY_META_FILE}")
