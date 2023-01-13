from utils.config import config
from utils.io import load_feather, load_pickle

DATA_DIR = config.get_dir("data")


def load_trimmed_resers():
    return load_pickle(DATA_DIR / "trimmed_resers.pickle")


def load_noncon_trimmed_resers():
    return load_pickle(DATA_DIR / "noncon_trim_res.pickle")


def load_res_huc2_map():
    return load_feather(config.d_spatial_data / "res_huc2.feather")


def determine_huc2_trimming_changes(noncon=False):
    if noncon:
        trimmed_res = load_noncon_trimmed_resers()
    else:
        trimmed_res = load_trimmed_resers()

    huc2_map = load_res_huc2_map()
    huc2_map = huc2_map.set_index("res_id")

    for nyear, resers in trimmed_res.items():
        resers = resers.astype(int).values
        huc2_map.loc[resers, nyear] = 1

    huc2_res_count = huc2_map.groupby("huc2_id").sum()

    print(huc2_res_count.to_markdown(floatfmt=".0f"))


if __name__ == "__main__":
    determine_huc2_trimming_changes(True)
