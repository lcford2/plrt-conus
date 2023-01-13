from utils.config import PDIRS
from utils.io import load_feather, load_pickle


def load_trimmed_resers():
    return load_pickle(PDIRS["PROJECT_DATA"] / "trimmed_resers.pickle")


def load_res_huc2_map():
    return load_feather(PDIRS["PROJECT_SPATIAL_DATA"] / "res_huc2.feather")


def determine_huc2_trimming_changes():
    trimmed_res = load_trimmed_resers()
    huc2_map = load_res_huc2_map()
    huc2_map = huc2_map.set_index("res_id")

    for nyear, resers in trimmed_res.items():
        resers = resers.astype(int).values
        huc2_map.loc[resers, nyear] = 1

    huc2_res_count = huc2_map.groupby("huc2_id").sum()

    print(huc2_res_count.to_markdown(floatfmt=".0f"))


if __name__ == "__main__":
    determine_huc2_trimming_changes()
