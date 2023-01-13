from utils.config import PDIRS
from utils.io import load_feather, load_pickle


def load_trimmed_resers():
    return load_pickle(PDIRS["PROJECT_DATA"] / "trimmed_resers.pickle")


def load_res_huc2_map():
    return load_feather(PDIRS["PROJECT_SPATIAL_DATA"] / "res_huc2.feather")


def determine_huc2_trimming_changes():
    trimmed_res = load_trimmed_resers()
    huc2_map = load_res_huc2_map()
    trimmed_res
    huc2_map
