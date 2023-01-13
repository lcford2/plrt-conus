import geopandas as gpd
import pandas as pd
from joblib import Parallel, delayed
from utils.config import GENERAL_DATA_DIR, PDIRS

GIS_DIR = GENERAL_DATA_DIR / "GIS"
SPAT_DIR = PDIRS["PROJECT_SPATIAL_DATA"]


def load_wbds():
    WBD_DIR = GIS_DIR / "WBD"
    file = "WBD_{:02}_HU2_Shape/Shape/WBDHU2.shp"
    bounds_files = [(WBD_DIR / file.format(i)).as_posix() for i in range(1, 19)]
    return [gpd.read_file(f) for f in bounds_files]


def find_res_contained_in_wbd(wbd_id, wbd, grand_df):
    poly = wbd["geometry"][0]
    contains = grand_df[poly.contains(grand_df["geometry"])]
    return [(g_id, wbd_id) for g_id in list(contains["GRAND_ID"].values)]


def match_grand_id_to_wbd():
    wbds = load_wbds()
    grand = gpd.read_file(SPAT_DIR / "grand_info")

    wbd_maps = Parallel(n_jobs=-1, verbose=11)(
        delayed(find_res_contained_in_wbd)(i + 1, wbd, grand)
        for i, wbd in enumerate(wbds)
    )
    wbd_map = []
    for wbd in wbd_maps:
        wbd_map.extend(wbd)

    wbd_df = pd.DataFrame.from_records(wbd_map, columns=["res_id", "huc2_id"])
    wbd_df.to_feather((SPAT_DIR / "res_huc2.feather").as_posix())


if __name__ == "__main__":
    match_grand_id_to_wbd()
