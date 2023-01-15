import geopandas as gpd
import matplotlib.pyplot as plt
import seaborn as sns
from palettable.cartocolors.qualitative import Vivid_5
from parameter_sweep_analysis import get_contiguous_wbds, setup_map
from utils.config import config
from utils.io import load_feather, load_pickle


def load_trimmed_resers():
    return load_pickle(config.get_dir("data_to_sync") / "trimmed_resers.pickle")


def load_noncon_trimmed_resers():
    return load_pickle(
        config.get_dir("data_to_sync") / "noncon_trim_res.pickle"
    )


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


def plot_trimming_changes_map(noncon=False):
    fig, ax = plt.subplots(1, 1)

    wbds = get_contiguous_wbds()

    other_bounds = [(b, "k") for b in wbds]

    west, south, east, north = (
        -127.441406,
        24.207069,
        -66.093750,
        53.382373,
    )
    m = setup_map(
        ax=ax, coords=[west, south, east, north], other_bound=other_bounds
    )

    grand = gpd.read_file(
        (config.get_dir("spatial_data") / "grand_info").as_posix()
    )
    grand = grand.set_index("GRAND_ID")

    if noncon:
        trimmed_res = load_noncon_trimmed_resers()
    else:
        trimmed_res = load_trimmed_resers()

    for key, resers in trimmed_res.items():
        resers = resers.astype(int)
        coords = grand.loc[resers]
        x, y = list(
            zip(
                *[
                    (row["LONG_DD"], row["LAT_DD"])
                    for _, row in coords.iterrows()
                ]
            )
        )
        n_res = len(x)
        m.scatter(
            x,
            y,
            latlon=True,
            marker="v",
            label=f"{key} year ({n_res})",
            zorder=3,
        )
    ax.legend(loc="lower left")

    plt.show()


if __name__ == "__main__":
    sns.set_theme(context="talk", palette=Vivid_5.mpl_colors)
    # determine_huc2_trimming_changes(True)
    plot_trimming_changes_map(True)
