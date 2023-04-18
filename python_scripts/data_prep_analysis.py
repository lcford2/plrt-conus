import geopandas as gpd
import matplotlib.pyplot as plt
import seaborn as sns
from fit_plrt_model import load_resopsus_data
from palettable.cartocolors.qualitative import Vivid_5
from parameter_sweep_analysis import setup_wbd_map
from utils.config import config
from utils.io import load_feather, load_pickle


def load_trimmed_resers():
    return load_pickle(config.get_dir("data_to_sync") / "trimmed_resers.pickle")


def load_noncon_trimmed_resers():
    return load_pickle(config.get_dir("data_to_sync") / "noncon_trim_res.pickle")


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
    fig, ax, m = setup_wbd_map()

    grand = gpd.read_file((config.get_dir("spatial_data") / "grand_info").as_posix())
    grand = grand.set_index("GRAND_ID")

    if noncon:
        trimmed_res = load_noncon_trimmed_resers()
    else:
        trimmed_res = load_trimmed_resers()

    for key, resers in trimmed_res.items():
        resers = resers.astype(int)
        coords = grand.loc[resers]
        x, y = list(
            zip(*[(row["LONG_DD"], row["LAT_DD"]) for _, row in coords.iterrows()]),
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


def plot_data_diff_map(results, year1, year2):
    fig, ax, m = setup_wbd_map()
    grand = gpd.read_file(config.get_dir("spatial_data") / "my_grand_info")
    big_grand = gpd.read_file(config.get_file("grand_file"))
    big_grand["GRAND_ID"] = big_grand["GRAND_ID"].astype(str)

    yr1_data, yr1_meta = load_resopsus_data(year1)
    yr2_data, yr2_meta = load_resopsus_data(year2)

    grand = grand.set_index("GRAND_ID")

    yr1_coords = grand.loc[yr1_meta.index, ["LONG_DD", "LAT_DD"]].values.tolist()
    yr2_coords = grand.loc[yr2_meta.index, ["LONG_DD", "LAT_DD"]].values.tolist()

    yr1_x, yr1_y = list(zip(*yr1_coords))
    yr2_x, yr2_y = list(zip(*yr2_coords))

    yr1_z = 4
    yr2_z = 5
    if len(yr1_x) < len(yr2_x):
        yr1_z = 4
        yr2_z = 5

    m.scatter(
        yr1_x,
        yr1_y,
        latlon=True,
        label=f"Min Years={year1}",
        marker="v",
        zorder=yr1_z,
    )
    m.scatter(
        yr2_x,
        yr2_y,
        latlon=True,
        label=f"Min Years={year2}",
        marker="v",
        zorder=yr2_z,
    )

    ax.legend(loc="lower left")

    plt.show()


if __name__ == "__main__":
    sns.set_theme(context="talk", palette=Vivid_5.mpl_colors)
    # determine_huc2_trimming_changes(True)
    plot_trimming_changes_map(True)
    # plot_data_diff_map(results, int(min_years), int(min_years) + 1)
    # plot_data_diff_map(results, 3, 5)
