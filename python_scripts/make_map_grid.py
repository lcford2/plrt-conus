import pathlib

import numpy as np
from PIL import Image

image_path = pathlib.Path(
    "/home/lucas/Dropbox/plrt-conus-figures/good_figures/"
    "op_group_analysis/op_group_entropy_maps"
)

images = [
    image_path / i
    for i in [
        "small_mid_rt.png",
        "medium_mid_rt.png",
        "medium_high_rt.png",
        "large.png",
        "very_large.png",
    ]
]


# left, upper, right, lower
# (0, 0, 1920, 972),
bounds = [
    (310, 0, 1525, 772),
    (400, 0, 1525, 772),
    (395, 0, 1620, 772),
    (310, 60, 1520, 832),
    (395, 60, 1625, 832),
]

cropped = []
for i in range(len(images)):
    im = Image.open(images[i])
    cropped.append(im.crop(bounds[i]))

empty = np.zeros_like(cropped[1]) + 255
row1 = np.hstack(cropped[:3])
row2 = np.hstack([*cropped[3:], empty])
out = np.vstack([row1, row2])

im = Image.fromarray(out)
im.save(image_path / "stitched_maps.png", compress_level=0, dpi=(450, 450))

legend = Image.open(image_path / "basin_mean_entropy_legend.png")

new_width = int(empty.shape[1] * 0.85)
new_height = int(legend.size[1] / legend.size[0] * new_width)
legend = legend.resize((new_width, new_height))
legend_size = np.array(legend).shape

middle_empty = (empty.shape[0] // 2, empty.shape[1] // 2)
half_legend_size = (legend_size[0] // 2, legend_size[1] // 2)
legend_start = [m - h for m, h in zip(middle_empty, half_legend_size)]
empty[
    legend_start[0] : legend_start[0] + legend_size[0],
    legend_start[1] : legend_start[1] + legend_size[1],
    :3,
] = np.array(legend)

row2_w_legend = np.hstack([*cropped[3:], empty])
out = np.vstack([row1, row2_w_legend])

im = Image.fromarray(out)
im.save(image_path / "stitched_maps_w_legend.png", compress_level=0, dpi=(450, 450))
