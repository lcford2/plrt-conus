import pathlib

import numpy as np
from PIL import Image

image_path = pathlib.Path(
    "/home/lucas/Dropbox/plrt-conus-figures/good_figures/"
    "op_group_analysis/op_group_entropy_maps/scaled",
)

images = [
    image_path / i
    for i in [
        "scaled_small_mid_rt.png",
        "scaled_medium_mid_rt.png",
        "scaled_medium_high_rt.png",
        "scaled_large.png",
        "scaled_very_large.png",
    ]
]

imgs = [Image.open(i) for i in images]

# left, upper, right, lower
sz = [0, 0, *imgs[0].size]
print(sz)
s2n = 0
s2e = 410
t2e = 190

bounds = [
    [sz[0] + s2n, sz[1], sz[2] - s2e, sz[3] - t2e],
    [sz[0] + s2e, sz[1], sz[2] - s2e, sz[3] - t2e],
    [sz[0] + s2e, sz[1], sz[2] - s2n, sz[3] - t2e],
    [sz[0] + s2n, sz[1] + t2e, sz[2] - s2e, sz[3]],
    [sz[0] + s2e, sz[1] + t2e, sz[2] - s2n, sz[3]],
]


extra_labels_img = imgs[2].crop([sz[0] + s2e + (s2e - s2n), 2900, sz[2] - s2n, sz[3]])

crop = True
if crop:
    imgs = [im.crop(b) for im, b in zip(imgs, bounds)]

empty = np.zeros_like(imgs[1]) + 255
label_insert = np.array(extra_labels_img)
empty[: label_insert.shape[0], : label_insert.shape[1], :] = label_insert
row1 = np.hstack(imgs[:3])
row2 = np.hstack([*imgs[3:], empty])
out = np.vstack([row1, row2])

im = Image.fromarray(out)
im.save(image_path / "stitched_maps.png", compress_level=0, dpi=(450, 450))

legend = Image.open(image_path / "scaled_basin_mean_entropy_legend.png")

new_width = int(empty.shape[1] * 0.98)
new_height = int(legend.size[1] / legend.size[0] * new_width)
legend = legend.resize((new_width, new_height))
legend_size = np.array(legend).shape

middle_empty = (empty.shape[0] // 2, empty.shape[1] // 2)
half_legend_size = (legend_size[0] // 2, legend_size[1] // 2)
legend_start = [m - h for m, h in zip(middle_empty, half_legend_size)]
empty[
    legend_start[0] : legend_start[0] + legend_size[0],
    legend_start[1] : legend_start[1] + legend_size[1],
    :,
] = np.array(legend)

row2_w_legend = np.hstack([*imgs[3:], empty])
out = np.vstack([row1, row2_w_legend])

im = Image.fromarray(out)
im.save(image_path / "stitched_maps_w_legend.png", compress_level=0, dpi=(450, 450))
