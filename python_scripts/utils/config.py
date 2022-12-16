import pathlib

PDIRS = {
    "RESOPS_PATH": pathlib.Path("~/data/ResOpsUS"),
    "PROJECT_ROOT": pathlib.Path("~/projects/plrt-conus"),
    "PROJECT_DATA": pathlib.Path("~/projects/plrt-conus/data"),
}
file_format = "feather"
FILES = {
    "RESOPS_AGG": PDIRS["PROJECT_DATA"] / "resopsus_agg" / f"sri_metric.{file_format}",
    "MODEL_READY_DATA": PDIRS["PROJECT_DATA"] / "model_ready" / f"resopsus.{file_format}",
    "MODEL_READY_META": PDIRS["PROJECT_DATA"]
    / "model_ready"
    / f"resopsus_meta.{file_format}",
}

RESOPSUS_UNTS = {
    "storage": "cubic meters",
    "release": "cubic meters per day",
    "inflow": "cubic meters per day",
}
