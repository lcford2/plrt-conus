import os
import pathlib

__HOME = pathlib.Path(os.path.expanduser("~"))

PDIRS = {
    "RESOPS_PATH": __HOME / "data" / "ResOpsUS",
    "PROJECT_ROOT": __HOME / "projects" / "plrt-conus",
    "PROJECT_DATA": __HOME / "projects" / "plrt-conus" / "data",
    "PROJECT_RESULTS": __HOME / "projects" / "plrt-conus" / "results",
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
