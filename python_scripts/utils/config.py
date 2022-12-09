import pathlib

PDIRS = {
    "RESOPS_PATH": pathlib.Path("~/data/ResOpsUS"),
    "PROJECT_ROOT": pathlib.Path("~/projects/plrt-conus"),
    "PROJECT_DATA": pathlib.Path("~/projects/plrt-conus/data"),
}

FILES = {
    "RESOPS_AGG": PDIRS["PROJECT_DATA"] / "resopsus_agg" / "sri_metric.pickle",
    "MODEL_READY_DATA": PDIRS["PROJECT_DATA"] / "model_ready" / "resopsus.pickle",
    "MODEL_READY_META": PDIRS["PROJECT_DATA"] / "model_ready" / "resopsus_meta.pickle",
}

RESOPSUS_UNTS = {
    "storage": "cubic meters",
    "release": "cubic meters per day",
    "inflow": "cubic meters per day",
}
