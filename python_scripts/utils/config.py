import os
import pathlib


class ConfigObject:
    def __init__(self):
        _home = pathlib.Path(os.path.expanduser("~"))
        _project_root = _home / "projects" / "plrt-conus"
        self.dirs = {
            "root": _project_root,
            "general_data": _home / "data",
            "resops": _home / "data" / "ResOpsUS",
            "data": _project_root / "data",
            "spatial_data": _project_root / "data" / "spatial_data",
            "model_ready_data": _project_root / "data" / "model_ready",
            "results": _project_root / "results",
            "agg_results": _project_root / "aggregated_results",
            "data_to_sync": _project_root / "data" / "to_sync",
        }

        self.pandas_format = "feather"

        self.files = {
            "resops_agg": self.dirs["data"]
            / "resopsus_agg"
            / f"sri_metric.{self.pandas_format}",
            "model_ready_data": self.dirs["model_ready_data"]
            / f"resopsus.{self.pandas_format}",
            "model_ready_meta": self.dirs["model_ready_data"]
            / f"resopsus_meta.{self.pandas_format}",
            "merged_data": self.dirs["model_ready_data"]
            / f"merged_data.{self.pandas_format}",
            "merged_meta": self.dirs["model_ready_data"]
            / f"merged_data.{self.pandas_format}",
        }

        self.resopsus_unts = {
            "storage": "cubic meters",
            "release": "cubic meters per day",
            "inflow": "cubic meters per day",
        }

        self.__set_file_attrs()
        self.__set_folder_attrs()

    def __set_folder_attrs(self):
        for key, value in self.dirs.items():
            attr_name = f"d_{key}"
            setattr(self, attr_name, value)

    def __set_file_attrs(self):
        for key, value in self.files.items():
            attr_name = f"f_{key}"
            setattr(self, attr_name, value)

    def get_dir(self, dir_key):
        if dir_key not in self.dirs:
            raise KeyError(f"{dir_key} is not valid.")
        return self.dirs[dir_key]

    def get_file(self, file_key):
        if file_key not in self.files:
            raise KeyError(f"{file_key} is not valid.")
        return self.files[file_key]


config = ConfigObject()
