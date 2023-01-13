import os
import pathlib

__HOME = pathlib.Path(os.path.expanduser("~"))


class ConfigObject:
    def __init__(self):
        _project_root = __HOME / "projects" / "plrt-conus"
        self.dirs = {
            "root": _project_root,
            "general_data": __HOME / "data",
            "resops": __HOME / "data" / "ResOpsUS",
            "data": _project_root / "data",
            "spatial_data": _project_root / "data" / "spatial_data",
            "results": _project_root / "results",
            "agg_results": _project_root / "aggregated_results",
        }

        self.pandas_format

        self.files = {
            "resops_agg": self.folders["data"]
            / "resopsus_agg"
            / f"sri_metric.{self.pandas_format}",
            "model_ready_data": self.folders["data"]
            / "model_ready"
            / f"resopsus.{self.pandas_format}",
            "model_ready_meta": self.folders["data"]
            / "model_ready"
            / f"resopsus_meta.{self.pandas_format}",
        }

        self.resopsus_unts = {
            "storage": "cubic meters",
            "release": "cubic meters per day",
            "inflow": "cubic meters per day",
        }

        self.__set_file_attrs()
        self.__set_folder_attrs()

    def __set_folder_attrs(self):
        for key, value in self.dirs:
            attr_name = f"d_{key}"
            setattr(self, attr_name, value)

    def __set_file_attrs(self):
        for key, value in self.files:
            attr_name = f"f_{key}"
            setattr(self, attr_name, value)

    def get_dir(self, dir_key):
        if dir_key not in self.dirs:
            raise KeyError(f"{dir_key} is not valid.")
        return self.folders[dir_key]

    def get_file(self, file_key):
        if file_key not in self.files:
            raise KeyError(f"{file_key} is not valid.")
        return self.file[file_key]


config = ConfigObject()
