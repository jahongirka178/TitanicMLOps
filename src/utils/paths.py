from pathlib import Path
import yaml


class ConfigPaths:
    def __init__(self, config_path=None):
        if config_path is None:
            base = Path(__file__).resolve().parents[2]
            config_path = base / "config.yaml"

        with open(config_path, "r", encoding="utf-8") as f:
            self.cfg = yaml.safe_load(f)

        self.base_dir = Path(self.cfg["paths"]["base_dir"]).resolve()

    def get_path_to_data(self):
        return self.base_dir / self.cfg["paths"]["data_path"]

    def get_path_to_artifacts(self):
        return self.base_dir / self.cfg["paths"]["artifact_path"]

    def get_path_to_catboost_info(self):
        return self.base_dir / self.cfg["paths"]["catboost_info_path"]
