from pydantic import validate_arguments, FilePath
import hydra
import yaml
from hydra import compose, initialize


@validate_arguments
def read_cfg(cfg_path: FilePath, with_overriding=False):
    if with_overriding:
        initialize(config_path=str('../config'), job_name="test_app")
        cfg = compose(config_name="entry.yaml", overrides=[f"experiments={cfg_path.name}",])
        return cfg
    else:
        with open(cfg_path, "r") as f:
            cfg = yaml.safe_load(f)
            return hydra.utils.instantiate(cfg)  # todo: why this returns an object?
