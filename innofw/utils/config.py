from typing import Optional

import hydra
import yaml
from hydra import compose
from hydra import initialize
from pydantic import FilePath
from pydantic import validate_arguments


def read_cfg_2_dict(cfg_path):
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg


@validate_arguments
def read_cfg(
    cfg_path: Optional[FilePath] = None, cfg_name="train.yaml", overrides=None
):
    if overrides is None and cfg_path is not None:
        with open(cfg_path, "r") as f:
            cfg = yaml.safe_load(f)
            # del cfg['name']  # todo:?
            return hydra.utils.instantiate(cfg)  # todo: why this returns an object?
    elif overrides is not None and cfg_path is None:
        initialize(config_path=str("../../config"), job_name="test_app")
        cfg = compose(config_name=cfg_name, overrides=overrides)
        return cfg
    else:
        raise ValueError("wrong set of arguments is provided")
