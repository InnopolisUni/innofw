from multiprocessing.sharedctypes import Value
from typing import Optional
from pydantic import validate_arguments, FilePath
import hydra
import yaml
from hydra import compose, initialize


@validate_arguments
def read_cfg(cfg_path: Optional[FilePath]=None, cfg_name="train.yaml", overrides=None):
    if overrides is None and cfg_path is not None:
        with open(cfg_path, "r") as f:
            cfg = yaml.safe_load(f)
            # del cfg['name']  # todo:?
            return hydra.utils.instantiate(cfg)  # todo: why this returns an object?
    elif overrides is not None and cfg_path is None:
        initialize(config_path=str('../../config'), job_name="test_app")
        cfg = compose(config_name=cfg_name, overrides=overrides)
        return cfg
    else:
        raise ValueError("wrong set of arguments is provided")
