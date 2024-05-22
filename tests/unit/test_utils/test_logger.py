from innofw.utils.loggers import setup_wandb, setup_clear_ml

import pytest
from omegaconf import DictConfig


@pytest.mark.skip(reason="clearml not set yet")
def test_clearml():
    config = DictConfig({"clear_ml": {"enable": True, "queue": None},
     "experiment_name": "name", "project": "sample"})
    assert setup_clear_ml(config) is not None

# def test_wandb():
#   config = DictConfig({"wandb": {"enable": True, "project": "tmp", "entity": "k-galliamov", "group": None}})
#   assert setup_wandb(config) is not None

def test_wandb_empty_cfg():
    assert setup_wandb([]) is None