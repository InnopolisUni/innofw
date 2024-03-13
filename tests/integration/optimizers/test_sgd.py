# other
import hydra
import pytest
import torch.optim
from omegaconf import DictConfig
from segmentation_models_pytorch import Unet

from innofw.constants import Frameworks
from innofw.utils.framework import (
    get_optimizer,
)

# local


def test_optimizer_creation():
    cfg = DictConfig(
        {
            "optimizers": {
                "_target_": "innofw.core.optimizers.custom_optimizers.optimizers.SGD",
                "lr": 1e-5,
            }
        }
    )
    task = "image-segmentation"
    framework = Frameworks.torch
    model = Unet()

    optim_cfg = get_optimizer(
        cfg, "optimizers", task, framework, params=model.parameters()
    )
    optim = hydra.utils.instantiate(optim_cfg, params=model.parameters())
    assert optim is not None and isinstance(optim, torch.optim.Optimizer)


def test_optimizer_creation_wrong_framework():
    cfg = DictConfig(
        {
            "optimizers": {
                "_target_": "innofw.core.optimizers.custom_optimizers.optimizers.SGD",
                "lr": 1e-5,
            }
        }
    )
    task = "image-segmentation"
    framework = Frameworks.sklearn
    model = Unet()

    with pytest.raises(ValueError):
        optim_cfg = get_optimizer(cfg, "optimizers", task, framework)
