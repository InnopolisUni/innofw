# other
import pytest
from omegaconf import DictConfig
from segmentation_models_pytorch import Unet

from innofw.constants import Frameworks
from innofw.utils.framework import (
    get_obj,
)

# local


def test_optimizer_creation():
    cfg = DictConfig(
        {
            "optimizers": {
                "task": ["all"],
                "implementations": {
                    "torch": {
                        "SGD": {
                            "object": {
                                "_target_": "torch.optim.SGD",
                                "lr": 1e-5,
                            }
                        },
                    }
                },
            }
        }
    )
    task = "image-segmentation"
    framework = Frameworks.torch
    model = Unet()

    optim = get_obj(
        cfg, "optimizers", task, framework, params=model.parameters()
    )


def test_optimizer_creation_wrong_framework():
    cfg = DictConfig(
        {
            "optimizers": {
                "task": ["all"],
                "implementations": {
                    "torch": {
                        "SGD": {
                            "object": {
                                "_target_": "torch.optim.SGD",
                                "lr": 1e-5,
                            }
                        },
                    }
                },
            }
        }
    )
    task = "image-segmentation"
    framework = Frameworks.sklearn
    model = Unet()

    with pytest.raises(ValueError):
        optim = get_obj(
            cfg, "optimizers", task, framework, params=model.parameters()
        )
