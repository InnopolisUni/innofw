# other
from omegaconf import DictConfig
import torch
import numpy as np

# local
from innofw.constants import Frameworks
from innofw.utils.framework import get_losses


def test_sklearn_loss_creation():
    cfg = DictConfig(
        {
            "losses": {
                "task": ["table-regression", "image-regression"],
                "implementations": {
                    "sklearn": {
                        "mse": {
                            "weight": 1.0,
                            "function": {
                                "_target_": "sklearn.metrics.mean_squared_error",
                            },
                        },
                    },
                    "torch": {
                        "mse": {
                            "weight": 1.0,
                            "object": {
                                "_target_": "torch.nn.MSELoss",
                            },
                        },
                    },
                },
            }
        }
    )
    task = "table-regression"
    framework = Frameworks.sklearn
    loss = get_losses(cfg, task, framework)

    assert loss is not None
    assert isinstance(loss, list)
    assert len(loss) == 1

    name, weight, func = loss[0]
    assert name == "mse"
    assert weight == 1.0

    pred = np.array([0, 0, 0])
    target = np.array([2, 2, 2])

    loss = func(pred, target)
    assert loss == 4


def test_torch_loss_creation():
    cfg = DictConfig(
        {
            "losses": {
                "task": ["table-regression", "image-regression"],
                "implementations": {
                    "sklearn": {
                        "mse": {
                            "weight": 1.0,
                            "function": {
                                "_target_": "sklearn.metrics.mean_squared_error",
                            },
                        },
                    },
                    "torch": {
                        "mse": {
                            "weight": 1.0,
                            "object": {
                                "_target_": "torch.nn.MSELoss",
                            },
                        },
                    },
                },
            }
        }
    )
    task = "table-regression"
    framework = Frameworks.torch
    loss = get_losses(cfg, task, framework)

    assert loss is not None
    assert isinstance(loss, list)
    assert len(loss) == 1

    pred = torch.tensor([0.0, 0.0, 0.0])
    target = torch.tensor([2.0, 2.0, 2.0])

    name, weight, func = loss[0]
    assert name == "mse"
    assert weight == 1.0

    loss = func(pred, target)

    assert loss == torch.tensor(4)
