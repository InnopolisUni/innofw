# other
import pytest
from omegaconf import DictConfig

# local
from innofw.constants import Frameworks
from innofw.utils.framework import get_losses


def test_loss_creation():
    cfg = DictConfig(
        {
            "losses": {
                "task": ["image-segmentation"],
                "implementations": {
                    "torch": {
                        "JaccardLoss": {
                            "weight": 0.3,
                            "object": {
                                "_target_": "pytorch_toolbelt.losses.JaccardLoss",
                                "mode": "binary",
                                "from_logits": False,
                            },
                        },
                        "BinaryFocalLoss": {
                            "weight": 0.7,
                            "object": {
                                "_target_": "pytorch_toolbelt.losses.BinaryFocalLoss"
                            },
                        },
                    }
                },
            }
        }
    )
    task = "image-segmentation"
    framework = Frameworks.torch
    criterions = get_losses(cfg, task, framework)

    assert criterions is not None
    assert isinstance(criterions, list)
    assert len(criterions) == 2

    name, weight, func = criterions[0]
    assert name == "JaccardLoss"
    assert weight == 0.3

    import torch

    pred = torch.tensor([0.0, 0.0, 0.0])
    target = torch.tensor([1, 1, 1])

    pred.unsqueeze_(0)
    target.unsqueeze_(0)

    loss1 = func(pred, target)
    assert loss1 == torch.tensor(1.0)


def test_loss_creation_wrong_task():
    cfg = DictConfig(
        {
            "losses": {
                "task": ["image-classification"],
                "implementations": {
                    "torch": {
                        "JaccardLoss": {
                            "weight": 0.3,
                            "object": {
                                "_target_": "pytorch_toolbelt.losses.JaccardLoss",
                                "mode": "binary",
                            },
                        },
                        "BinaryFocalLoss": {
                            "weight": 0.7,
                            "object": {
                                "_target_": "pytorch_toolbelt.losses.BinaryFocalLoss"
                            },
                        },
                    }
                },
            }
        }
    )
    task = "image-segmentation"
    framework = Frameworks.torch

    with pytest.raises(ValueError):
        loss = get_losses(cfg, task, framework)


def test_loss_creation_wrong_framework():
    cfg = DictConfig(
        {
            "losses": {
                "task": ["image-segmentation"],
                "implementations": {
                    "torch": {
                        "JaccardLoss": {
                            "weight": 0.3,
                            "object": {
                                "_target_": "pytorch_toolbelt.losses.JaccardLoss",
                                "mode": "binary",
                            },
                        },
                        "BinaryFocalLoss": {
                            "weight": 0.7,
                            "object": {
                                "_target_": "pytorch_toolbelt.losses.BinaryFocalLoss"
                            },
                        },
                    }
                },
            }
        }
    )
    task = "image-segmentation"
    framework = Frameworks.sklearn

    with pytest.raises(ValueError):
        loss = get_losses(cfg, task, framework)
