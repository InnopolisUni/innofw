# std
# other
import torch
from omegaconf import DictConfig

from innofw.constants import Frameworks
from innofw.utils.framework import get_losses

# local


def test_loss_creation():
    cfg = DictConfig(
        {
            "losses": {
                "task": ["image-detection"],
                "implementations": {
                    "torch": {
                        "CrossEntropyLoss": {
                            "weight": 0.5,
                            "function": {
                                "_target_": "torch.nn.functional.cross_entropy",
                            },
                        },
                        "MSE": {
                            "weight": 0.5,
                            "function": {
                                "_target_": "torch.nn.functional.mse_loss"
                            },
                        },
                    }
                },
            }
        }
    )
    task = "image-detection"
    framework = Frameworks.torch
    criterions = get_losses(cfg, task, framework)

    assert criterions is not None
    assert isinstance(criterions, list)
    assert len(criterions) == 2

    # =============== 1 ======================

    name, weight, func = criterions[0]
    assert name == "CrossEntropyLoss"
    assert weight == 0.5

    pred = torch.tensor([0.5])
    target = torch.tensor([1.0])

    pred.unsqueeze_(0)
    target.unsqueeze_(0)

    loss1 = func(pred, target)

    # ================== 2 =============
    name, weight, func = criterions[1]
    assert name == "MSE"
    assert weight == 0.5

    pred = torch.tensor([0.0, 10.0, 5.0, 7.0])
    target = torch.tensor([1.0, 9.0, 4.0, 8.0])

    pred.unsqueeze_(0)
    target.unsqueeze_(0)

    loss2 = func(pred, target)
    assert loss2 == torch.tensor(1.0)
