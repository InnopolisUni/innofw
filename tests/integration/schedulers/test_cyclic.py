# other
import hydra
from omegaconf import DictConfig
from segmentation_models_pytorch import Unet

from innofw.constants import Frameworks
from innofw.utils.framework import get_obj
from innofw.utils.framework import get_optimizer


def test_scheduler_creation():
    cfg = DictConfig(
        {
            "optimizers": {
                "_target_": "torch.optim.SGD",
                "lr": 1e-5,
            },
            "schedulers": {
                "task": ["all"],
                "implementations": {
                    "torch": {
                        "CosineAnnealingWarmRestarts": {
                            "object": {
                                "_target_": "torch.optim.lr_scheduler.CosineAnnealingWarmRestarts",
                                "T_0": 10,
                                "T_mult": 1,
                                "eta_min": 3e-6,
                            }
                        },
                    }
                },
            },
        }
    )
    task = "image-segmentation"
    framework = Frameworks.torch
    model = Unet()
    optim_cfg = get_optimizer(cfg, "optimizers", task, framework)
    optim = hydra.utils.instantiate(optim_cfg, params=model.parameters())
    scheduler = get_obj(cfg, "schedulers", task, framework, optimizer=optim)
