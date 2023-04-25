# standard libraries
from pathlib import Path

import hydra
import pytest
from omegaconf import DictConfig
from segmentation_models_pytorch import Unet

import tests.fixtures.config.losses as fixt_losses
from innofw import InnoModel
from innofw.constants import Frameworks
from innofw.utils.framework import get_datamodule
from innofw.utils.framework import get_losses
from innofw.utils.framework import get_obj
from innofw.utils.framework import get_optimizer
from tests.fixtures.config.datasets import arable_segmentation_cfg_w_target
from tests.fixtures.config.trainers import base_trainer_on_cpu_cfg
from tests.utils import get_project_config_folder_path

# other
# local modules


config_files = list(
    Path(get_project_config_folder_path() / "schedulers").iterdir()
)
config_files = [[i] for i in config_files]


# test that validates correctness of every scheduler inside the schedulers config folder
@pytest.mark.parametrize(["scheduler_cfg"], config_files)
def test_scheduler_configs(scheduler_cfg, tmp_path):
    cfg = DictConfig(
        {
            "optimizers": {
                "_target_": "torch.optim.SGD",
                "lr": 1e-5,
            }
        }
    )

    task = "image-segmentation"
    framework = Frameworks.torch
    model = Unet(in_channels=4)
    optim_cfg = get_optimizer(cfg, "optimizers", task, framework)
    optim = hydra.utils.instantiate(optim_cfg, params=model.parameters())
    scheduler = get_obj(cfg, scheduler_cfg, task, framework, optimizer=optim)
    losses = get_losses(fixt_losses.jaccard_loss_w_target, task, framework)
    model_params = {
        "model": model,
        "task": task,
        "optimizers_cfg": optim_cfg,
        "schedulers_cfg": scheduler,
        "losses": losses,
        "experiment": "experiment_name",
        "trainer_cfg": base_trainer_on_cpu_cfg,
        "log_dir": tmp_path,
    }
    wrapped_model = InnoModel(**model_params)
    dm = get_datamodule(arable_segmentation_cfg_w_target, framework, task)
    wrapped_model.train(dm)
