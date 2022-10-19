import os
import datetime
from pathlib import Path
from typing import Optional

from omegaconf import open_dict, DictConfig

from innofw.core.active_learning import ActiveLearnTrainer


def get_trainer_cfg(cfg):
    trainer = cfg.get("trainer")
    epochs = cfg.get("epochs")
    accelerator = cfg.get("accelerator")
    gpus = cfg.get("gpus")
    devices = cfg.get("devices")

    if trainer is not None:
        trainer_cfg = cfg.trainer  # .objects
        with open_dict(trainer_cfg):
            trainer_cfg.max_epochs = epochs
            trainer_cfg.accelerator = accelerator
            trainer_cfg.gpus = gpus
            trainer_cfg.devices = devices
    else:
        trainer_cfg = DictConfig(
            {
                "max_epochs": epochs,
                "accelerator": accelerator,
                "gpus": gpus,
                "devices": devices,
            }
        )
    return trainer_cfg


def get_log_dir(project, stage, experiment_name, log_root=None):
    log_root = os.getcwd() if log_root is None else log_root
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    log_dir = Path(log_root, stage, project, experiment_name, current_time)
    Path(log_dir).mkdir(exist_ok=False, parents=True)
    return log_dir


def get_a_learner(cfg, inno_model, datamodule):
    a_learner_params = {
        "epochs_num": cfg.extra.active_learning.get("epochs_num", 100),
        "query_size": cfg.extra.active_learning.get("query_size", 1),
        "use_data_uncertainty": cfg.extra.active_learning.get(
            "use_data_uncertainty", True
        ),
    }
    a_learner = ActiveLearnTrainer(
        model=inno_model, datamodule=datamodule, **a_learner_params
    )
    return a_learner
