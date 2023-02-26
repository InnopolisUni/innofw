import os

from omegaconf import OmegaConf
from hydra.core.hydra_config import HydraConfig
import pathlib
import yaml

TASK = None

def setup_clear_ml(cfg):
    if "clear_ml" not in cfg:
        return
    clear_ml_cfg = cfg.get("clear_ml")
    if clear_ml_cfg and clear_ml_cfg.get("enable"):
        from clearml import Task

        task = Task.init(project_name=cfg["project"], task_name=cfg["experiment_name"])
        setup_agent(task, clear_ml_cfg)
        global TASK
        TASK = task
        task.connect(OmegaConf.to_container(cfg, resolve=True))
        return task


def setup_agent(task, cfg):
    if cfg["queue"]:
        task.execute_remotely(queue_name=cfg["queue"])
