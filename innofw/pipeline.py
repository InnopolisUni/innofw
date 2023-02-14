# standard library
import logging
from uuid import uuid4
from pathlib import Path
from typing import Optional

# third party packages
from pytorch_lightning import seed_everything
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf, DictConfig

# local modules
from innofw.utils.framework import (
    get_callbacks,
    get_optimizer,
    get_ckpt_path,
    get_datamodule,
    get_losses,
    get_model,
    get_obj,
    map_model_to_framework,
)
from innofw.constants import Stages


# from innofw.utils.clear_ml import setup_clear_ml
from innofw import InnoModel
from innofw.utils.getters import get_trainer_cfg, get_log_dir, get_a_learner
from innofw.utils.print_config import print_config_tree
from innofw.utils.defaults import default_model_for_datamodule
import hydra


def run_pipeline(
        cfg: DictConfig,
        train=True,
        test=False,
        predict=False,
        log_root: Optional[Path] = None,
) -> float:
    print_config_tree(cfg)
    try:
        experiment_name = cfg.experiment_name
    except ValueError:  # hydra config not set, happens when run_pipeline is called from tests
        experiment_name = str(uuid4()).split("-")[0]

    if cfg.get("random_seed"):
        seed_everything(
            cfg.random_seed, workers=True
        )

    stage = (
        "train" if train else "test" if test else "infer"
    )
    project = cfg.get("project")
    data_stage = Stages.predict if predict else Stages.train
    trainer_cfg = get_trainer_cfg(cfg)
    task = cfg.get("task")
    # Model
    if 'models' in cfg:
        model = get_model(cfg.models, trainer_cfg)
    else:
        datamodule = cfg.datasets.get('_target_')
        if datamodule is None:
            raise ValueError("wrong configuration: no model and dataset _target_ specified")
        else:
            model = hydra.utils.instantiate(default_model_for_datamodule(task, datamodule))

    framework = map_model_to_framework(model)
    # weights initialization
    initializations = get_obj(
        cfg, "initializations", task, framework, _recursive_=False
    )

    augmentations = get_obj(cfg, "augmentations", task, framework)
    metrics = get_obj(cfg, "metrics", task, framework)
    optimizers = get_optimizer(cfg, "optimizers", task, framework)
    schedulers = get_obj(cfg, "schedulers", task, framework)
    datamodule = get_datamodule(
        cfg.datasets,
        framework,
        task=task,
        stage=data_stage,
        augmentations=augmentations,
        batch_size=cfg.get("batch_size"),
    )
    losses = get_losses(cfg, task, framework)
    callbacks = get_callbacks(
        cfg, task, framework, metrics=metrics, losses=losses, datamodule=datamodule
    )

    log_dir = get_log_dir(project, stage, experiment_name, log_root=log_root)

    # wrap the model
    model_params = {
        "model": model,
        "task": task,
        "trainer_cfg": trainer_cfg,
        "optimizers_cfg": optimizers,
        "schedulers_cfg": schedulers,
        "losses": losses,
        "callbacks": callbacks,
        "initializations": initializations,
        "log_dir": log_dir,
        "experiment": experiment_name,
        "project": project,
        "stop_param": cfg.get("stop_param"),
        "weights_path": cfg.get("weights_path"),
        "weights_freq": cfg.get("weights_freq"),
    }

    inno_model = InnoModel(**model_params)
    result = None

    stages = []
    if train:
        stages.append(Stages.train)
    if test:
        stages.append(Stages.test)
    if predict:
        stages.append(Stages.predict)

    stage_to_func = {
        Stages.test: inno_model.test,
        Stages.predict: inno_model.predict,
        Stages.train: inno_model.train,
    }

    if "extra" in cfg and "active_learning" in cfg.extra:
        a_learner = get_a_learner(cfg, inno_model, datamodule)
        stage_to_func[Stages.train] = a_learner.run

    ckpt_path = get_ckpt_path(cfg)
    logging.info(f"Using checkpoint: {ckpt_path}")
    for stage in stages:
        result = stage_to_func[stage](datamodule, ckpt_path=ckpt_path)

    return result
