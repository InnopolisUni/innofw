#
import logging
import os
import shutil
from pathlib import Path
import pytest
import yaml
from omegaconf import DictConfig

from innofw.constants import Frameworks
from innofw.utils import get_project_root
from innofw.utils.framework import get_datamodule
from innofw.utils.framework import get_experiment
from innofw.utils.framework import get_model
from tests.fixtures.config.trainers import base_trainer_on_cpu_cfg

#
#

config_path = get_project_root() / "config"

models_config_path = config_path / "models"
datasets_config_path = config_path / "datasets"
experiments_config_path = config_path / "experiments"

models_config_files = [[item] for item in models_config_path.rglob("*.yaml")]
datasets_config_files = []
experiment_config_files = [[item] for item in experiments_config_path.rglob("*.yaml")]

for item in datasets_config_path.iterdir():
    if not any([i in str(item) for i in {"_infer", "tmqm", "qm9", "brain", "lung"}]):
        datasets_config_files.append([item])


@pytest.mark.parametrize(["model_config_file"], models_config_files)
def test_models(model_config_file):
    with open(model_config_file, "r") as f:
        model_config = DictConfig(yaml.safe_load(f))
        get_model(model_config, base_trainer_on_cpu_cfg)


# @pytest.mark.skip(reason="some problems with dataset downloading")
@pytest.mark.parametrize(["dataset_config_file"], datasets_config_files)
def test_datasets(dataset_config_file, tmp_path):
    if os.path.isfile(dataset_config_file):
        with open(dataset_config_file, "r") as f:
            dataset_config = DictConfig(yaml.safe_load(f))

            for stage in ["train", "test", "infer"]:
                try:
                    dataset_config[stage]["target"] = tmp_path / stage
                except:
                    pass

            logging.info(dataset_config)

            dm = None
            task = dataset_config['task'][0]
            for framework in Frameworks:
                try:
                    dm = get_datamodule(dataset_config, framework, task)
                    break
                except Exception as e:
                    logging.exception(e)

            assert dm is not None
    else:
        pass


# @pytest.mark.skip(reason="some problems with dataset downloading")
@pytest.mark.parametrize(["experiment_config_file"], experiment_config_files)
def test_experiments(experiment_config_file):
    from hydra import compose, initialize
    from hydra.core.global_hydra import GlobalHydra

    if "example" in str(experiment_config_file):
        GlobalHydra.instance().clear()
        initialize(config_path="../../../config", job_name="test_app")

        experiment_file = f"{str(os.path.splitext(experiment_config_file)[0]).split('/experiments/')[-1]}"

        cfg = compose(
            config_name="train",
            overrides=[f"experiments={experiment_file}"],  # experiment_config_file.stem
            return_hydra_config=True,
        )
        get_experiment(cfg)
        # cfg = OmegaConf.to_yaml(cfg)
        # logging.info(cfg)
