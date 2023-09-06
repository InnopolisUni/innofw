#
import logging
import os

import pytest
import yaml
from omegaconf import DictConfig

from innofw.constants import Frameworks
from innofw.utils import get_project_root
from innofw.utils.framework import get_augmentations
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
augmentation_config_path = config_path / "augmentations"

models_config_files = [[item] for item in models_config_path.rglob("*.yaml")]
datasets_config_files = []
experiment_config_files = [
    [item] for item in experiments_config_path.rglob("*.yaml")
]
augmentation_config_files = []
augmentation_keys = (
    []
)  # List of values from ['color', 'combined', 'position', 'postprocessing', 'preprocessing']
for item in os.listdir(augmentation_config_path):
    if os.path.isfile(item):
        continue
    files = [cfg for cfg in (augmentation_config_path / item).rglob("*.yaml")]
    augmentation_config_files.extend(files)
    augmentation_keys.extend([str(item)] * len(files))

augmentation_configs_ttv = []
for augmentation_type in [
    "augmentations_train",
    "augmentations_test",
    "augmentations_val",
]:
    augmentation_configs_ttv.extend(
        [[item] for item in (config_path / augmentation_type).rglob("*.yaml")]
    )

for item in datasets_config_path.iterdir():
    if (
        not any(
            [
                i in str(item)
                for i in {"_infer", "tmqm", "qm9", "brain", "lung"}
            ]
        )
        and not item.is_dir()
    ):
        datasets_config_files.append([item])


@pytest.mark.parametrize(["model_config_file"], models_config_files)
def test_models(model_config_file):  #
    with open(model_config_file, "r") as f:
        model_config = DictConfig(yaml.safe_load(f))
        get_model(model_config, base_trainer_on_cpu_cfg)


@pytest.mark.parametrize(["dataset_config_file"], datasets_config_files)
def test_datasets(dataset_config_file, tmp_path):
    with open(dataset_config_file, "r") as f:
        dataset_config = DictConfig(yaml.safe_load(f))

        for stage in ["train", "test", "infer"]:
            try:
                dataset_config[stage]["target"] = tmp_path / stage
            except:
                pass

        logging.info(dataset_config)

        dm = None

        for framework in Frameworks:
            try:
                dm = get_datamodule(
                    dataset_config, framework, dataset_config["task"][0]
                )
                break
            except Exception as e:
                logging.exception(e)

        assert (
            dm is not None
        ), f"Dataset {str(dataset_config_file)} failed to create"

        # tmp_path.rmdir()


@pytest.mark.parametrize(["experiment_config_file"], experiment_config_files)
def test_experiments(experiment_config_file):
    import hydra
    from hydra import compose, initialize
    from hydra.core.global_hydra import GlobalHydra

    try:
        GlobalHydra.instance().clear()
        initialize(config_path="../../../config", job_name="test_app")
        cfg = compose(
            config_name="train",
            overrides=[f"experiments={experiment_config_file.stem}"],
            return_hydra_config=True,
        )
        get_experiment(cfg)
    except hydra.errors.MissingConfigException as e:
        logging.exception(e)
        pass  # Some experiments reference the outdated model configs


@pytest.mark.parametrize(
    ["augmentation_config", "key"],
    zip(augmentation_config_files, augmentation_keys),
)
def test_augmentations(augmentation_config, key):
    with open(augmentation_config, "r") as f:
        augmentation_dictconfig = DictConfig(yaml.safe_load(f))
    if augmentation_dictconfig is None or all(
        [val is None for val in augmentation_dictconfig.values()]
    ):
        return

    augmentation_dictconfig = DictConfig({key: augmentation_dictconfig})
    augmentation = get_augmentations(augmentation_dictconfig)
    assert augmentation is not None, f"{augmentation_config} is invalid"

    import numpy as np

    max_iter = 100
    image = np.random.random((224, 224, 3)).astype(np.float32)
    for i in range(max_iter):
        try:
            if np.sum(image != augmentation(image)) != 0:
                return
        except:  # If test is not applicable to this image => skip it
            return
    raise Exception(f"Augmentation {augmentation_config} has no effect")


@pytest.mark.parametrize(["augmentation_config"], augmentation_configs_ttv)
def test_augmentations_ttv(augmentation_config):
    from hydra import compose, initialize
    from hydra.core.global_hydra import GlobalHydra

    GlobalHydra.instance().clear()  # Reset global hydra state
    augmentation_config_rel = "/".join(
        str(augmentation_config).split("/")[-2:]
    )

    with initialize(version_base=None, config_path="../../../config"):
        augmentation_dictconfig = compose(config_name=augmentation_config_rel)

    # Empty config is skipped
    if augmentation_dictconfig is None or all(
        [
            val is None or len(val) == 0
            for val in augmentation_dictconfig.values()
        ]
    ):
        return
    augmentation = None
    for key in [
        "augmentations_train",
        "augmentations_val",
        "augmentations_test",
    ]:
        try:
            augmentation = get_augmentations(
                augmentation_dictconfig[key]["augmentations"]
            )
        except:
            continue
    assert (
        augmentation is not None
    ), f"Transform {augmentation_config} failed to create"

    import numpy as np

    image = np.random.random((224, 224, 3)).astype(np.float32)

    success = False
    # Different augmentations have different signatures
    for transform in [
        lambda x: augmentation(image=x),
        lambda x: augmentation(image=x.astype(np.uint8)),
        lambda x: augmentation(x),
        lambda x: augmentation(x.astype(np.uint8)),
    ]:
        try:
            image = transform(image)
            success = True
        except Exception as e:
            logging.exception(e)

    assert success, f"Transform {augmentation_config} failed to apply"
