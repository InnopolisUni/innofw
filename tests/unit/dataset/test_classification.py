# standard libraries
from pathlib import Path

# third party libraries
from omegaconf import DictConfig
import pytest

# local modules
from innofw.constants import Frameworks
from innofw.utils.framework import get_datamodule, get_obj
from tests.utils import get_test_data_folder_path


def test_classification_dataset_creation():
    cfg = DictConfig(
        {
            "datasets": {
                "name": "something",
                "description": "something",
                "markup_info": "something",
                "date_time": "something",
                "task": ["image-classification"],
                "train": {
                    "source": str(
                        (
                            get_test_data_folder_path()
                            / "images/classification/office-character-classification/"
                        ).resolve()
                    )
                },
                "test": {
                    "source": str(
                        (
                            get_test_data_folder_path()
                            / "images/classification/office-character-classification/"
                        ).resolve()
                    )
                },
            },
        }
    )
    task = "image-classification"
    framework = Frameworks.torch
    dm = get_datamodule(cfg.datasets, framework, task=task)

    assert dm


def test_classification_dataset_creation_with_augmentations():
    cfg = DictConfig(
        {
            "datasets": {
                "name": "something",
                "description": "something",
                "markup_info": "something",
                "date_time": "something",
                "task": ["image-classification"],
                "train": {
                    "source": str(
                        (
                            get_test_data_folder_path()
                            / "images/classification/office-character-classification/"
                        ).resolve()
                    )
                },
                "test": {
                    "source": str(
                        (
                            get_test_data_folder_path()
                            / "images/classification/office-character-classification/"
                        ).resolve()
                    )
                },
            },
            "augmentations": {
                "task": ["image-classification"],
                "implementations": {
                    "torch": {
                        "Compose": {
                            "object": {
                                "_target_": "albumentations.Compose",
                                "transforms": [
                                    {
                                        "_target_": "albumentations.Resize",
                                        "height": 300,
                                        "width": 300,
                                        "always_apply": True,
                                    },
                                    {"_target_": "albumentations.Flip"},
                                ],
                            }
                        }
                    },
                },
            },
        }
    )
    task = "image-classification"
    framework = Frameworks.torch
    augmentations = get_obj(cfg, "augmentations", task, framework)
    dm = get_datamodule(cfg.datasets, framework, augmentations=augmentations, task=task)
    assert dm
    dm.setup()

    assert iter(dm.train_dataloader()).next()
    assert dm.aug is not None


def test_classification_dataset_creation_wrong_framework():
    cfg = DictConfig(
        {
            "datasets": {
                "name": "something",
                "description": "something",
                "markup_info": "something",
                "date_time": "something",
                "task": ["image-classification"],
                "train": {
                    "source": str(
                        (
                            get_test_data_folder_path()
                            / "images/classification/office-character-classification/"
                        ).resolve()
                    )
                },
                "test": {
                    "source": str(
                        (
                            get_test_data_folder_path()
                            / "images/classification/office-character-classification/"
                        ).resolve()
                    )
                },
            },
        }
    )
    task = "image-classification"
    framework = Frameworks.sklearn

    with pytest.raises(ValueError):
        dm = get_datamodule(cfg.datasets, framework, task=task)
