# standard libraries
import logging
from pathlib import Path

# third party libraries
from omegaconf import DictConfig
import pytest

# local modules
from innofw.constants import Frameworks
from innofw.utils.framework import get_datamodule, get_obj

from tests.fixtures.config.augmentations import resize_augmentation_torchvision
from tests.fixtures.config.datasets import arable_segmentation_cfg_w_target


def test_segmentation_dataset_creation():
    framework = Frameworks.torch
    dm = get_datamodule(arable_segmentation_cfg_w_target, framework, task="image-segmentation")
    assert dm
    dm.setup()
    assert iter(dm.train_dataloader()).next()


#     assert dm.train_dataset is not None and dm.test_dataset is not None
#     val_data = dm.val_dataloader()
#     test_data = dm.test_dataloader()
#     train_data = dm.train_dataloader()
#
#     assert val_data is not None and test_data is not None and train_data is not None


def test_segmentation_dataset_creation_with_augmentations():
    task = "image-segmentation"
    framework = Frameworks.torch
    augmentations = get_obj(resize_augmentation_torchvision, "augmentations", task, framework)
    dm = get_datamodule(
        arable_segmentation_cfg_w_target, framework, task=task, augmentations=augmentations
    )  # task,
    assert dm.aug is not None


def test_segmentation_dataset_creation_wrong_framework():
    task = "image-segmentation"
    framework = Frameworks.sklearn

    with pytest.raises(ValueError):
        dm = get_datamodule(arable_segmentation_cfg_w_target, framework, task=task)
