import pytest

from innofw.constants import Frameworks
from innofw.utils.framework import get_datamodule
from tests.fixtures.config.datasets import (
    wheat_datamodule_cfg_w_target,
    dicom_datamodule_cfg_w_target,
)


def test_coco_detection_dataset():
    framework = Frameworks.torch
    task = "image-detection"
    dm = get_datamodule(wheat_datamodule_cfg_w_target, framework, task=task)
    assert dm
    dm.setup()
    assert iter(dm.train_dataloader()).next()


def test_dicom_coco_detection_dataset():
    framework = Frameworks.torch
    task = "image-detection"
    dm = get_datamodule(dicom_datamodule_cfg_w_target, framework, task=task)
    assert dm
    dm.setup()
    assert iter(dm.train_dataloader()).next()
