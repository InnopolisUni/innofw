import pytest

from innofw.constants import Frameworks
from innofw.constants import Stages
from innofw.core.datamodules.lightning_datamodules.semantic_segmentation.tiff import (
    SegmentationDM,
)
from innofw.utils.framework import get_datamodule
from tests.fixtures.config.datasets import tiff_datamodule_cfg_w_target

# local modules


def test_smoke():
    # create a tiff datamodule
    fw = Frameworks.torch
    task = "image-segmentation"
    sut: SegmentationDM = get_datamodule(
        tiff_datamodule_cfg_w_target, fw, task=task
    )
    assert sut is not None

    # initialize train and test datasets
    sut.setup(Stages.train)
    assert sut.train_ds is not None
    assert sut.val_ds is not None


@pytest.mark.parametrize("stage", [Stages.train, Stages.test])
def test_train_datamodule(stage):
    # create a tiff datamodule
    fw = Frameworks.torch
    task = "image-segmentation"
    sut: SegmentationDM = get_datamodule(
        tiff_datamodule_cfg_w_target, fw, task=task
    )
    assert sut is not None

    # initialize train and test datasets
    sut.setup(stage)
    assert sut.train_ds is not None
    assert sut.val_ds is not None

    # get dataloader by stage
    dl = sut.get_stage_dataloader(stage)
    assert dl is not None
