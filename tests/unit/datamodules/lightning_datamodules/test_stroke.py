import pytest

from innofw.constants import Frameworks
from innofw.constants import Stages
from innofw.core.datamodules.lightning_datamodules.semantic_segmentation.stroke_dm import (
    StrokeSegmentationDatamodule,
)
from innofw.utils.framework import get_datamodule
from tests.fixtures.config.datasets import stroke_segmentation_datamodule_cfg_w_target

# local modules

# @pytest.mark.skip
def test_smoke():
    # create a qsar datamodule
    fw = Frameworks.torch
    task = "image-segmentation"
    dm: StrokeSegmentationDatamodule = get_datamodule(
        stroke_segmentation_datamodule_cfg_w_target, fw, task=task
    )
    assert dm is not None

    # initialize train and test datasets
    dm.setup()

    assert dm.channels_num is not None
    assert dm.val_size is not None
    assert dm.random_seed is not None 
    assert dm.train is not None
    assert dm.test is not None

# @pytest.mark.skip
@pytest.mark.parametrize("stage", [Stages.train, Stages.test])
def test_train_datamodule(stage):
    # create a qsar datamodule
    fw = Frameworks.torch
    task = "image-segmentation"
    dm: StrokeSegmentationDatamodule = get_datamodule(
        stroke_segmentation_datamodule_cfg_w_target, fw, task=task
    )
    assert dm is not None

    # initialize train and test datasets
    dm.setup(stage)

    # get dataloader by stage
    dl = dm.get_stage_dataloader(stage)
    assert dl is not None
