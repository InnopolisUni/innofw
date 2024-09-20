import pytest

from innofw.constants import Frameworks
from innofw.constants import Stages
from innofw.core.datamodules.lightning_datamodules.semantic_segmentation.hdf5 import (
                                                        HDF5LightningDataModule, 
)
from innofw.utils.framework import get_datamodule
from tests.fixtures.config.datasets import arable_segmentation_cfg_w_target

# local modules


def test_smoke():
    # create a datamodule
    fw = Frameworks.torch
    task = "image-segmentation"
    dm: HDF5LightningDataModule = get_datamodule(
        arable_segmentation_cfg_w_target, fw, task=task
    )
    assert dm is not None

    # initialize train and test datasets
    dm.setup #(Stages.train)
    assert dm.channels_num is not None
    assert dm.val_size is not None
    assert dm.random_seed is not None 
    assert dm.w_sampler is not None
    assert dm.train is not None
    assert dm.test is not None



@pytest.mark.parametrize("stage", [Stages.train]) #, Stages.test])
def test_train_datamodule(stage):
    # create  datamodule
    fw = Frameworks.torch
    task = "image-segmentation"
    dm: HDF5LightningDataModule = get_datamodule(
        arable_segmentation_cfg_w_target, fw, task=task
    )
    assert dm is not None

    # initialize train and test datasets
    dm.setup(stage)


    # get dataloader by stage
    dl = dm.get_stage_dataloader(stage)
    assert dl is not None
