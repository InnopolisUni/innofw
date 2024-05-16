import pytest
from pathlib import Path

from innofw.constants import Frameworks
from innofw.constants import Stages, SegDataKeys
from innofw.core.datamodules.lightning_datamodules.semantic_segmentation.stroke_dm import (
    DirSegmentationLightningDataModule,
    StrokeSegmentationDatamodule,
)
from innofw.utils.framework import get_datamodule
from tests.fixtures.config.datasets import pngstroke_segmentation_datamodule_cfg_w_target, dirstroke_segmentation_datamodule_cfg_w_target, dicomstroke_segmentation_datamodule_cfg_w_target

# local modules

# @pytest.mark.skip
# def test_smoke():
#     # create a qsar datamodule
#     fw = Frameworks.torch
#     task = "image-segmentation"
#     dm: StrokeSegmentationDatamodule = get_datamodule(
#         stroke_segmentation_datamodule_cfg_w_target, fw, task=task
#     )
#     assert dm is not None
#
#     # initialize train and test datasets
#     dm.setup()
#
#     assert dm.channels_num is not None
#     assert dm.val_size is not None
#     assert dm.random_seed is not None
#     assert dm.train is not None
#     assert dm.test is not None
#
# # @pytest.mark.skip
# @pytest.mark.parametrize("stage", [Stages.train, Stages.test])
# def test_train_datamodule(stage):
#     # create a qsar datamodule
#     fw = Frameworks.torch
#     task = "image-segmentation"
#     dm: StrokeSegmentationDatamodule = get_datamodule(
#         stroke_segmentation_datamodule_cfg_w_target, fw, task=task
#     )
#     assert dm is not None
#
#     # initialize train and test datasets
#     dm.setup(stage)
#
#     # get dataloader by stage
#     dl = dm.get_stage_dataloader(stage)
#     assert dl is not None


def test_DirSegmentationLightningDataModule_AND_SegmentationDataset_AND_setup_train_test_val():
    fw = Frameworks.torch
    task = "image-segmentation"
    dm: DirSegmentationLightningDataModule = get_datamodule(
        dirstroke_segmentation_datamodule_cfg_w_target, fw, task=task
    )
    assert dm is not None

    # initialize train and test datasets
    dm.setup()

    el = dm.train_dataset[0]

    assert dm.channels_num is not None
    assert dm.val_size is not None
    assert dm.random_seed is not None
    assert dm.train is not None
    assert dm.test is not None
    assert el is not None


def test_StrokeSegmentationDatamodule_AND_StrokeSegmentationDataset_AND_setup_train_test_val_AND_save_preds_AND_setup_infer():
    fw = Frameworks.torch
    task = "image-segmentation"
    dm: StrokeSegmentationDatamodule = get_datamodule(
        pngstroke_segmentation_datamodule_cfg_w_target, fw, task=task
    )
    assert dm is not None

    # initialize train and test datasets
    dm.setup()
    dm.setup_infer()


    el = dm.train_dataset[0]
    test_el = dm.test_dataset[0]
    dm.save_preds([test_el[SegDataKeys.label].data],  Stages.test, Path("/tmp"))


    assert dm.channels_num is not None
    assert dm.val_size is not None
    assert dm.random_seed is not None
    assert dm.train is not None
    assert dm.test is not None
    assert el is not None
    assert len(dm.train_dataset) != 0


def test_DicomDirSegmentationLightningDataModule_AND_setup_train_test_val_AND_save_preds_AND_setup_infer():
    fw = Frameworks.torch
    task = "image-segmentation"
    dm: StrokeSegmentationDatamodule = get_datamodule(
        dicomstroke_segmentation_datamodule_cfg_w_target, fw, task=task
    )
    assert dm is not None

    # initialize train and test datasets
    dm.setup()
    dm.setup_infer()

    el = dm.train_dataset[0]
    test_el = dm.test_dataset[0]
    dm.save_preds([test_el[SegDataKeys.label].data], Stages.test, Path("/tmp"))

    assert dm.channels_num is not None
    assert dm.val_size is not None
    assert dm.random_seed is not None
    assert dm.train is not None
    assert dm.test is not None
    assert el is not None
    assert len(dm.train_dataset) != 0