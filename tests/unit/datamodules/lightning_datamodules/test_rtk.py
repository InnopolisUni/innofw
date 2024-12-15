import os
import shutil

import pytest

from innofw.core.datamodules.lightning_datamodules.coco_rtk import (
    DicomCocoComplexingDataModule,
    DicomCocoDataModuleRTK,
)
from innofw.core.datasets.coco_rtk import DicomCocoDatasetRTK
from innofw.core.datamodules.pandas_datamodules.lung_description_decision_datamodule import (
    LungDescriptionDecisionPandasDataModule,
)

rtk_complex = "https://api.blackhole.ai.innopolis.university/public-datasets/rtk/complex_infer.zip"
rtk_segm = "https://api.blackhole.ai.innopolis.university/public-datasets/rtk/infer.zip"
lungs = "https://api.blackhole.ai.innopolis.university/public-datasets/rtk/labels.zip"


def test_DicomCocoComplexingDataModule():
    target_dir = "./data/complex/infer"
    if os.path.exists(target_dir):
        shutil.rmtree(target_dir)
    path = {"source": rtk_complex, "target": target_dir}
    dm = DicomCocoComplexingDataModule(infer=path)
    dm.setup_infer()
    ds = dm.predict_dataloader()
    for batch in ds:
        break
    for k in ["image", "mask", "path"]:
        assert k in batch


def test_DicomCocoDataModuleRTK():
    target_dir = "./data/rtk/infer"
    if os.path.exists(target_dir):
        shutil.rmtree(target_dir)

    path = {"source": rtk_segm, "target": target_dir}
    dm = DicomCocoDataModuleRTK(infer=path)
    dm.setup_infer()
    ds = dm.predict_dataloader()
    for batch in ds:
        break
    # for k in ["image", "path"]:
    #     assert k in batch
    #

def test_DicomCocoDataset_rtk():
    """
    import this test to run after previous
    """
    path = "./data/rtk/infer"
    ds = DicomCocoDatasetRTK(data_dir=path)
    for batch in ds:
        break
    for k in ["image", "mask", "path"]:
        assert k in batch


def test_datamodule_description():
    target_dir = "./data/lung_description/infer"
    if os.path.exists(target_dir):
        shutil.rmtree(target_dir)
    path = {"source": lungs, "target": target_dir}
    dm = LungDescriptionDecisionPandasDataModule(infer=path)
    dm.setup_infer()
    ds = dm.predict_dataloader()
    for key in "x", "y":
        assert key in ds
