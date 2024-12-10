import pytest

from innofw.core.datamodules.lightning_datamodules.coco_rtk import (
    DicomCocoComplexingDataModule,
    DicomCocoDataModuleRTK,
)
from innofw.core.datasets.coco_rtk import DicomCocoDatasetRTK
from innofw.core.datamodules.pandas_datamodules.lung_description_decision_datamodule import (
    LungDescriptionDecisionPandasDataModule,
)


def test_DicomCocoDataset_rtk():
    path = "./data/rtk/infer"
    ds = DicomCocoDatasetRTK(data_dir=path)
    for batch in ds:
        break
    for k in ["image", "mask", "path"]:
        assert k in batch


def test_DicomCocoComplexingDataModule():
    path = {"source": "./data/complex/infer", "target": "./data/complex/infer"}
    dm = DicomCocoComplexingDataModule(infer=path)
    dm.setup_infer()
    ds = dm.predict_dataloader()
    for batch in ds:
        break
    for k in ["image", "mask", "path"]:
        assert k in batch


def test_DicomCocoDataModuleRTK():
    path = {"source": "./data/rtk/infer", "target": "./data/rtk/infer"}
    dm = DicomCocoDataModuleRTK(infer=path)
    dm.setup_infer()
    ds = dm.predict_dataloader()
    for batch in ds:
        break
    for k in ["image", "mask", "path"]:
        assert k in batch


def test_datamodule_description():
    path = {
        "source": "./data/lung_description/infer",
        "target": "./data/lung_description/infer",
    }

    dm = LungDescriptionDecisionPandasDataModule(infer=path)
    dm.setup_infer()
    ds = dm.predict_dataloader()
    for key in "x", "y":
        assert key in ds
