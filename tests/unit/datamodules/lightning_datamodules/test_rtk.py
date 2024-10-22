from innofw.core.datamodules.lightning_datamodules.detection_coco import (
    DicomCocoComplexingDataModule,
    DicomCocoDataModuleRTK,
)
from innofw.core.datasets.coco import DicomCocoDatasetRTK


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
