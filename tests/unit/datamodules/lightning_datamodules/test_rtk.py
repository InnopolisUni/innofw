from unittest.mock import patch
import os
import shutil

import pytest
import numpy as np

from innofw.core.datamodules.lightning_datamodules.coco_rtk import (
    DEFAULT_TRANSFORM,
    DicomCocoComplexingDataModule,
    DicomCocoDataModuleRTK,
)
from innofw.core.datamodules.pandas_datamodules.lung_description_decision_datamodule import (
    LungDescriptionDecisionPandasDataModule,
)
from innofw.core.datasets.coco_rtk import DicomCocoDatasetRTK
from innofw.utils.data_utils.preprocessing.CT_hemorrhage_contrast_rtk import (
    hemorrhage_contrast,
    transform as resize_transform,
)
from innofw.utils.data_utils.preprocessing.CT_hemorrhage_contrast_metrics import (
    hemorrhage_contrast_metrics,
)
from innofw.utils.data_utils.rtk.CT_hemorrhage_metrics import process_metrics

rtk_complex = "https://api.blackhole.ai.innopolis.university/public-datasets/rtk/complex_infer.zip"
rtk_segm = "https://api.blackhole.ai.innopolis.university/public-datasets/rtk/infer.zip"
lungs = "https://api.blackhole.ai.innopolis.university/public-datasets/rtk/labels.zip"


transforms = [None, DEFAULT_TRANSFORM]


@pytest.mark.parametrize("transform", transforms)
def test_DicomCocoComplexingDataModule(transform):
    target_dir = "./data/complex/infer"
    if os.path.exists(target_dir):
        shutil.rmtree(target_dir)
    path = {"source": rtk_complex, "target": target_dir}
    dm = DicomCocoComplexingDataModule(infer=path, transform=transform)
    dm.setup_infer()
    ds = dm.predict_dataloader()
    for batch in ds:
        break
    for k in ["image", "mask", "path"]:
        assert k in batch, f"no suck key {k}"


@pytest.mark.parametrize("transform", transforms)
def test_DicomCocoDataModuleRTK(transform):
    target_dir = "./data/rtk/infer"
    if os.path.exists(target_dir):
        shutil.rmtree(target_dir)

    path = {"source": rtk_segm, "target": target_dir}
    dm = DicomCocoDataModuleRTK(infer=path, transform=resize_transform)
    dm.setup_infer()
    ds = dm.predict_dataloader()
    for batch in ds:
        break
    for k in ["image", "path"]:
        assert k in batch, f"no suck key {k}"


@pytest.mark.parametrize("transform", transforms)
def test_DicomCocoDataset_rtk(transform):
    """
    import this test to run after test_DicomCocoDataModuleRTK
    """
    path = "./data/rtk/infer"
    ds = DicomCocoDatasetRTK(data_dir=path, transform=transform)
    for batch in ds:
        break
    for k in ["image", "mask", "path"]:
        assert k in batch, f"no suck key {k}"


@pytest.mark.parametrize("transform", transforms)
def test_DicomCocoDataset_rtk_no_annotation(transform):
    """
    import this test to run after previous
    """
    path = "./data/rtk/infer"
    annotations_file = os.path.join(path, "annotations", "instances_default.json")
    if os.path.exists(annotations_file):
        os.remove(annotations_file)
    ds = DicomCocoDatasetRTK(data_dir=path, transform=transform)
    for batch in ds:
        break
    for k in ["image", "path"]:
        assert k in batch, f"no suck key {k}"


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


@patch("matplotlib.pyplot.show")
def test_hemor_contrast(mock_show, tmp_path_factory):
    target_dir = "./data/rtk/infer"
    if os.path.exists(target_dir):
        shutil.rmtree(target_dir)
    out_ = tmp_path_factory.mktemp("out")
    hemorrhage_contrast(input_path=rtk_segm, output_folder=out_)
    content = os.listdir(out_)
    assert len(content) > 0
    assert len(content) % 3 == 0
    assert np.any([x.endswith("npy") for x in content])
    assert np.any([x.endswith("png") for x in content])

    hemorrhage_contrast_metrics(out_)
    assert mock_show.call_count > 0


@pytest.mark.parametrize("task", ["segmentation", "detection"])
@patch("matplotlib.pyplot.show")
def test_segm_detection_pipeline_metrics(mock_show, tmp_path_factory, task):

    # just to imitate data loading
    target_dir = tmp_path_factory.mktemp("target_dir")
    path = {"source": rtk_segm, "target": target_dir}
    dm = DicomCocoDataModuleRTK(infer=path, transform=resize_transform)
    dm.setup_infer()
    ds = dm.predict_dataloader()
    ds.transform = resize_transform

    samples_number = len(ds)

    out_dir = tmp_path_factory.mktemp("out")
    for i in range(samples_number):
        random_numpy = np.random.randint(0, 1, [256, 256, 1])
        np.save(os.path.join(out_dir, f"{i}.npy"), random_numpy)

    process_metrics(input_path=target_dir, output_folder=out_dir)
    assert mock_show.call_count > 0
