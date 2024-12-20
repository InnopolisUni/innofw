from unittest.mock import patch, MagicMock
from pathlib import Path
import os
import shutil

import pytest
import numpy as np
import torch


from innofw.constants import Stages
from innofw.core.datamodules.lightning_datamodules.coco_rtk import (
    DEFAULT_TRANSFORM,
    DicomCocoComplexingDataModule,
    DicomCocoDataModuleRTK,
    CustomNormalize,
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


@pytest.fixture()
def rtk_data(tmp_path_factory):
    target_dir = tmp_path_factory.mktemp("infer")
    if os.path.exists(target_dir):
        shutil.rmtree(target_dir)
    path = {"source": rtk_segm, "target": target_dir}
    dm = DicomCocoDataModuleRTK(infer=path, transform=resize_transform)
    dm.setup_infer()
    return target_dir


@pytest.fixture
def rtk_downloader(rtk_data):
    with patch(
        "innofw.core.datamodules.lightning_datamodules.coco_rtk.DicomCocoDataModuleRTK._get_data"
    ) as mock:
        mock.return_value = [rtk_data, rtk_data]
        yield mock


@pytest.fixture()
def complex_data(tmp_path_factory):
    target_dir = tmp_path_factory.mktemp("infer")
    if os.path.exists(target_dir):
        shutil.rmtree(target_dir)
    path = {"source": rtk_complex, "target": target_dir}
    dm = DicomCocoDataModuleRTK(infer=path, transform=resize_transform)
    dm.setup_infer()
    return target_dir


@pytest.fixture
def complex_downloader(complex_data):
    with patch(
        "innofw.core.datamodules.lightning_datamodules.coco_rtk.DicomCocoComplexingDataModule._get_data"
    ) as mock:
        mock.return_value = [complex_data, complex_data]
        yield mock


# transforms
def test_default_transform():
    image = np.random.randint(0, 255, size=(32, 32, 3), dtype=np.uint8)
    mask = np.random.randint(0, 2, size=(32, 32), dtype=np.uint8)

    transformed = DEFAULT_TRANSFORM(image=image, mask=mask)
    assert transformed["image"].shape == (3, 256, 256)
    assert transformed["mask"].shape[-2:] == (256, 256)
    assert isinstance(transformed["image"], torch.Tensor)
    assert isinstance(transformed["mask"], torch.Tensor)
    assert np.allclose(transformed["image"].min(), 0, atol=1e-6)
    assert np.allclose(transformed["image"].max(), 1, atol=1e-6)


def test_normalize():
    image = np.array([[1, 2, 3], [4, 5, 6]])
    normalize = CustomNormalize()
    normalized_image = normalize(image)

    assert np.allclose(normalized_image.min(), 0.0)
    assert np.allclose(normalized_image.max(), 1.0)


# datamodules
@pytest.mark.parametrize("transform", transforms)
def test_DicomCocoComplexingDataModule_predict_dataloader(
    transform, complex_data, complex_downloader
):
    path = {"source": rtk_complex, "target": complex_data}
    dm = DicomCocoComplexingDataModule(infer=path, transform=transform)
    dm.setup_infer()
    ds = dm.predict_dataloader()
    for batch in ds:
        break
    for k in ["image", "mask", "path"]:
        assert k in batch, f"no suck key {k}"
    assert complex_downloader.call_count == 3


@pytest.mark.parametrize("transform", transforms)
def test_DicomCocoDataModuleRTK_predict_dataloader(transform, rtk_data, rtk_downloader):
    path = {"source": rtk_segm, "target": rtk_data}
    dm = DicomCocoDataModuleRTK(infer=path, transform=transform)
    dm.setup_infer()
    ds = dm.predict_dataloader()
    for batch in ds:
        break
    for k in ["image", "path"]:
        assert k in batch, f"no suck key {k}"
    assert rtk_downloader.call_count == 3


@patch("numpy.save")
def test_ComplexDataModule_save_preds(
    mock_save, tmp_path_factory, complex_data, complex_downloader
):
    path = {"source": rtk_complex, "target": complex_data}
    dm = DicomCocoComplexingDataModule(infer=path)
    dm.setup_infer()

    batch_size = 2
    height, width = 32, 32
    batches = 2
    preds = torch.tensor(np.random.rand(batches, batch_size, 1, height, width))
    dst_path = tmp_path_factory.mktemp("output")
    dm.save_preds(preds, Stages.predict, dst_path)
    assert mock_save.call_count == batch_size * batches
    for i in range(batch_size * batches):
        args, kwargs = mock_save.call_args_list[i]
        assert isinstance(args[1], np.ndarray)
        assert args[1].shape == (height, width, 1)


@patch("numpy.save")
def test_RTKDataModule_save_preds(mock_save, rtk_data, rtk_downloader, tmp_path):
    path = {"source": rtk_segm, "target": rtk_data}
    dm = DicomCocoDataModuleRTK(infer=path, transform=resize_transform)
    dm.setup_infer()
    size = 32
    preds = torch.tensor(np.random.rand(2, 2, 1, size, size))
    dst_path = tmp_path
    dm.save_preds(preds, Stages.predict, dst_path)
    assert mock_save.call_count == preds.shape[0] * preds.shape[1]
    args, kwargs = mock_save.call_args_list[0]
    assert args[1].shape == (size, size, 1)


@pytest.mark.parametrize("transform", transforms)
def test_DicomCocoDataset_rtk(transform, rtk_data):
    ds = DicomCocoDatasetRTK(data_dir=rtk_data, transform=transform)
    for batch in ds:
        break
    for k in ["image", "mask", "path"]:
        assert k in batch, f"no suck key {k}"


@pytest.mark.parametrize("transform", transforms)
def test_DicomCocoDataset_rtk_no_annotation(transform, tmp_path_factory):
    """
    since we modify data we download it again
    """
    target_dir = str(tmp_path_factory.mktemp("infer"))
    path = {"source": str(rtk_complex), "target": target_dir}
    dm = DicomCocoDataModuleRTK(infer=path, transform=transform)
    dm.setup_infer()

    annotations_file = os.path.join(target_dir, "annotations", "instances_default.json")
    if os.path.exists(annotations_file):
        os.remove(annotations_file)
    ds = DicomCocoDatasetRTK(data_dir=target_dir, transform=transform)
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
def test_hemor_contrast(mock_show, rtk_data, tmp_path_factory):
    out_ = str(tmp_path_factory.mktemp("out"))
    hemorrhage_contrast(input_path=str(rtk_data), output_folder=out_)
    content = os.listdir(out_)
    assert len(content) > 0
    assert len(content) % 3 == 0
    assert np.any([x.endswith("npy") for x in content])
    assert np.any([x.endswith("png") for x in content])

    hemorrhage_contrast_metrics(out_)
    assert mock_show.call_count > 0


@pytest.mark.parametrize("task", ["segmentation", "detection"])
@patch("matplotlib.pyplot.show")
def test_segm_detection_pipeline_metrics(
    mock_show, tmp_path_factory, task, rtk_data, rtk_downloader
):
    path = {"source": rtk_segm, "target": rtk_data}
    dm = DicomCocoDataModuleRTK(infer=path, transform=resize_transform)
    dm.setup_infer()
    ds = dm.predict_dataloader()
    ds.transform = resize_transform

    samples_number = len(ds)
    out_dir = tmp_path_factory.mktemp("out")
    for i in range(samples_number):
        random_numpy = np.random.randint(0, 1, [256, 256, 1])
        np.save(os.path.join(out_dir, f"{i}.npy"), random_numpy)
    process_metrics(input_path=rtk_data, output_folder=out_dir)
    assert mock_show.call_count > 0
    assert rtk_downloader.call_count > 0
