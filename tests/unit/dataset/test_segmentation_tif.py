import numpy as np
import pytest
import torch
from hydra.errors import InstantiationException
from hydra.utils import instantiate
from numpy import ndarray

from innofw.constants import SegDataKeys
from innofw.core.datasets.segmentation import (
    SegmentationDataset
)
from innofw.core.datasets.segmentation_tif import get_metadata
from tests.fixtures.config.augmentations import (
    bare_aug_torchvision as resize_augmentation,
)
from tests.fixtures.config.datasets_2 import roads_tiff_dataset_w_masks


@pytest.mark.parametrize(
    ["cfg", "w_mask", "size", "n_channels"],
    [
        [roads_tiff_dataset_w_masks.copy(), True, 2048, 3],
        [roads_tiff_dataset_w_masks.copy(), False, 2048, 3],
    ],
)
def test_read(cfg, w_mask, size, n_channels):
    if not w_mask:
        cfg["masks"] = None

    ds: SegmentationDataset = instantiate(cfg, _convert_="partial")
    assert ds is not None
    assert len(ds) > 0

    item = ds[0]
    assert item is not None

    for item in ds:
        assert isinstance(item[SegDataKeys.image], ndarray) or isinstance(
            item[SegDataKeys.image], torch.Tensor
        )
        assert item[SegDataKeys.image].shape == (n_channels, size, size)

        if w_mask:
            assert isinstance(item[SegDataKeys.image], ndarray) or isinstance(
                item[SegDataKeys.image], torch.Tensor
            )
            assert item[SegDataKeys.label].shape == (1, size, size)

            # as it is a binary segmentation data
            # it should have at most two distinct values
            assert len(np.unique(item[SegDataKeys.label])) <= 2
            # max value is 1
            assert item[SegDataKeys.label].max() <= 1
            # min value is 0
            assert item[SegDataKeys.label].min() == 0
        else:
            with pytest.raises(KeyError):
                assert item[SegDataKeys.label]

def test_get_metadata():
    meta = get_metadata('tests/data/images/segmentation/forest/train/1/B02.tif')
    assert meta is not None

@pytest.mark.parametrize(
    ["ds_cfg", "aug_cfg"],
    [[roads_tiff_dataset_w_masks.copy(), resize_augmentation.copy()]],
)
def test_ds_w_transform(ds_cfg, aug_cfg):
    ds_cfg["transform"] = aug_cfg
    test_read(ds_cfg, False, 244, 3)  # resize should decrease the size


@pytest.mark.parametrize(["ds_cfg"], [[roads_tiff_dataset_w_masks.copy()]])
def test_channels(ds_cfg):
    # data should contain 4 channels
    # but 3 needed
    ds_cfg["channels"] = 2
    test_read(ds_cfg, False, 2048, 2)


@pytest.mark.parametrize(["cfg"], [[roads_tiff_dataset_w_masks.copy()]])
def test_ds_w_caching(cfg):
    cfg["w_caching"] = True
    # with masks
    test_read(cfg, True, 2048, 3)
    # without masks
    test_read(cfg, False, 2048, 3)


@pytest.mark.parametrize(["cfg"], [[roads_tiff_dataset_w_masks.copy()]])
def test_wrong_img_mask_number(cfg):
    cfg["images"] = cfg["images"][:1]

    with pytest.raises(InstantiationException):
        ds: SegmentationDataset = instantiate(cfg, _convert_="partial")

