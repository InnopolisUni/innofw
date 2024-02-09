import numpy as np
import pytest
import torch
from hydra.errors import InstantiationException
from hydra.utils import instantiate
from numpy import ndarray

from innofw.constants import SegDataKeys
from innofw.constants import Frameworks
from innofw.utils.framework import get_augmentations, get_obj
from innofw.core.datasets.segmentation_hdf5_old_pipe import (
    Dataset,
    DatasetUnion,
    WeightedRandomCropDataset,
    TiledDataset,
    _to_tensor,
    _augment_and_preproc,
    _get_preprocessing_fn,
    _get_class_weights
)
from tests.fixtures.config.augmentations import (
    resize_augmentation_albu as resize_augmentation,
)
from tests.fixtures.config.datasets import arable_segmentation_cfg_w_target


@pytest.mark.parametrize(
    ["cfg", "with_mosaic", 'preprocessing', "in_channels"],
    [
        [arable_segmentation_cfg_w_target.copy(), False, None, 4],
        [arable_segmentation_cfg_w_target.copy(), True, None, 4],
        # [arable_segmentation_cfg_w_target.copy(), None, None, 4],
    ],
)
def test_read(cfg, with_mosaic, preprocessing, in_channels):


    ds: Dataset = instantiate(cfg, _convert_="partial")
    assert ds is not None
    # assert ds.len > 0

    ds.setup()


    assert iter(ds.train_dataloader()).next()
    # assert item is not None

    # for item in ds:
    item = iter(ds.train_dataloader()).next()
    assert isinstance(item[SegDataKeys.image], ndarray) or isinstance(
        item[SegDataKeys.image], torch.Tensor
    )
    assert item[SegDataKeys.image].shape[1] == in_channels
    assert isinstance(item[SegDataKeys.label], ndarray) or isinstance(
        item[SegDataKeys.label], torch.Tensor
    )

    
    assert item[SegDataKeys.label].max() <= 1
    # min value is 0
    assert item[SegDataKeys.label].min() == 0

    if with_mosaic:
        assert item[SegDataKeys.image].shape[2] == item[SegDataKeys.label].shape[2]
        assert item[SegDataKeys.image].shape[3] == item[SegDataKeys.label].shape[3]
    else:
        assert item[SegDataKeys.image].shape[2] == item[SegDataKeys.label].shape[2]
        assert item[SegDataKeys.image].shape[3] == item[SegDataKeys.label].shape[3]


@pytest.mark.parametrize(
    ["ds_cfg", "aug_cfg"],
    [[arable_segmentation_cfg_w_target.copy(), resize_augmentation.copy()]],
)
def test_ds_w_transform(ds_cfg, aug_cfg):
    ds_cfg["transform"] = aug_cfg
    test_read(ds_cfg, False, None, in_channels=4)  


# @pytest.fixture
# def input_data():
#     x = np.random.rand(3,512,512)
#     return x

# @pytest.fixture
# def input_mask():
#     x = np.random.rand(1,256,256)
#     return x

def test__to_tensor():
    x = np.random.rand(256,256,3)

    out_data = _to_tensor(x)
    assert out_data.shape[0] == 3

# @pytest.mark.parametrize(
#     ["aug"],
#     [[resize_augmentation.copy()]],
# )
def test__augment_and_preproc():
    cfg = {}
    input_data = np.random.rand(256,256,3)
    input_mask = np.random.rand(256,256,1)
    framework = Frameworks.torch
    task = "image-segmentation"
    # augmentations = get_augmentations(resize_augmentation)
    augmentations = get_obj(resize_augmentation, "augmentations", task, framework)
    
    x, y = _augment_and_preproc(input_data, input_mask, augmentations, None)
    assert isinstance(x, ndarray) or isinstance(x, torch.Tensor)
    assert isinstance(y, ndarray) or isinstance(y, torch.Tensor)

    assert x.shape[1] == 244
    assert x.shape[1] == y.shape[1]

def test__get_class_weights():
    mask = np.random.randint(0, 2, (256,256,1))
    weights = _get_class_weights(mask)
    assert weights is not None   


