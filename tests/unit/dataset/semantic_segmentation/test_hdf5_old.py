import numpy as np
import pytest
import torch
from typing import List
from hydra.errors import InstantiationException
from hydra.utils import instantiate
from numpy import ndarray
from omegaconf import DictConfig

from innofw.constants import SegDataKeys
from innofw.constants import Frameworks
from innofw.utils.framework import get_augmentations, get_obj
from innofw.core.datasets.segmentation_hdf5_old_pipe import (
    Dataset,
    DatasetUnion,
    TiledDataset,
    _to_tensor,
    _augment_and_preproc,
    _get_preprocessing_fn,
    _get_class_weights
)
from tests.fixtures.config.augmentations import (
    resize_augmentation_albu as resize_augmentation,
)
from tests.utils import get_test_folder_path
from tests.fixtures.config.datasets import arable_segmentation_cfg_w_target

datasets = [str(get_test_folder_path() / "data/images/segmentation/arable/test/test.hdf5"),
            str(get_test_folder_path() / "data/images/segmentation/arable/train/train.hdf5")]

@pytest.mark.parametrize(
    ["path_to_hdf5", "with_mosaic", 'augmentations', "in_channels"],
    [
        [str(get_test_folder_path() / "data/images/segmentation/arable/train/train.hdf5"), 
                False, 
                resize_augmentation, 
                4],
        [str(get_test_folder_path() / "data/images/segmentation/arable/test/test.hdf5"), 
                True, 
                None, 
                4],
        [datasets, 
                True, 
                None, 
                4],
                ]
            
)
def test_read(path_to_hdf5, with_mosaic, augmentations, in_channels):
    framework = Frameworks.torch
    task = "image-segmentation"
    # augmentations = get_augmentations(resize_augmentation)
    aug = get_obj(resize_augmentation, "augmentations", task, framework)
    if isinstance(path_to_hdf5, List):
        ds = DatasetUnion(
                    [
                        Dataset(path_to_hdf5=f,
                                in_channels=in_channels,
                                augmentations=aug,
                        )
                        for f in path_to_hdf5
                    ])
    else:
        ds = Dataset(path_to_hdf5=path_to_hdf5,
                    in_channels=in_channels,
                    augmentations=aug)
    # ds: Dataset = instantiate(cfg, _convert_="partial")
    assert ds is not None
    # assert ds.len > 0

    # ds.setup()


    item = ds[0]
    assert item is not None

    # for item in ds:
    
    assert isinstance(item[SegDataKeys.image], ndarray) or isinstance(
        item[SegDataKeys.image], torch.Tensor
    )
    assert item[SegDataKeys.image].shape[0] == in_channels
    assert isinstance(item[SegDataKeys.label], ndarray) or isinstance(
        item[SegDataKeys.label], torch.Tensor
    )

    
    assert item[SegDataKeys.label].max() <= 1
    # min value is 0
    assert item[SegDataKeys.label].min() == 0

    if with_mosaic:
        assert item[SegDataKeys.image].shape[1] == item[SegDataKeys.label].shape[1]
        assert item[SegDataKeys.image].shape[2] == item[SegDataKeys.label].shape[2]
    else:
        assert item[SegDataKeys.image].shape[1] == item[SegDataKeys.label].shape[1]
        assert item[SegDataKeys.image].shape[2] == item[SegDataKeys.label].shape[2]


# @pytest.mark.parametrize(
#     ["ds_cfg", "aug_cfg"],
#     [[arable_segmentation_cfg_w_target.copy(), resize_augmentation.copy()]],
# )
# def test_ds_w_transform(ds_cfg, aug_cfg):
#     ds_cfg["transform"] = aug_cfg
#     test_read(ds_cfg, False, None, in_channels=4)  


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


