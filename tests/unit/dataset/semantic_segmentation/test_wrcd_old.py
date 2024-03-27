import numpy as np
import pytest
import torch
from hydra.errors import InstantiationException
from hydra.utils import instantiate
from numpy import ndarray
from omegaconf import DictConfig

from innofw.constants import SegDataKeys
from innofw.constants import Frameworks
from innofw.utils.framework import get_augmentations, get_obj
from innofw.core.datasets.segmentation_hdf5_old_pipe import (
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
from tests.utils import get_test_folder_path
from tests.fixtures.config.datasets import arable_segmentation_cfg_w_target


@pytest.mark.parametrize(
    ["tif_folders", "is_train", 'augmentations'],
    [
        [str(get_test_folder_path() / "data/images/segmentation/forest/train/1"), 
                False, 
                resize_augmentation],
        [str(get_test_folder_path() / "data/images/segmentation/forest/test/1"), 
                True, 
                None],]
)
def test_read(tif_folders, is_train, augmentations):
    framework = Frameworks.torch
    task = "image-segmentation"
    # augmentations = get_augmentations(resize_augmentation)
    aug = get_obj(resize_augmentation, "augmentations", task, framework)
    ds = WeightedRandomCropDataset(tif_folders=[tif_folders],
                 is_train=is_train,
                 augmentations=aug)
    # ds: Dataset = instantiate(cfg, _convert_="partial")
    assert ds is not None
    # assert ds.len > 0

    # ds.setup()


    item = ds[0]
    assert item is not None

    # for item in ds:
    
    assert isinstance(item['image'], ndarray) or isinstance(
        item['image'], torch.Tensor
    )
    assert item['image'].shape[0] == 3
    assert isinstance(item['mask'], ndarray) or isinstance(
        item['mask'], torch.Tensor
    )

    
    assert item['mask'].max() <= 1
    # min value is 0
    assert item['mask'].min() == 0

    if is_train:
        assert item['image'].shape[1] == item['mask'].shape[1]
        assert item['image'].shape[2] == item['mask'].shape[2]
    else:
        assert item['image'].shape[1] == item['mask'].shape[1]
        assert item['image'].shape[2] == item['mask'].shape[2]




