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
    TiledDataset,

)
from tests.fixtures.config.augmentations import (
    resize_augmentation_albu as resize_augmentation,
)
from tests.utils import get_test_folder_path
from tests.fixtures.config.datasets import arable_segmentation_cfg_w_target


@pytest.mark.parametrize(
    ["tif_folders", 'crop_size', "crop_step", 'augmentations'],
    [
        [str(get_test_folder_path() / "data/images/segmentation/forest/train/1"), 
                (128, 128),
                (128, 128), 
                resize_augmentation],
        [str(get_test_folder_path() / "data/images/segmentation/forest/test/1"), 
                (128, 128),
                (64, 64), 
                None],]
)
def test_read(tif_folders, crop_size, crop_step, augmentations):
    framework = Frameworks.torch
    task = "image-segmentation"
    # augmentations = get_augmentations(resize_augmentation)
    aug = get_obj(resize_augmentation, "augmentations", task, framework)
    ds = TiledDataset(tif_folders=[tif_folders],
                      crop_size=crop_size,
                      crop_step=crop_step,
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

    assert item['image'].shape[1] == item['mask'].shape[1]
    assert item['image'].shape[2] == item['mask'].shape[2]




