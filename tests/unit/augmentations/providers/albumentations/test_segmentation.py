import albumentations as A
import numpy as np
import pytest
from hydra.core.global_hydra import GlobalHydra

from innofw.core.augmentations import Augmentation
from innofw.core.augmentations import get_augs_adapter, register_augmentations_adapter
from innofw.utils.config import read_cfg, read_cfg_2_dict
from innofw.utils.framework import get_augmentations


def test_albumentations():
    albumenatations_transform = A.Compose(
        [
            A.RandomCrop(
                width=256,
                height=256,
            ),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.2),
        ]
    )
    aug = Augmentation(albumenatations_transform)
    assert aug is not None


# @pytest.mark.skip(reason="some problems with config")
def test_stages():
    GlobalHydra.instance().clear()
    cfg = read_cfg(
        
        overrides=[
            "augmentations_train=linear-roads-bin-seg",
            "experiments=semantic-segmentation/linear-roads-bin-seg/KA_160223_39ek249_linear_roads", #regression/KA_130722_9f7134db_linear_regression
        ])

    aug = get_augmentations(cfg.get("augmentations_train"))
    img = np.random.randint(0, 255, (3, 64, 64))

    aug_img = Augmentation(aug)(img)
    assert aug_img.min() >= 0 and aug_img.max() <= 1

    mask = np.random.randint(0, 2, (3, 64, 64))
    aug_img, aug_mask = Augmentation(aug)(img, mask)
    assert aug_img.min() >= 0 and aug_img.max() <= 1
    assert all(np.unique(aug_mask) == np.unique(mask))  # should not be any division

    #test wrong channel order handling
    img = np.random.randint(0,255, (64, 64, 3))
    mask = np.random.randint(0, 2, (64, 64, 3))
    aug_img = Augmentation(aug)(img, mask)
    assert aug_img[0].shape[0] == 3 and len(aug_img[0].shape) == 3

def test_torch_tensor_postproceessing():
    GlobalHydra.instance().clear()
    cfg = read_cfg(
    
    overrides=[
        "augmentations_train=linear-roads-bin-seg-test",
        "experiments=semantic-segmentation/linear-roads-bin-seg/KA_160223_39ek249_linear_roads", #regression/KA_130722_9f7134db_linear_regression
    ])

    aug = get_augmentations(cfg.get("augmentations_train"))
    img = np.random.randint(0, 255, (64, 64, 3))
    aug_img = Augmentation(aug)(img)
    assert aug_img.shape[0] == 3 and len(aug_img.shape) == 3
