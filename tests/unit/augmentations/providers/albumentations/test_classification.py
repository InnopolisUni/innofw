import albumentations as A
import numpy as np

from innofw.core.augmentations import Augmentation
from innofw.utils.config import read_cfg
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


def test_stages():
    cfg = read_cfg(
        overrides=[
            "augmentations_train=linear-roads-bin-seg",
            "experiments=regression/KA_130722_9f7134db_linear_regression",
        ]
    )
    aug = get_augmentations(cfg["augmentations_train"]["augmentations"])
    img = np.random.randint(0, 255, (3, 64, 64))

    aug_img = Augmentation(aug)(img)
    assert aug_img.min() >= 0 and aug_img.max() <= 1

    mask = np.random.randint(0, 2, (3, 64, 64))
    aug_img, aug_mask = Augmentation(aug)(img, mask)
    assert aug_img.min() >= 0 and aug_img.max() <= 1
    assert all(np.unique(aug_mask) == np.unique(mask))  # should not be any division
