import albumentations as A

from innofw.core.augmentations import Augmentation


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
