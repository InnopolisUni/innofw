import torchvision.transforms as T

from innofw.core.augmentations import Augmentation
from innofw.core.augmentations.torchvision_adapter import TorchvisionAdapter


def test_torchvision():
    torchvision_transform = T.Compose(
        [
            T.RandomCrop(size=256),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomAutocontrast(p=0.2),
        ]
    )

    aug = Augmentation(torchvision_transform)
    assert aug is not None
    assert repr(aug).startswith("Torchvision: Compose(")