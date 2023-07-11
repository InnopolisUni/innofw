import albumentations as A
import numpy as np
import torch

from innofw.core.augmentations import register_augmentations_adapter
from innofw.core.augmentations.base import BaseAugmentationAdapter

"""
references:
1. https://stackoverflow.com/questions/69151052/using-imagefolder-with-albumentations-in-pytorch
"""


@register_augmentations_adapter(name="albumentations_adapter")
class AlbumentationsAdapter(BaseAugmentationAdapter):
    """
    Class that adapts albumentations transformations

    Attributes
    ----------

    Methods
    -------
    is_suitable_input(optimizer):
       checks if the augmentation is suitable for the adapter
    forward(x, y=None):
        performs transformations
    """

    def __init__(self, transforms, *args, **kwargs):
        super().__init__(transforms)

    def forward(self, x, y=None, z=None):
        if y is not None:
            result = self.transforms(image=np.array(x), mask=y)
            if len(result["image"].shape) == 3 and result["image"].shape[2] == 3:
                img = np.moveaxis(result["image"], -1, 0)
            else:
                img = result["image"]

            return img, result["mask"]
        img = self.transforms(image=np.array(x))["image"]

        if len(img.shape) == 3 and img.shape[2] == 3:
            if isinstance(img, np.ndarray):
                img = np.moveaxis(img, -1, 0)  # HWC -> CHW
            elif isinstance(img, torch.Tensor):
                img = torch.moveaxis(img, -1, 0)  # HWC -> CHW for tensors
            else:
                raise NotImplementedError()
        return img

    def __repr__(self):
        return f"Albumentations: {self.transforms}"

    @staticmethod
    def is_suitable_input(transforms) -> bool:
        return isinstance(transforms, A.Compose) or isinstance(
            transforms, A.core.transforms_interface.BasicTransform
        )
