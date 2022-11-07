import albumentations as A
import numpy as np
from innofw.core.augmentations.base import BaseAugmentationAdapter
from innofw.core.augmentations import register_augmentations_adapter

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
    def forward(x, y=None):
        performs transformations
    """
    def __init__(self, transforms, *args, **kwargs):
        super().__init__(transforms)

    def forward(self, x, y=None):
        # todo: work with tensors right away????
        if y is not None:
            result = self.transforms(image=np.array(x), mask=y)
            return result["image"], result["mask"]

        return self.transforms(image=np.array(x))["image"]

    def __repr__(self):
        return f"Albumentations: {self.transforms}"

    @staticmethod
    def is_suitable_input(transforms) -> bool:
        return isinstance(transforms, A.Compose) or isinstance(
            transforms, A.core.transforms_interface.BasicTransform
        )
