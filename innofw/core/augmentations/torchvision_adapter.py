#
import numpy as np
import torch.nn as nn
import torchvision

from innofw.core.augmentations import register_augmentations_adapter
from innofw.core.augmentations.base import BaseAugmentationAdapter

#


@register_augmentations_adapter(name="torchvision_adapter")
class TorchvisionAdapter(BaseAugmentationAdapter):
    """
    Class that adapts Torchvision transformations

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

    @staticmethod
    def is_suitable_input(transforms) -> bool:
        return isinstance(transforms, torchvision.transforms.Compose) or isinstance(
            transforms, nn.Module
        )

    def forward(self, x):
        return self.transforms(np.array(x))

    def __repr__(self):
        return f"Torchvision: {self.transforms}"
