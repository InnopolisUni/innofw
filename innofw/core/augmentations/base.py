"""
author: Kazybek Askarbek
date: 12.08.22
description:

"""
#
from abc import ABC
from abc import abstractmethod

import torch.nn as nn

#


class BaseAugmentationAdapter(ABC, nn.Module):
    """
    An abstract class to define interface and methods of optimizer adapter
    Attributes
    ----------
    transforms : transforms
        list of transformations to perform

    Methods
    -------
    is_suitable_input(optimizer):
       checks if the augmentation is suitable for the adapter
    """

    def __init__(self, transforms):
        super().__init__()
        self.transforms = transforms

    @abstractmethod
    def forward(self, x): # pragma: no cover
        pass

    @staticmethod
    @abstractmethod
    def is_suitable_input(transform) -> bool: # pragma: no cover
        pass
