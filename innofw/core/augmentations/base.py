"""
author: Kazybek Askarbek
date: 12.08.22
description:

"""
#
from abc import ABC, abstractmethod

#
import torch.nn as nn


class BaseAugmentationAdapter(ABC, nn.Module):
    def __init__(self, transforms):
        super().__init__()
        self.transforms = transforms

    @abstractmethod
    def forward(self, x):
        pass

    @staticmethod
    @abstractmethod
    def is_suitable_input(transform) -> bool:
        pass
