"""
author: Kazybek Askarbek
date: 13.08.22
description:

"""
#
from abc import ABC, abstractmethod

#
import torch.nn as nn


class BaseLossAdapter(ABC, nn.Module):
    def __init__(self, loss):
        super().__init__()
        self.loss = loss

    @abstractmethod
    def forward(self, x):
        pass

    @staticmethod
    @abstractmethod
    def is_suitable_input(transform) -> bool:
        pass
