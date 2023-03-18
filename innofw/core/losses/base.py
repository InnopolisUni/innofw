"""
author: Kazybek Askarbek
date: 13.08.22
description:

"""
#
from abc import ABC
from abc import abstractmethod

import torch.nn as nn

#


class BaseLossAdapter(ABC, nn.Module):
    """
    An abstract class to define interface and methods of loss adapter

    Methods
    -------
    is_suitable_input(loss)
        checks if the loss is suitable for the adapter
    forward(*args, **kwargs)
        computes the loss and outputs in the desired format
    """

    def __init__(self, loss):
        super().__init__()
        self.loss = loss

    @abstractmethod
    def forward(self, x):
        pass

    @staticmethod
    @abstractmethod
    def is_suitable_input(loss) -> bool:
        pass
