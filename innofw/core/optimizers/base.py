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


class BaseOptimizerAdapter(ABC, nn.Module):
    """
    An abstract class to define interface and methods of optimizer adapter

    Methods
    -------
    is_suitable_input(optimizer)
        checks if the optimizer is suitable for the adapter
    step()
        updates a model parameters
    """

    def __init__(self, optimizer):
        super().__init__()
        self.optimizer = optimizer

    @abstractmethod
    def step(self):
        pass

    @staticmethod
    @abstractmethod
    def is_suitable_input(optimizer) -> bool:
        pass
