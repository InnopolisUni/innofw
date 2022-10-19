"""
author: Kazybek Askarbek
date: 12.08.22
description:

"""
#
from abc import ABC, abstractmethod

#
import torch.nn as nn


class BaseOptimizerAdapter(ABC, nn.Module):
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
