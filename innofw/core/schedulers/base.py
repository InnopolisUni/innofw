"""
author: Kazybek Askarbek
date: 14.09.22
description:

"""
#
from abc import ABC, abstractmethod

#
import torch.nn as nn


class BaseSchedulerAdapter(ABC, nn.Module):
    def __init__(self, scheduler, optimizer, *args, **kwargs):
        super().__init__()
        self.scheduler = scheduler(optimizer.optim.optimizer, *args, **kwargs)

    @abstractmethod
    def step(self):
        pass

    @staticmethod
    @abstractmethod
    def is_suitable_input(optimizer) -> bool:
        pass
