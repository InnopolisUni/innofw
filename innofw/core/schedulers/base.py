"""
author: Kazybek Askarbek
date: 14.09.22
description:

"""
#
from abc import ABC
from abc import abstractmethod

import torch.nn as nn

#


class BaseSchedulerAdapter(ABC, nn.Module):
    """
    An abstract class to define interface and methods of scheduler adapter

    Methods
    -------
    is_suitable_input(scheduler)
        checks if the scheduler is suitable for the adapter
    step()
        updates a learning rate
    """

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
