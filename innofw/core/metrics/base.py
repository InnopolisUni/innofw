"""
author: Kazybek Askarbek
date: 12.08.22
description:

"""
#
from abc import ABC, abstractmethod

#
import torch.nn as nn


class BaseMetricAdapter(ABC, nn.Module):
    """
    Abstract class for working with different metric libraries
    ...

    Attributes
    ----------
    metric : metric
        metric function

    Methods
    -------
    forward(x):
        abstract function to perform metric

    """
    def __init__(self, metric):
        super().__init__()
        self.metric = metric

    @abstractmethod
    def forward(self, x):
        pass

    @staticmethod
    @abstractmethod
    def is_suitable_input(metric) -> bool:
        pass

