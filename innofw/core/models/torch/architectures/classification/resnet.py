#
from typing import Optional

import torch.nn as nn
import torchvision.models as models
import torch

#


class Resnet18(nn.Module):
    """
    Resnet model for classification task
    ...

    Attributes
    ----------
    num_classes : int
        number of classes to predict
    model : nn.Module
        Resnet model by torchvision

    Methods
    -------
    forward(x):
        returns result of the data forwarding

    """

    def __init__(
        self, num_classes: Optional[int] = 2, pretrained=True, *args, **kwargs
    ):
        super().__init__()

        self.num_classes = num_classes
        self.model = models.resnet18(pretrained)
        self.model.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        # x = x.unsqueeze(0)
        import torch
        
        x = torch.moveaxis(x, -1, 1)
        out = self.model(x)
        return nn.functional.softmax(out, dim=1)


class Resnet34(nn.Module):
    """
    Resnet model for classification task
    ...

    Attributes
    ----------
    num_classes : int
        number of classes to predict
    model : nn.Module
        Resnet model by torchvision

    Methods
    -------
    forward(x):
        returns result of the data forwarding

    """

    def __init__(
        self, num_classes: Optional[int] = 2, pretrained=True, *args, **kwargs
    ):
        super().__init__()

        self.num_classes = num_classes
        self.model = models.resnet34(pretrained)
        self.model.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        # x = x.unsqueeze(0)
        out = self.model(x)
        return nn.functional.softmax(out, dim=1)
