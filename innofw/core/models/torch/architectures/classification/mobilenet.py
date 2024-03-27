#
from typing import Optional

import torch.nn as nn
import torchvision.models as models

#


class MobileNetV2(nn.Module):
    """
    MobileNetV2 model for classification task
    ...

    Attributes
    ----------
    num_classes : int
        number of classes to predict
    model : nn.Module
        MobileNetV2 model by torchvision

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
        self.model = models.mobilenet_v2(pretrained)
        self.model.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        out = self.model(x)
        return nn.functional.softmax(out, dim=1)
