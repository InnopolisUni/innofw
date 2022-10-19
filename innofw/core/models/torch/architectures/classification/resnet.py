#
from typing import Optional

#
import torch
import torch.nn as nn
import torchvision.models as models


class Resnet18(nn.Module):
    def __init__(
        self, num_classes: Optional[int] = 2, pretrained=True, *args, **kwargs
    ):
        super().__init__()

        self.num_classes = num_classes
        self.model = models.resnet18(pretrained)
        self.model.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        # x = x.unsqueeze(0)
        out = self.model(x)
        return nn.functional.softmax(out, dim=1)
