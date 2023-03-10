from typing import Any

import torch.nn as nn
from pytorch_lightning import LightningModule
from torch.optim import SGD


class DummyTorchModel(nn.Module):
    def __init__(self, in_channels=10, out_channels=10):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Linear(in_channels, 100), nn.Linear(100, out_channels)
        )

    def forward(self, x):
        return self.layers(x)


class DummyLightningModel(LightningModule):
    def __init__(self, model, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.model = model

    def configure_optimizers(self):
        optimizer = SGD(self.model.parameters(), lr=1e-3)
        return optimizer

    def forward(self, x):
        return self.model(x)
