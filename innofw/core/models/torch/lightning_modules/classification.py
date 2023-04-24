from typing import Any

import torch

from innofw.core.models.torch.lightning_modules.base import BaseLightningModule


class ClassificationLightningModule(BaseLightningModule):
    """
    PyTorchLightning module for Classification task
    ...

    Attributes
    ----------
    model : nn.Module
        model to train
    losses : losses
        loss to use while training
    optimizer_cfg : cfg
        optimizer configurations
    scheduler_cfg : cfg
        scheduler configuration

    Methods
    -------
    forward(x):
        returns result of prediction
    calc_losses(output1, output2, label)
        calculates losses and returns total loss

    """

    def __init__(
        self,
        model,
        losses,
        optimizer_cfg,
        scheduler_cfg,
        *args: Any,
        **kwargs: Any,
    ):
        super().__init__(*args, **kwargs)
        self.model = model
        self.losses = losses
        self.optimizer_cfg = optimizer_cfg
        self.scheduler_cfg = scheduler_cfg

    def forward(self, x, *args, **kwargs) -> Any:
        return self.model(x)

    def training_step(self, batch, batch_idx):
        """
        Arguments
            batch - tensor
        """
        image, target = batch
        outputs = self.forward(image.float())
        loss = self.losses(outputs, target.long())
        self.log_metrics("train", outputs, target.long())
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        image, target = batch
        outputs = self.forward(image.float())
        loss = self.losses(outputs, target.long())
        self.log_metrics("val", outputs, target.long())
        self.log("val_loss", loss, prog_bar=True)
        return {"val_loss": loss}

    def predict_step(self, batch, batch_idx, **kwargs):
        outputs = self.forward(batch.float())
        outputs = torch.argmax(outputs, 1)
        return outputs
