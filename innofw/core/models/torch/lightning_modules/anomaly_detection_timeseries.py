from typing import Any

import torch

from innofw.core.models.torch.lightning_modules.base import BaseLightningModule


class AnomalyDetectionTimeSeriesLightningModule(BaseLightningModule):
    """
    PyTorchLightning module for Anomaly Detection in Time Series
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

    def forward(self, seq_true, *args, **kwargs) -> Any:
        return self.model(seq_true)

    def training_step(self, batch, batch_idx):
        """
        Arguments
            batch - tensor
        """

        x, y = batch
        seq_pred = self.forward(x[0])
        loss = self.calc_losses(x[0], seq_pred)
        self.log_metrics("train", x[0], seq_pred)
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        x, y = batch
        seq_pred = self.forward(x[0])
        loss = self.calc_losses(x[0], seq_pred)
        self.log_metrics("val", x[0], seq_pred)
        self.log("val_loss", loss, on_step=False, on_epoch=True)
        return {"val_loss": loss}

    def predict_step(self, batch, batch_idx, **kwargs):
        x, y = batch
        seq_pred = self.forward(x[0])
        loss = self.calc_losses(x[0], seq_pred)
        return loss.item()

    def calc_losses(self, seq_true, seq_pred) -> torch.FloatTensor:
        """Function to compute losses"""
        total_loss = 0.0
        for loss_name, weight, loss in self.losses:
            # for loss_name in loss_dict:
            ls_mask = loss(seq_true, seq_pred)
            total_loss += weight * ls_mask

        return total_loss
