import pytorch_lightning as pl
from typing import Any
import torch


class AnomalyDetectionTimeSeriesLightningModule(pl.LightningModule):
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
        self, model, losses, optimizer_cfg, scheduler_cfg, *args: Any, **kwargs: Any
    ):
        super().__init__(*args, **kwargs)
        self.model = model
        self.losses = losses
        self.optimizer_cfg = optimizer_cfg
        self.scheduler_cfg = scheduler_cfg

    def forward(self, seq_true, *args, **kwargs) -> Any:
        return self.model(seq_true)

    def configure_optimizers(self):
        """Function to set up optimizers and schedulers"""
        # get all trainable model parameters
        params = [x for x in self.model.parameters() if x.requires_grad]
        # instantiate models from configurations
        optim = self.optimizer_cfg(params=params)
        # optim = hydra.utils.instantiate(self.optimizer_cfg, params=params)

        # instantiate scheduler from configurations
        try:
            scheduler = self.scheduler_cfg(optim)
            # scheduler = hydra.utils.instantiate(self.scheduler_cfg, optim)
            # return optimizers and schedulers
            return [optim], [scheduler]
        except:
            return [optim]

    def training_step(self, batch, batch_idx):
        """
        Arguments
            batch - tensor
        """

        x, y = batch
        pred = self.forward(x[0])
        loss = self.calc_losses(x[0], pred)

        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        x, y = batch
        seq_pred = self.forward(x[0])
        loss = self.calc_losses(x[0], seq_pred)

        return {"val_loss": loss}

    def predict_step(self, batch, batch_idx, **kwargs):
        x, y = batch
        seq_pred = self.forward(x[0])
        loss = self.calc_losses(x[0], seq_pred)
        return loss.item()

    def calc_losses(self, seq_true, seq_pred) -> torch.FloatTensor:
        """Function to compute losses"""
        total_loss = 0.0  # todo: redefine it as torch.FloatTensor(0.0)
        for loss_name, weight, loss in self.losses:
            # for loss_name in loss_dict:
            ls_mask = loss(seq_true, seq_pred)
            total_loss += weight * ls_mask

        return total_loss
