import logging
import pytorch_lightning as pl
from typing import Any
import torch
import torch.nn.functional as F
from omegaconf import DictConfig
import hydra
from innofw.core.models.torch.lightning_modules.base import BaseLightningModule


class OneShotLearningLightningModule(BaseLightningModule):
    """
    PyTorchLightning module for One Shot Learning
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
        self.automatic_optimization = False

    def forward(self, img0, img1, *args, **kwargs) -> Any:
        img0 = img0.view(-1, 1, 100, 100)
        img1 = img1.view(-1, 1, 100, 100)
        return self.model(img0, img1)

    def training_step(self, batch, batch_idx):
        """
        Arguments
            batch - tensor
        """

        img0, img1, label = batch
        output1, output2 = self.forward(img0, img1)
        loss = self.calc_losses(output1, output2, label)

        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        img0, img1, label = batch
        output1, output2 = self.forward(img0, img1)
        loss = self.calc_losses(output1, output2, label)
        self.log("val_loss", loss, prog_bar=True)
        return {"val_loss": loss}

    def predict_step(self, batch, batch_idx, **kwargs):
        img0, img1 = batch
        output1, output2 = self.forward(img0, img1)
        euclidean_distance = F.pairwise_distance(output1, output2)
        result = euclidean_distance.tolist()
        if isinstance(result, list):
            for i in result:
                logging.info(f"Image difference: {i}")
        else:
            logging.info(f"Image difference: {result}")
        return result

    def calc_losses(self, output1, output2, label) -> torch.FloatTensor:
        """Function to compute losses"""
        total_loss = 0
        for loss_name, weight, loss in self.losses:
            # for loss_name in loss_dict:
            ls_mask = loss(output1, output2, label)
            total_loss += weight * ls_mask

        return total_loss
