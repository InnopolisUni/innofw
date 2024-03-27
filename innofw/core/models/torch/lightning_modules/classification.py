from typing import Any

import torch

from innofw.core.models.torch.lightning_modules.base import BaseLightningModule
from torchmetrics.classification import (
    MulticlassAccuracy,
    MulticlassRecall,
    MulticlassPrecision,
    MulticlassF1Score,
)
from torchmetrics import MetricCollection


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
    threshold: float
        threshold to use while training

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
        threshold=0.5,
        *args: Any,
        **kwargs: Any,
    ):
        super().__init__(*args, **kwargs)
        self.model = model
        self.losses = losses
        self.optimizer_cfg = optimizer_cfg
        self.scheduler_cfg = scheduler_cfg
        self.threshold = threshold
        metrics = MetricCollection(
            [
                MulticlassAccuracy(num_classes=model.num_classes, threshold=threshold), 
                MulticlassPrecision(num_classes=model.num_classes, threshold=threshold), 
                MulticlassRecall(num_classes=model.num_classes, threshold=threshold), 
                MulticlassF1Score(num_classes=model.num_classes, threshold=threshold),
            ]
        )
        self.train_metrics = metrics.clone(prefix="train_")
        self.val_metrics = metrics.clone(prefix="val_")
        self.test_metrics = metrics.clone(prefix="test_")

    def forward(self, x, *args, **kwargs) -> Any:
        return self.model(x)

    def training_step(self, batch, batch_idx):
        """
        Arguments
            batch - tensor
        """
        return self.stage_step("train", batch)

    def validation_step(self, batch, batch_idx):
        return self.stage_step("val", batch)
    
    def test_step(self, batch, batch_idx):
        return self.stage_step("test", batch)

    def predict_step(self, batch, batch_idx, **kwargs):
        input_tensor, _ = batch
        outputs = self.forward(input_tensor.float())
        outputs = torch.argmax(outputs, 1)
        return outputs
    
    def stage_step(self, stage, batch, do_logging=False, *args, **kwargs):
        output = dict()
        # todo: check that model is in mode no autograd
        image, label = batch

        predictions = self.forward(image.float())

        if stage in ["train", "val"]:
            loss = self.calc_losses(predictions, label)
            self.log(f"{stage}_loss", loss, on_step=False, on_epoch=True)
            output["loss"] = loss
        if stage != "predict":
            metrics = self.compute_metrics(
                stage, predictions, label
            )
            self.log_metrics(stage, metrics)
        return output

    def calc_losses(self, label, pred) -> torch.FloatTensor:
        """Function to compute losses"""
        total_loss = 0.0
        if isinstance(self.losses, list):
            for loss_name, weight, loss in self.losses:
                # for loss_name in loss_dict:
                ls_mask = loss(label, pred)
                total_loss += weight * ls_mask
        else:
            total_loss = self.losses(label, pred)

        return total_loss

    def log_metrics(self, stage, metrics_res):
        for key, value in metrics_res.items():
            self.log(key, value, sync_dist=True, on_step=False, on_epoch=True)

    def compute_metrics(self, stage, predictions, labels):
        if stage == "train":
            return self.train_metrics(predictions, labels)
        elif stage == "val":
            return self.val_metrics(predictions, labels)
        elif stage == "test":
            return self.test_metrics(predictions, labels)