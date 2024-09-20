from typing import Any

import torch
from torchmetrics import MetricCollection
from torchmetrics.classification import BinaryJaccardIndex, BinaryF1Score, BinaryPrecision, \
    BinaryRecall
from torchmetrics.regression import MeanAbsoluteError, MeanSquaredError
from lovely_numpy import lo

from innofw.core.models.torch.lightning_modules.base import BaseLightningModule


class AnomalyDetectionImagesLightningModule(BaseLightningModule):
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

        self.loss_fn = torch.nn.MSELoss()

        metrics = MetricCollection(
            [MeanSquaredError(), MeanAbsoluteError()]
        )

        self.train_metrics = metrics.clone(prefix='train')
        segmentation_metrics = MetricCollection(
            [
                BinaryF1Score(),
                BinaryPrecision(),
                BinaryRecall(),
                BinaryJaccardIndex(),
            ]
        )
        self.val_metrics = segmentation_metrics.clone(prefix='val')
        self.test_metrics = segmentation_metrics.clone(prefix='val')

    def forward(self, x, *args, **kwargs) -> Any:
        return self.model(x.float())

    def training_step(self, x, batch_idx):
        x_rec = self.forward(x)
        loss = self.loss_fn(x, x_rec)
        metrics = self.compute_metrics('train', x_rec, x)
        self.log_metrics('train', metrics)
        self.log("train_loss", loss, on_step=False, on_epoch=True)
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y = y.bool()
        x_rec = self.forward(x)
        loss = self.loss_fn(x, x_rec)
        mask = self.compute_anomaly_mask(x)
        metrics = self.compute_metrics('val', mask, y)
        self.log_metrics('val', metrics)
        print(mask.float().mean(), y.float().mean())
        self.log("val_loss", loss, on_step=False, on_epoch=True)
        return {"loss": loss}

    def test_step(self, batch, batch_idx):
        x, y = batch
        x_rec = self.forward(x)
        loss = self.loss_fn(x, x_rec)
        mask = self.compute_anomaly_mask(x)
        metrics = self.compute_metrics('test', mask, y)
        self.log_metrics('test', metrics)
        self.log("test_loss", loss, on_step=False, on_epoch=True)
        return {"loss": loss}

    def predict_step(self, x, batch_idx, **kwargs):
        return (x, self.compute_anomaly_mask(x))

    def compute_anomaly_mask(self, x):
        x_rec = self.forward(x)  # (B, C, W, H)
        diff = ((x - x_rec) ** 2).sum(dim=1)  # sum across channels
        mask = diff >= self.model.anomaly_threshold
        return mask

    def log_metrics(self, stage, metrics_res, *args, **kwargs):
        for key, value in metrics_res.items():
            self.log(key, value)  # , sync_dist=True

    def compute_metrics(self, stage, predictions, labels):
        # Reshape labels from [B, 1, H, W] to [B, H, W]
        if labels.shape[1] == 1:
            labels = labels.squeeze(1)
            labels = labels.type(dtype=torch.long)

        if stage == "train":
            return self.train_metrics(predictions, labels)
        elif stage == "val":
            out1 = self.val_metrics(predictions, labels)
            return out1
        elif stage == "test":
            return self.test_metrics(predictions, labels)
