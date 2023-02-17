__all__ = ["SegmentationLM"]

# standard libraries
import logging
from typing import Any, Optional

# third-party libraries
import hydra
import pytorch_lightning as pl
from omegaconf import DictConfig
from pytorch_lightning.utilities.types import STEP_OUTPUT
from torchmetrics.classification import (
    BinaryF1Score,
    BinaryRecall,
    BinaryPrecision,
    BinaryJaccardIndex,
)
import torch
from torchmetrics import MetricCollection
# import lovely_tensors as lt

# local modules
from innofw.constants import SegDataKeys, SegOutKeys
from innofw.core.models.torch.lightning_modules.base import BaseLightningModule


# lt.monkey_patch()


class SemanticSegmentationLightningModule(BaseLightningModule):
    def __init__(
        self,
        model,
        losses,
        optimizer_cfg,
        scheduler_cfg=None,
        threshold=0.5,
        *args: Any,
        **kwargs: Any,
    ):
        super().__init__(*args, **kwargs)
        if isinstance(model, DictConfig):
            self.model = hydra.utils.instantiate(model)
        else:
            self.model = model

        self.losses = losses
        self.optimizer_cfg = optimizer_cfg
        self.scheduler_cfg = scheduler_cfg
        self.threshold = threshold

        metrics = MetricCollection(
            [
                BinaryF1Score(threshold=threshold),
                BinaryPrecision(threshold=threshold),
                BinaryRecall(threshold=threshold),
                BinaryJaccardIndex(threshold=threshold),
            ]
        )
        self.train_metrics = metrics.clone(prefix="train_")
        self.val_metrics = metrics.clone(prefix="val_")
        self.test_metrics = metrics.clone(prefix="test_")

        assert self.losses is not None
        assert self.optimizer_cfg is not None

    def forward(self, raster):
        return self.model(raster)

    def log_losses(
            self, name: str, logits: torch.Tensor, masks: torch.Tensor
    ) -> torch.FloatTensor:
        """Function to compute and log losses"""
        total_loss = 0
        for loss_name, weight, loss in self.losses:
            # for loss_name in loss_dict:
            ls_mask = loss(logits, masks)
            total_loss += weight * ls_mask

            self.log(
                f"loss/{name}/{weight} * {loss_name}",
                ls_mask,
                on_step=False,
                on_epoch=True,
            )

        self.log(f"loss/{name}", total_loss, on_step=False, on_epoch=True)
        return total_loss

    def stage_step(self, stage, batch, do_logging=False, *args, **kwargs):
        output = dict()
        # todo: check that model is in mode no autograd
        raster, label = batch[SegDataKeys.image], batch[SegDataKeys.label]

        predictions = self.forward(raster)
        if (
            predictions.max() > 1 or predictions.min() < 0
        ):  # todo: should be configurable via cfg file
            predictions = torch.sigmoid(predictions)

        output[SegOutKeys.predictions] = predictions

        if stage in ["train", "val"]:
            loss = self.log_losses(stage, predictions, label)
            output["loss"] = loss

        # if stage != "predict":
        #     metrics = self.compute_metrics(stage, predictions, label)  # todo: uncomment
        #     self.log_metrics(stage, metrics)

        return output

    def training_step(self, batch, *args, **kwargs) -> STEP_OUTPUT:
        return self.stage_step("train", batch, do_logging=True)

    def validation_step(self, batch, *args, **kwargs) -> Optional[STEP_OUTPUT]:
        return self.stage_step("val", batch)

    def test_step(self, batch, *args, **kwargs) -> Optional[STEP_OUTPUT]:
        return self.stage_step("test", batch)
