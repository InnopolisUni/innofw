__all__ = ["SegmentationLM"]

# standard libraries
from typing import Any, Optional

# third-party libraries
import hydra
from omegaconf import DictConfig
from pytorch_lightning.utilities.types import STEP_OUTPUT
from torchmetrics.classification import (
    BinaryF1Score,
    BinaryRecall,
    BinaryPrecision,
    BinaryJaccardIndex,
    MulticlassF1Score,
    MulticlassRecall,
    MulticlassPrecision,
    MulticlassJaccardIndex,
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
        """PyTorchLightning module for Semantic Segmentation task

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
        model_load_checkpoint(path):
            load checkpoints to the model, used to start with pretrained weights
        """
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
        )  # todo: it is slow

        self.train_metrics = metrics.clone(prefix="train_")
        self.val_metrics = metrics.clone(prefix="val_")
        self.test_metrics = metrics.clone(prefix="test_")

        assert self.losses is not None
        assert self.optimizer_cfg is not None

    # def forward(self, batch: torch.Tensor):
    #     return (self.model(batch) > self.threshold).to(torch.uint8)

    def predict_proba(self, batch: torch.Tensor) -> torch.Tensor:
        """Predict and output probabilities"""
        out = self.model(batch)
        return out

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        """Predict and output probabilities"""
        out = self.model(batch)
        return out

    def log_losses(
        self, name: str, logits: torch.Tensor, masks: torch.Tensor
    ) -> torch.FloatTensor:
        """Function to compute and log losses"""
        total_loss = 0
        for loss_name, weight, loss in self.losses:
            # for loss_name in loss_dict:
            if masks.shape[-1] == 1:
                masks = logits.squeeze(-1)
            try:
                ls_mask = loss(logits, masks)
            except RuntimeError:
                ls_mask = loss(logits, masks[:, 0, ...])

            total_loss += weight * ls_mask

            self.log(
                f"loss/{name}/{weight} * {loss_name}",
                ls_mask,
                on_step=False,
                on_epoch=True,
            )
        # val_loss and train_loss
        self.log(f"{name}_loss", total_loss, on_step=False, on_epoch=True)
        return total_loss

    def stage_step(self, stage, batch, do_logging=False, *args, **kwargs):
        output = dict()
        # todo: check that model is in mode no autograd
        raster, label = batch[SegDataKeys.image], batch[SegDataKeys.label]

        predictions = self.forward(raster)
        # if (
        #     predictions.max() > 1 or predictions.min() < 0
        # ):  # todo: should be configurable via cfg file
        #     predictions = torch.sigmoid(predictions)

        output[SegOutKeys.predictions] = predictions

        if stage in ["train", "val"]:
            loss = self.log_losses(stage, predictions, label)
            output["loss"] = loss

        if stage != "predict":
            metrics = self.compute_metrics(stage, predictions, label)  # todo: uncomment
            self.log_metrics(stage, metrics)

        return output

    def compute_metrics(self, stage, predictions, labels):
        if stage == "train":
            return self.train_metrics(predictions.view(-1), labels.view(-1))
        elif stage == "val":
            out1 = self.val_metrics(predictions.view(-1), labels.view(-1))
            return out1
        elif stage == "test":
            return self.test_metrics(predictions.view(-1), labels.view(-1))

    def log_metrics(self, stage, metrics_res):
        for key, value in metrics_res.items():
            self.log(key, value)  # , sync_dist=True

    def training_step(self, batch, *args, **kwargs) -> STEP_OUTPUT:
        return self.stage_step("train", batch, do_logging=True)

    def validation_step(self, batch, *args, **kwargs) -> Optional[STEP_OUTPUT]:
        return self.stage_step("val", batch)

    def test_step(self, batch, *args, **kwargs) -> Optional[STEP_OUTPUT]:
        return self.stage_step("test", batch)

    def model_load_checkpoint(self, path):
        self.model.load_state_dict(torch.load(path)["state_dict"])

    def predict_step(self, batch: Any, batch_indx: int) -> torch.Tensor:
        """Predict and output binary predictions"""
        if isinstance(batch, dict):
            input_tensor = batch[SegDataKeys.image]
        else:
            input_tensor = batch

        proba = self.predict_proba(input_tensor)
        predictions = (proba > self.threshold).to(torch.uint8)
        return predictions


class MulticlassSemanticSegmentationLightningModule(BaseLightningModule):
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
        """PyTorchLightning module for Semantic Segmentation task

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
        model_load_checkpoint(path):
            load checkpoints to the model, used to start with pretrained weights
        """
        super().__init__(*args, **kwargs)
        if isinstance(model, DictConfig):
            self.model = hydra.utils.instantiate(model)
        else:
            self.model = model

        self.losses = losses
        self.optimizer_cfg = optimizer_cfg
        self.scheduler_cfg = scheduler_cfg
        self.threshold = threshold

        if hasattr(model, "num_classes"):
            num_classes = model.num_classes
        elif hasattr(model, "num_labels"):
            num_classes = model.num_labels
        elif hasattr(model, "segmentation_head"):
            num_classes = model.segmentation_head[0].out_channels
        else:
            raise AttributeError(
                f'Please make sure {type(model).__name__} has either "num_classes" or "num_labels" attribute,'
                f" or has a segmentation head."
            )

        metrics = MetricCollection(
            [
                MulticlassF1Score(
                    threshold=threshold,
                    num_classes=num_classes,
                    average="weighted",
                    ignore_index=0,
                ),
                MulticlassRecall(
                    threshold=threshold,
                    num_classes=num_classes,
                    average="weighted",
                    ignore_index=0,
                ),
                MulticlassPrecision(
                    threshold=threshold,
                    num_classes=num_classes,
                    average="weighted",
                    ignore_index=0,
                ),
                MulticlassJaccardIndex(
                    threshold=threshold,
                    num_classes=num_classes,
                    average="weighted",
                    ignore_index=0,
                ),
            ]
        )

        self.train_metrics = metrics.clone(prefix="train_")
        self.val_metrics = metrics.clone(prefix="val_")

        assert self.losses is not None
        assert self.optimizer_cfg is not None

    # def forward(self, batch: torch.Tensor):
    #     return (self.model(batch) > self.threshold).to(torch.uint8)

    def predict_proba(self, batch: torch.Tensor) -> torch.Tensor:
        """Predict and output probabilities"""
        out = self.model(batch)
        return out

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        """Predict and output probabilities"""
        out = self.model(batch)
        return out

    def log_losses(
        self, name: str, logits: torch.Tensor, masks: torch.Tensor
    ) -> torch.FloatTensor:
        """Function to compute and log losses"""
        total_loss = 0
        for loss_name, weight, loss in self.losses:
            # for loss_name in loss_dict:
            if masks.shape[-1] == 1:
                masks = logits.squeeze(-1)
            if masks.shape[1] == 1:
                masks = masks.squeeze(1)
                # Masks are considered to be indices
                masks = masks.type(dtype=torch.long)

            ls_mask = loss(logits, masks)
            total_loss += weight * ls_mask

            self.log(
                f"loss/{name}/{weight} * {loss_name}",
                ls_mask,
                on_step=False,
                on_epoch=True,
            )
        # val_loss and train_loss
        self.log(f"{name}_loss", total_loss, on_step=False, on_epoch=True)
        return total_loss

    def stage_step(self, stage, batch, do_logging=False, *args, **kwargs):
        output = dict()
        raster, label = batch[SegDataKeys.image], batch[SegDataKeys.label]

        predictions = self.forward(raster)

        output[SegOutKeys.predictions] = predictions

        if stage in ["train", "val"]:
            loss = self.log_losses(stage, predictions, label)
            output["loss"] = loss

        if stage != "predict":
            metrics = self.compute_metrics(stage, predictions, label)
            self.log_metrics(stage, metrics)

        return output

    def compute_metrics(self, stage, predictions, labels):
        # Reshape labels from [B, 1, H, W] to [B, H, W]
        if labels.shape[1] == 1:
            labels = labels.squeeze(1)
            # Labels are masks of indices
            labels = labels.type(dtype=torch.long)

        if stage == "train":
            return self.train_metrics(predictions, labels)
        elif stage == "val":
            out1 = self.val_metrics(predictions, labels)
            return out1
        elif stage == "test":
            return self.test_metrics(predictions, labels)

    def log_metrics(self, stage, metrics_res):
        for key, value in metrics_res.items():
            self.log(key, value)  # , sync_dist=True

    def training_step(self, batch, *args, **kwargs) -> STEP_OUTPUT:
        return self.stage_step("train", batch, do_logging=True)

    def validation_step(self, batch, *args, **kwargs) -> Optional[STEP_OUTPUT]:
        return self.stage_step("val", batch)

    def test_step(self, batch, *args, **kwargs) -> Optional[STEP_OUTPUT]:
        return self.stage_step("test", batch)

    def model_load_checkpoint(self, path):
        self.model.load_state_dict(torch.load(path)["state_dict"])
