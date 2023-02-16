# # third party libraries
# from typing import Any

# # from pytorch_lightning import LightningModule
from innofw.core.models.torch.lightning_modules.base import BaseLightningModule
# import torch


# class SemanticSegmentationLightningModule(
#     BaseLightningModule
# ):
#     """
#     PyTorchLightning module for Semantic Segmentation task
#     ...

#     Attributes
#     ----------
#     model : nn.Module
#         model to train
#     losses : losses
#         loss to use while training
#     optimizer_cfg : cfg
#         optimizer configurations
#     scheduler_cfg : cfg
#         scheduler configuration
#     threshold: float
#         threshold to use while training

#     Methods
#     -------
#     forward(x):
#         returns result of prediction
#     model_load_checkpoint(path):
#         load checkpoints to the model, used to start with pretrained weights

#     """

#     def __init__(
#             self,
#             model,
#             losses,
#             optimizer_cfg,
#             scheduler_cfg,
#             threshold: float = 0.5,
#             *args,
#             **kwargs,
#     ):
#         super().__init__(*args, **kwargs)
#         self.model = model
#         self.losses = losses

#         self.optimizer_cfg = optimizer_cfg
#         self.scheduler_cfg = scheduler_cfg

#         self.threshold = threshold

#         assert self.losses is not None
#         assert self.optimizer_cfg is not None


#     def model_load_checkpoint(self, path):
#         self.model.load_state_dict(torch.load(path)["state_dict"])

#     def forward(self, batch: torch.Tensor) -> torch.Tensor:
#         """Make a prediction"""
#         logits = self.model(batch)
#         outs = (logits > self.threshold).to(torch.uint8)
#         return outs

#     def predict_proba(self, batch: torch.Tensor) -> torch.Tensor:
#         """Predict and output probabilities"""
#         out = self.model(batch)
#         return out

#     def training_step(self, batch, batch_idx):
#         """Process a batch in a training loop"""
#         images, masks = batch["scenes"], batch["labels"]
#         logits = self.predict_proba(images)
#         # compute and log losses
#         total_loss = self.log_losses("train", logits.squeeze(), masks.squeeze())
#         self.log_metrics("train", torch.sigmoid(logits).view(-1), masks.to(torch.uint8).squeeze().unsqueeze(1).view(-1))
#         return {"loss": total_loss, "logits": logits}

#     def validation_step(self, batch, batch_id):
#         """Process a batch in a validation loop"""
#         images, masks = batch["scenes"], batch["labels"]
#         logits = self.predict_proba(images)
#         # compute and log losses
#         total_loss = self.log_losses("val", logits.squeeze(), masks.squeeze())
#         self.log("val_loss", total_loss, prog_bar=True)
#         return {"loss": total_loss, "logits": logits}

#     def test_step(self, batch, batch_index):
#         """Process a batch in a testing loop"""
#         images = batch["scenes"]

#         preds = self.forward(images)
#         return {"preds": preds}

#     def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Any:
#         if isinstance(batch, dict):
#             batch = batch['scenes']
#         preds = self.forward(batch)
#         return preds

#     def log_losses(
#             self, name: str, logits: torch.Tensor, masks: torch.Tensor
#     ) -> torch.FloatTensor:
#         """Function to compute and log losses"""
#         total_loss = 0
#         for loss_name, weight, loss in self.losses:
#             # for loss_name in loss_dict:
#             ls_mask = loss(logits, masks)
#             total_loss += weight * ls_mask

#             self.log(
#                 f"loss/{name}/{weight} * {loss_name}",
#                 ls_mask,
#                 on_step=False,
#                 on_epoch=True,
#             )

#         self.log(f"loss/{name}", total_loss, on_step=False, on_epoch=True)
#         return total_loss
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
# from torch.cuda.amp import GradScaler, autocast
import torch
from torchmetrics import MetricCollection
# import lovely_tensors as lt

# local modules
from innofw.constants import SegDataKeys, SegOutKeys


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

        # self.scaler = GradScaler(enabled=True)
        # self.save_hyperparameters(ignore=["metrics", "optim_config", "scheduler_cfg"])
        assert self.losses is not None
        assert self.optimizer_cfg is not None

#     def load_torch_ckpt(self, ckpt_path):
#         self.model.load_state_dict(torch.load(ckpt_path, map_location="cuda:0")['model_boundary_state_dict'])  # state_dict

    def forward(self, raster):
        return self.model(raster)


#     # def configure_optimizers(self):
#     #     output = {}

#     #     # instantiate the optimizer
#     #     optimizer = hydra.utils.instantiate(
#     #         self.optim_config, params=self.model.parameters()
#     #     )
#     #     output["optimizer"] = optimizer

#     #     if self.scheduler_cfg is not None:
#     #         # instantiate the scheduler
#     #         scheduler = hydra.utils.instantiate(self.scheduler_cfg, optimizer=optimizer)
#     #         output["lr_scheduler"] = scheduler

#     #     return output

#     # def backward(
#     #     self,
#     #     loss: Tensor,
#     #     optimizer: Optional[Optimizer],
#     #     optimizer_idx: Optional[int],
#     #     *args,
#     #     **kwargs,
#     # ) -> None:
#     #     # return super().backward(loss, optimizer, optimizer_idx, *args, **kwargs):
#     #     self.scaler.scale(loss).backward()
#     #     self.scaler.step(optimizer)
#     #     self.scaler.update()
#     #     self.scheduler.step()
#     #     torch.cuda.synchronize()

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

#     def compute_metrics(self, stage, predictions, labels):
#         if stage == "train":
#             return self.train_metrics(predictions.view(-1), labels.view(-1))
#         elif stage == "val":
#             out1 = self.val_metrics(predictions.view(-1), labels.view(-1))
#             return out1
#         elif stage == "test":
#             return self.test_metrics(predictions.view(-1), labels.view(-1))

#     # def log_losses(self, stage, losses_res):
#     #     self.log(
#     #         f"{stage}_loss", losses_res, sync_dist=True
#     #     )  # todo: check when to use this sync_dist=True

#     def log_metrics(self, stage, metrics_res):
#         for key, value in metrics_res.items():
#             self.log(key, value, sync_dist=True)

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

#     # def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: Optional[int] = None) -> Any:
#     #     tile, coords = batch[SegDataKeys.image], batch[SegDataKeys.coords]
#     #
#     #     prediction = self.forward(tile)
#     #     if dataloader_idx is None:
#     #         self.trainer.predict_dataloaders[0].dataset.add_prediction(prediction, coords, batch_idx)
#     #     else:
#     #         self.trainer.predict_dataloaders[dataloader_idx].dataset.add_prediction(prediction, coords, batch_idx)
