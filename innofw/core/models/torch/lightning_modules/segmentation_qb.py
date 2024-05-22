# __all__ = ["SegmentationLM"]
#
# # standard libraries
# from typing import Any, Optional
#
# # third-party libraries
# import hydra
# from omegaconf import DictConfig
# from pytorch_lightning.utilities.types import STEP_OUTPUT
# from torchmetrics.classification import (
#     BinaryF1Score,
#     BinaryRecall,
#     BinaryPrecision,
#     BinaryJaccardIndex,
# )
# import torch
# from torchmetrics import MetricCollection
# import lovely_tensors as lt
#
# # local modules
# from innofw.constants import SegDataKeys, SegOutKeys
# from innofw.core.models.torch.lightning_modules.base import BaseLightningModule
#
#
# lt.monkey_patch()
#
#
# class SegmentationLM(BaseLightningModule):
#     """
#     PyTorchLightning module for Semantic Segmentation task
#     ...
#
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
#
#     Methods
#     -------
#     forward(x):
#         returns result of prediction
#     model_load_checkpoint(path):
#         load checkpoints to the model, used to start with pretrained weights
#
#     """
#
#     def __init__(
#         self,
#         model,
#         losses,
#         # metrics,  # todo: add
#         optimizer_cfg,
#         scheduler_cfg=None,
#         threshold=0.5,
#         *args: Any,
#         **kwargs: Any,
#     ):
#         super().__init__(*args, **kwargs)
#         if isinstance(model, DictConfig):
#             self.model = hydra.utils.instantiate(model)
#         else:
#             self.model = model
#
#         self.loss = hydra.utils.instantiate(losses)
#         self.optim_config = optimizer_cfg
#         self.scheduler_cfg = scheduler_cfg
#         self.threshold = threshold
#
#         metrics = MetricCollection(
#             [
#                 BinaryF1Score(threshold=threshold),
#                 BinaryPrecision(threshold=threshold),
#                 BinaryRecall(threshold=threshold),
#                 BinaryJaccardIndex(threshold=threshold),
#             ]
#         )
#         self.train_metrics = metrics.clone(prefix="train_")
#         self.val_metrics = metrics.clone(prefix="val_")
#         self.test_metrics = metrics.clone(prefix="test_")
#
#         # self.scaler = GradScaler(enabled=True)
#         self.save_hyperparameters(
#             ignore=["metrics", "optim_config", "scheduler_cfg"]
#         )
#
#     def model_load_checkpoint(self, path):
#         self.model.load_state_dict(torch.load(path)["state_dict"])
#
#     def forward(self, raster):
#         return self.model(raster)
#
#     def configure_optimizers(self):
#         output = {}
#
#         # instantiate the optimizer
#         optimizer = hydra.utils.instantiate(
#             self.optim_config, params=self.model.parameters()
#         )
#         output["optimizer"] = optimizer
#
#         if self.scheduler_cfg is not None:
#             # instantiate the scheduler
#             scheduler = hydra.utils.instantiate(
#                 self.scheduler_cfg, optimizer=optimizer
#             )
#             output["lr_scheduler"] = scheduler
#
#         return output
#
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
#
#     def compute_loss(self, predictions, labels):
#         loss = self.loss(predictions, labels)
#
#         # with autocast(enabled=train_cfg["AMP"]):
#         #     logits = model(img)
#         #     loss = loss_fn(logits, lbl)
#         return loss
#         # return self.scaler.scale(loss)  # todo: refactor !!!!
#
#     def compute_metrics(self, stage, predictions, labels):
#         if stage == "train":
#             return self.train_metrics(predictions.view(-1), labels.view(-1))
#         elif stage == "val":
#             out1 = self.val_metrics(predictions.view(-1), labels.view(-1))
#             return out1
#         elif stage == "test":
#             return self.test_metrics(predictions.view(-1), labels.view(-1))
#
#     def log_losses(self, stage, losses_res):
#         self.log(
#             f"{stage}_loss", losses_res, sync_dist=True
#         )  # todo: check when to use this sync_dist=True
#
#     def log_metrics(self, stage, metrics_res):
#         for key, value in metrics_res.items():
#             self.log(key, value, sync_dist=True)
#
#     def stage_step(self, stage, batch, do_logging=False, *args, **kwargs):
#         output = dict()
#         # todo: check that model is in mode no autograd
#         raster, label = batch[SegDataKeys.image], batch[SegDataKeys.label]
#
#         predictions = self.forward(raster)
#         # if (
#         #     predictions.max() > 1 or predictions.min() < 0
#         # ):  # todo: should be configurable via cfg file
#         #     predictions = torch.sigmoid(predictions)
#
#         output[SegOutKeys.predictions] = predictions
#
#         if stage in ["train", "val"]:
#             loss = self.compute_loss(predictions, label)
#             self.log_losses(stage, loss)
#             output["loss"] = loss
#
#         if stage != "predict":
#             metrics = self.compute_metrics(
#                 stage, predictions, label
#             )  # todo: uncomment
#             self.log_metrics(stage, metrics)
#
#         return output
#
#     def training_step(self, batch, *args, **kwargs) -> STEP_OUTPUT:
#         return self.stage_step("train", batch, do_logging=True)
#
#     def validation_step(self, batch, *args, **kwargs) -> Optional[STEP_OUTPUT]:
#         return self.stage_step("val", batch)
#
#     def test_step(self, batch, *args, **kwargs) -> Optional[STEP_OUTPUT]:
#         return self.stage_step("test", batch)
#
#     # def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: Optional[int] = None) -> Any:
#     #     tile, coords = batch[SegDataKeys.image], batch[SegDataKeys.coords]
#     #
#     #     prediction = self.forward(tile)
#     #     if dataloader_idx is None:
#     #         self.trainer.predict_dataloaders[0].dataset.add_prediction(prediction, coords, batch_idx)
#     #     else:
#     #         self.trainer.predict_dataloaders[dataloader_idx].dataset.add_prediction(prediction, coords, batch_idx)
