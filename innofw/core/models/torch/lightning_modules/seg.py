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
#
# # import lovely_tensors as lt
#
# # local modules
# from innofw.constants import SegDataKeys, SegOutKeys
# from innofw.core.models.torch.lightning_modules.base import BaseLightningModule
#
#
# # lt.monkey_patch()
#
#
# class SemanticSegmentationLightningModule(BaseLightningModule):
#     def __init__(
#         self,
#         model,
#         losses,
#         optimizer_cfg,
#         scheduler_cfg=None,
#         threshold=0.5,
#         *args: Any,
#         **kwargs: Any,
#     ):
#         """
#         PyTorchLightning module for Semantic Segmentation task
#         ...
#
#         Attributes
#         ----------
#         model : nn.Module
#             model to train
#         losses : losses
#             loss to use while training
#         optimizer_cfg : cfg
#             optimizer configurations
#         scheduler_cfg : cfg
#             scheduler configuration
#         threshold: float
#             threshold to use while training
#
#         Methods
#         -------
#         forward(x):
#             returns result of prediction
#         model_load_checkpoint(path):
#             load checkpoints to the model, used to start with pretrained weights
#
#         """
#         super().__init__(*args, **kwargs)
#         if isinstance(model, DictConfig):
#             self.model = hydra.utils.instantiate(model)
#         else:
#             self.model = model
#
#         self.losses = losses
#         self.optimizer_cfg = optimizer_cfg
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
#         assert self.losses is not None
#         assert self.optimizer_cfg is not None
#
#         self.save_hyperparameters(
#             ignore=["metrics", "optim_config", "scheduler_cfg"]
#         )
#
#     def forward(self, raster):
#         return self.model(raster)
#
#     #     def forward(self, batch: torch.Tensor) -> torch.Tensor:
#     #         """Make a prediction"""
#     #         logits = self.model(batch)
#     #         outs = (logits > self.threshold).to(torch.uint8)
#     #         return outs
#
#     #     def predict_proba(self, batch: torch.Tensor) -> torch.Tensor:
#     #         """Predict and output probabilities"""
#     #         out = self.model(batch)
#     #         return out
#
#     def log_losses(
#         self, name: str, logits: torch.Tensor, masks: torch.Tensor
#     ) -> torch.FloatTensor:
#         """Function to compute and log losses"""
#         total_loss = 0
#         for loss_name, weight, loss in self.losses:
#             # for loss_name in loss_dict:
#             ls_mask = loss(logits, masks)
#             total_loss += weight * ls_mask
#
#             self.log(
#                 f"loss/{name}/{weight} * {loss_name}",
#                 ls_mask,
#                 on_step=False,
#                 on_epoch=True,
#             )
#
#         self.log(f"loss/{name}", total_loss, on_step=False, on_epoch=True)
#         return total_loss
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
#     def log_metrics(self, stage, metrics_res):
#         for key, value in metrics_res.items():
#             self.log(key, value)  # , sync_dist=True
#
#     def stage_step(self, stage, batch, do_logging=False, *args, **kwargs):
#         output = dict()
#         #  check that model is in mode no autograd
#         raster, label = batch[SegDataKeys.image], batch[SegDataKeys.label]
#
#         predictions = self.forward(raster)
#
#         if (
#             predictions.max() > 1 or predictions.min() < 0
#         ):  #  should be configurable via cfg file
#             predictions = torch.sigmoid(predictions)
#
#         output[SegOutKeys.predictions] = predictions
#
#         if stage in ["train", "val"]:
#             loss = self.log_losses(stage, predictions, label)
#             output["loss"] = loss
#
#         # if stage != "predict":
#         #     metrics = self.compute_metrics(stage, predictions, label)
#         #     self.log_metrics(stage, metrics)
#
#         return output
#
#     #     def training_step(self, batch, batch_idx):
#     #         """Process a batch in a training loop"""
#     #         images, masks = batch["scenes"], batch["labels"]
#     #         logits = self.predict_proba(images)
#     #         # compute and log losses
#     #         total_loss = self.log_losses("train", logits.squeeze(), masks.squeeze())
#     #         self.log_metrics("train", torch.sigmoid(logits).view(-1), masks.to(torch.uint8).squeeze().unsqueeze(1).view(-1))
#     #         return {"loss": total_loss, "logits": logits}
#
#     def training_step(self, batch, *args, **kwargs) -> STEP_OUTPUT:
#         return self.stage_step("train", batch, do_logging=True)
#
#     def validation_step(self, batch, *args, **kwargs) -> Optional[STEP_OUTPUT]:
#         return self.stage_step("val", batch)
#
#     #     def model_load_checkpoint(self, path):
#     #         self.model.load_state_dict(torch.load(path)["state_dict"])
#
#     #     def validation_step(self, batch, batch_id):
#     #         """Process a batch in a validation loop"""
#     #         images, masks = batch["scenes"], batch["labels"]
#     #         logits = self.predict_proba(images)
#     #         # compute and log losses
#     #         total_loss = self.log_losses("val", logits.squeeze(), masks.squeeze())
#     #         self.log("val_loss", total_loss, prog_bar=True)
#     #         return {"loss": total_loss, "logits": logits}
#
#     #     def test_step(self, batch, batch_index):
#     #         """Process a batch in a testing loop"""
#     #         images = batch["scenes"]
#
#     #         preds = self.forward(images)
#     #         return {"preds": preds}
#
#     #     def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Any:
#     #         if isinstance(batch, dict):
#     #             batch = batch['scenes']
#     #         preds = self.forward(batch)
#     #         return preds
#
#     def test_step(self, batch, *args, **kwargs) -> Optional[STEP_OUTPUT]:
#         return self.stage_step("test", batch)
#
#
# #     # def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: Optional[int] = None) -> Any:
# #     #     tile, coords = batch[SegDataKeys.image], batch[SegDataKeys.coords]
# #     #
# #     #     prediction = self.forward(tile)
# #     #     if dataloader_idx is None:
# #     #         self.trainer.predict_dataloaders[0].dataset.add_prediction(prediction, coords, batch_idx)
# #     #     else:
# #     #         self.trainer.predict_dataloaders[dataloader_idx].dataset.add_prediction(prediction, coords, batch_idx)
