# import logging

# import numpy as np
# import pytorch_lightning as pl
# import torch
# from pytorch_lightning import Callback
# from segmentation.constants import SegDataKeys
# from segmentation.constants import SegOutKeys

# import wandb

# #
# #


# class SegmentationLogPredictions(Callback):
#     def __init__(self, logging_interval: int = 5, num_predictions: int = 4):
#         self.logging_interval = logging_interval
#         self.num_predictions = num_predictions

#         self.different_images = True

#         self.train_containers = {
#             SegDataKeys.image: [],
#             SegDataKeys.label: [],
#             SegOutKeys.predictions: [],
#         }
#         self.val_containers = {
#             SegDataKeys.image: [],
#             SegDataKeys.label: [],
#             SegOutKeys.predictions: [],
#         }
#         self.class_labels = {0: "background", 1: "arable"}

#     def setup(self, trainer, pl_module, stage=None, *args, **kwargs):
#         # 1. init
#         #   a. get number of batches in the set
#         tdl = trainer.datamodule.train_dataloader()
#         vdl = trainer.datamodule.val_dataloader()

#         self.train_batch_size: int = len(iter(vdl).next()[SegDataKeys.image])
#         self.val_batch_size: int = len(iter(vdl).next()[SegDataKeys.image])

#         train_ds_size = len(tdl.dataset)
#         val_ds_size = len(vdl.dataset)

#         self.train_num_predictions = (
#             self.num_predictions
#             if train_ds_size > self.num_predictions
#             else train_ds_size
#         )
#         self.val_num_predictions = (
#             self.num_predictions
#             if val_ds_size > self.num_predictions
#             else val_ds_size
#         )

#         self.train_indices = sorted(
#             np.random.choice(train_ds_size, self.train_num_predictions)
#         )
#         self.val_indices = sorted(
#             np.random.choice(val_ds_size, self.val_num_predictions)
#         )

#         logging.info(f"train indices {self.train_indices}")
#         logging.info(f"val indices {self.val_indices}")

#     def on_stage_batch_end(
#         self,
#         indices,
#         batch_size,
#         batch,
#         outputs,
#         batch_idx,
#         container,
#         trainer,
#     ):
#         if trainer.current_epoch % self.logging_interval != 0:
#             return

#         for i in indices:
#             if i >= (batch_idx + 1) * batch_size:
#                 break

#             if i >= batch_idx * batch_size:
#                 offset = i - batch_idx * batch_size
#                 prediction = outputs[SegOutKeys.predictions][offset]
#                 image = batch[SegDataKeys.image][offset]
#                 label = batch[SegDataKeys.label][offset]

#                 container[SegDataKeys.image].append(image)
#                 container[SegDataKeys.label].append(label)
#                 container[SegOutKeys.predictions].append(prediction)

#     def on_validation_batch_end(
#         self,
#         trainer,
#         pl_module,
#         outputs,
#         batch,
#         batch_idx,
#         dataloader_idx=None,
#     ):
#         self.on_stage_batch_end(
#             self.val_indices,
#             self.val_batch_size,
#             batch,
#             outputs,
#             batch_idx,
#             self.val_containers,
#             trainer,
#         )

#     def on_train_batch_end(
#         self,
#         trainer,
#         pl_module,
#         outputs,
#         batch,
#         batch_idx,
#         dataloader_idx=None,
#     ):
#         self.on_stage_batch_end(
#             self.train_indices,
#             self.train_batch_size,
#             batch,
#             outputs,
#             batch_idx,
#             self.train_containers,
#             trainer,
#         )

#     def on_train_epoch_start(
#         self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
#     ) -> None:
#         self.train_containers = {
#             SegDataKeys.image: [],
#             SegDataKeys.label: [],
#             SegOutKeys.predictions: [],
#         }

#     def on_validation_epoch_start(
#         self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
#     ) -> None:
#         self.val_containers = {
#             SegDataKeys.image: [],
#             SegDataKeys.label: [],
#             SegOutKeys.predictions: [],
#         }

#     def on_train_epoch_end(
#         self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
#     ) -> None:
#         self.log_predictions(trainer, "train", self.train_containers)

#     def on_validation_epoch_end(
#         self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
#     ) -> None:
#         self.log_predictions(trainer, "val", self.val_containers)

#     def log_predictions(self, trainer, stage, container):
#         try:
#             predictions = container[SegOutKeys.predictions]
#             images = container[SegDataKeys.image]
#             labels = container[SegDataKeys.label]

#             if any(map(lambda x: len(x) == 0, [predictions, images, labels])):
#                 return

#             for prediction, label, image in zip(
#                 predictions, labels, images, range(self.num_predictions)
#             ):
#                 prep_prediction = (
#                     torch.sigmoid(prediction[0, ...]).detach().cpu().numpy()
#                     > 0.5
#                 ).astype(np.uint8)
#                 prep_mask = label.cpu().numpy().astype(np.uint8)

#                 img_pred_mask = wandb.Image(
#                     image,
#                     masks={
#                         "predictions": {
#                             "mask_data": prep_prediction,
#                             "class_labels": self.class_labels,
#                         },
#                         "ground_truth": {
#                             "mask_data": prep_mask,
#                             "class_labels": self.class_labels,
#                         },
#                     },  # .float()
#                 )
#                 # logger log an image
#                 trainer.logger.experiment.log(
#                     {f"{stage}_images": img_pred_mask}
#                 )

#         except Exception as e:
#             logging.info(e)
