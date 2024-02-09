from abc import ABC
from typing import Any

import numpy as np
from pytorch_lightning.callbacks import Callback

class BaseLogPredictionsCallback(Callback, ABC):
    pass
    # def __init__(self, logging_batch_interval: int = 10):
    #     pass


# class LogVisualTensorboard(Callback):
#     pass
#
#
# class LogScalarTensorboard(Callback):
#     pass

import cv2
import random

def plot_one_box(x, image, color=None, label=None, line_thickness=None):
    """
    source: https://github.com/waittim/draw-YOLO-box/blob/main/draw_box.py
    """
    # Plots one bounding box on image img
    tl = (
        line_thickness
        or round(0.001 * (image.shape[0] + image.shape[1]) / 2) + 1
    )  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(image, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(image, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(
            image,
            label,
            (c1[0], c1[1] - 2),
            0,
            tl / 3,
            [225, 255, 255],
            thickness=tf,
            lineType=cv2.LINE_AA,
        )


class LogPredictionsDetectionCallback(BaseLogPredictionsCallback):
    """
    reference: https://github.com/Lightning-AI/lightning-bolts/blob/master/pl_bolts/callbacks/vision/confused_logit.py#L20-L164
    """

    def __init__(self, logging_batch_interval: int = 10, *args, **kwargs):
        self.logging_batch_interval: int = logging_batch_interval
        self.labels = ["lep1", "lep2", "lep3", "lep4"]
        num_classes = len(self.labels)
        self.colors = [
            random.randint(0, 255)
            for _ in range(3)
            for _ in range(num_classes)
        ]

    def on_train_batch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        outputs,  # : STEP_OUTPUT
        batch: Any,
        batch_idx: int,
        unused: int = 0,
    ) -> None:
        if (
            batch_idx == 0
            or (batch_idx + 1) % self.logging_batch_interval != 0
        ):
            return

        tensorboard = pl_module.logger.experiment
        tensorboard.add_scalar("one", batch_idx, batch_idx)

        inputs, targets = batch
        img = inputs[0].detach().cpu().numpy()
        img = np.moveaxis(img, 0, -1)

        for box, label in zip(
            targets[0]["boxes"].detach().cpu().numpy(),
            targets[0]["labels"].detach().cpu().numpy(),
        ):
            plot_one_box(
                list(box),
                img,
                color=self.colors[label],
                label=self.labels[label],
                line_thickness=None,
            )

        img = np.moveaxis(img, -1, 0)
        tensorboard.add_image("image with labels", img, batch_idx)

class LogPredictionsSegmentationCallback(BaseLogPredictionsCallback):
    pass
class BaseLogMetricsCallback(Callback):
    pass
class LogMetricsDetectionCallback(BaseLogMetricsCallback):
    pass
class LogMetricsSegmentationCallback(BaseLogMetricsCallback):
    pass


# class LoggingSMPMetricsCallback(Callback):
#     def __init__(self, metrics, log_every_n_steps=50,
#                  *args,
#                  **kwargs
#                  ):
#         self.metrics = metrics
#         self.train_scores = {}
#         self.val_scores = {}
#         self.test_scores = {}
#         self.log_every_n_steps = log_every_n_steps
#
#     def on_stage_batch_end(self, outputs, batch, dict_to_store):
#         masks = batch['labels'].long().squeeze()
#         logits = outputs['logits'].squeeze()
#         threshold = 0.4
#         tp, fp, fn, tn = smp.metrics.get_stats(logits, masks, mode='binary', threshold=threshold)
#
#         def compute_func(func):
#             res = hydra.utils.instantiate(func, tp, fp, fn, tn, reduction="micro").item()
#             res = round(res, 3)
#             return res
#
#         for name, func in self.metrics.items():
#             score = compute_func(func)
#
#             if name not in dict_to_store:
#                 dict_to_store[name] = [score]
#             else:
#                 dict_to_store[name].append(score)
#
#     def on_stage_epoch_end(self, trainer, pl_module: "pl.LightningModule", dict_with_scores):
#         for name, scores in dict_with_scores.items():
#             tensorboard = pl_module.logger.experiment
#             mean_score = torch.mean(torch.tensor(scores))
#             # for logger in trainer.loggers:
#             #     logger.add_scalar(f"metrics/train/{name}", mean_score,
#             #                       step=trainer.fit_loop.epoch_loop.batch_idx)
#             tensorboard.add_scalar(f"metrics/train/{name}", mean_score,
#                                    trainer.current_epoch)
#        if (batch_idx + 1) % self.logging_batch_interval != 0:
#             return
#
#         tensorboard = pl_module.logger.experiment

#     def on_train_batch_end(
#             self,
#             trainer: "pl.Trainer",
#             pl_module: "pl.LightningModule",
#             outputs,  # : STEP_OUTPUT
#             batch: Any,
#             batch_idx: int,
#             unused: Optional[int] = 0,
#     ) -> None:
#         if (batch_idx + 1) % self.log_every_n_steps == 0:
#             return
#
#         self.on_stage_batch_end(outputs, batch, self.train_scores)
#
#     def on_validation_batch_end(
#             self,
#             trainer: "pl.Trainer",
#             pl_module: "pl.LightningModule",
#             outputs,
#             batch: Any,
#             batch_idx: int,
#             dataloader_idx: int,
#     ) -> None:
#         if (batch_idx + 1) % self.log_every_n_steps == 0:
#             return
#
#         self.on_stage_batch_end(outputs, batch, self.val_scores)
#
#     def on_train_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
#         self.on_stage_epoch_end(trainer, pl_module, self.train_scores)
#
#     def on_validation_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
#         self.on_stage_epoch_end(trainer, pl_module, self.val_scores)
#
#
#
# # class SegmentationImagesPredictionsDisplayCallback(Callback):
# #     def __init__(self):
# #         self.train_items = []
# #         self.val_items = []
# #
# #     def on_stage_batch_end(self, outputs, batch, stage_items):
# #         images = batch['scenes']
# #         masks = batch['labels'].long().squeeze()
# #         logits = outputs['logits'].squeeze()
# #
# #         threshold = 0.5
# #         logits = (logits > threshold).to(torch.uint8)
# #
# #         stage_items.append({
# #             'image': images[0].squeeze(),
# #             'mask': masks[0],
# #             'pred': logits[0],
# #         })
# #
# #     def on_stage_epoch_end(self, trainer, pl_module: "pl.LightningModule", stage_items):
# #         tensorboard = pl_module.logger.experiment
# #         images = [item['image'].cpu().numpy() for item in stage_items][:5]
# #         masks = [item['mask'].cpu().numpy() for item in stage_items][:5]
# #         preds = [item['pred'].cpu().numpy() for item in stage_items][:5]
# #
# #         images = np.array(images)
# #         masks = np.array(masks)
# #         preds = np.array(preds)
# #
# #         masks = np.expand_dims(masks, 1)
# #         preds = np.expand_dims(preds, 1)
# #
# #         tensorboard.add_images('images', images, trainer.current_epoch)
# #         tensorboard.add_images('preds', preds, trainer.current_epoch)
# #         tensorboard.add_images('masks', masks, trainer.current_epoch)
# #
# #     def on_train_batch_end(
# #             self,
# #             trainer: "pl.Trainer",
# #             pl_module: "pl.LightningModule",
# #             outputs,
# #             batch: Any,
# #             batch_idx: int,
# #             unused: Optional[int] = 0,
# #     ) -> None:
# #         self.on_stage_batch_end(outputs, batch, self.train_items)
# #
# #     def on_train_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
# #         self.on_stage_epoch_end(trainer, pl_module, self.train_items)
