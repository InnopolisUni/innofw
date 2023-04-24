# from pytorch_lightning.callbacks import Callback
# import torch
# import numpy as np
# from typing import Optional, Any
# class SegmentationImagesPredictionsDisplayCallback(Callback):
#     def __init__(self):
#         self.train_items = []
#         self.val_items = []
#     def on_stage_batch_end(self, outputs, batch, stage_items):
#         images = batch["scenes"]
#         masks = batch["labels"].long().squeeze()
#         logits = outputs["logits"].squeeze()
#         threshold = 0.5
#         logits = (logits > threshold).to(torch.uint8)
#         stage_items.append(
#             {
#                 "image": images[0].squeeze(),
#                 "mask": masks[0],
#                 "pred": logits[0],
#             }
#         )
#     def on_stage_epoch_end(self, trainer, pl_module: "pl.LightningModule", stage_items):
#         tensorboard = pl_module.logger.experiment
#         images = [item["image"].cpu().numpy() for item in stage_items][:5]
#         masks = [item["mask"].cpu().numpy() for item in stage_items][:5]
#         preds = [item["pred"].cpu().numpy() for item in stage_items][:5]
#         images = np.array(images)
#         masks = np.array(masks)
#         preds = np.array(preds)
#         masks = np.expand_dims(masks, 1)
#         preds = np.expand_dims(preds, 1)
#         tensorboard.add_images("images", images, trainer.current_epoch)
#         tensorboard.add_images("preds", preds, trainer.current_epoch)
#         tensorboard.add_images("masks", masks, trainer.current_epoch)
#     def on_train_batch_end(
#         self,
#         trainer: "pl.Trainer",
#         pl_module: "pl.LightningModule",
#         outputs,
#         batch: Any,
#         batch_idx: int,
#         unused: Optional[int] = 0,
#     ) -> None:
#         self.on_stage_batch_end(outputs, batch, self.train_items)
#     def on_train_epoch_end(
#         self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
#     ) -> None:
#         self.on_stage_epoch_end(trainer, pl_module, self.train_items)
# random interim images
# top best and worst images
# section 1: Wandb image logging
#
# section 2: callback
from abc import ABC
from abc import abstractmethod

import torch
from pytorch_lightning import Callback

import wandb


class BaseCallback(Callback, ABC):
    @abstractmethod
    def is_callback_reqs_met(*args, **kwargs):
        pass


class WandbSegInterimImgVisCallback(BaseCallback):
    def __init__(self, val_samples, num_samples=32):
        super().__init__()
        self.val_imgs, self.val_labels = val_samples
        self.val_imgs = self.val_imgs[:num_samples]
        self.val_labels = self.val_labels[:num_samples]

    def on_validation_epoch_end(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:
        super().on_validation_epoch_end(trainer, pl_module)
        threshold = 0.5

        val_imgs = self.val_imgs.to(device=pl_module.device)
        logits = pl_module(val_imgs)
        preds = (logits > threshold).to(torch.uint8)

        # trainer.logger.experiment.log(
        #     {"val/logits": wandb.Histogram(torch.flatten(torch.cat()))}  # todo: histogram
        # )
        trainer.logger.experiment.log(
            {
                "examples": [
                    wandb.Image(x, caption=f"Pred:{pred}, Label:{y}")
                    for x, pred, y in zip(val_imgs, preds, self.val_labels)
                ],
                "global_step": trainer.global_step,
            }
        )


# class WandbSegTopImgVisCallback(BaseCallback):
#     pass


# # section 3: validators
# class CheckBatchGradient(pl.Callback):

#     def on_train_start(self, trainer, model):
#         n = 0

#         example_input = model.example_input_array.to(model.device)
#         example_input.requires_grad = True

#         model.zero_grad()
#         output = model(example_input)
#         output[n].abs().sum().backward()

#         zero_grad_inds = list(range(example_input.size(0)))
#         zero_grad_inds.pop(n)

#         if example_input.grad[zero_grad_inds].abs().sum().item() > 0
#             raise RuntimeError("Your model mixes data across the batch dimension!")
