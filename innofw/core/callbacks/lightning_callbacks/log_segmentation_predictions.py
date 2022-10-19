from pytorch_lightning.callbacks import Callback
import torch
import numpy as np
from typing import Optional, Any


class SegmentationImagesPredictionsDisplayCallback(Callback):
    def __init__(self):
        self.train_items = []
        self.val_items = []

    def on_stage_batch_end(self, outputs, batch, stage_items):
        images = batch["scenes"]
        masks = batch["labels"].long().squeeze()
        logits = outputs["logits"].squeeze()

        threshold = 0.5
        logits = (logits > threshold).to(torch.uint8)

        stage_items.append(
            {
                "image": images[0].squeeze(),
                "mask": masks[0],
                "pred": logits[0],
            }
        )

    def on_stage_epoch_end(self, trainer, pl_module: "pl.LightningModule", stage_items):
        tensorboard = pl_module.logger.experiment
        images = [item["image"].cpu().numpy() for item in stage_items][:5]
        masks = [item["mask"].cpu().numpy() for item in stage_items][:5]
        preds = [item["pred"].cpu().numpy() for item in stage_items][:5]

        images = np.array(images)
        masks = np.array(masks)
        preds = np.array(preds)

        masks = np.expand_dims(masks, 1)
        preds = np.expand_dims(preds, 1)

        tensorboard.add_images("images", images, trainer.current_epoch)
        tensorboard.add_images("preds", preds, trainer.current_epoch)
        tensorboard.add_images("masks", masks, trainer.current_epoch)

    def on_train_batch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        outputs,
        batch: Any,
        batch_idx: int,
        unused: Optional[int] = 0,
    ) -> None:
        self.on_stage_batch_end(outputs, batch, self.train_items)

    def on_train_epoch_end(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:
        self.on_stage_epoch_end(trainer, pl_module, self.train_items)
