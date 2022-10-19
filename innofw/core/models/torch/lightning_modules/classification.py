import pytorch_lightning as pl
from typing import Any

import torch


class ClassificationLightningModule(pl.LightningModule):
    """ """

    def __init__(
        self, model, losses, optimizer_cfg, scheduler_cfg, *args: Any, **kwargs: Any
    ):
        super().__init__(*args, **kwargs)
        self.model = model
        self.losses = losses
        self.optimizer_cfg = optimizer_cfg
        self.scheduler_cfg = scheduler_cfg

    def forward(self, x, *args, **kwargs) -> Any:
        return self.model(x)

    def configure_optimizers(self):
        """Function to set up optimizers and schedulers"""
        # get all trainable model parameters
        params = [x for x in self.model.parameters() if x.requires_grad]
        # instantiate models from configurations
        optim = self.optimizer_cfg(params=params)
        # optim = hydra.utils.instantiate(self.optimizer_cfg, params=params)

        # instantiate scheduler from configurations
        try:
            scheduler = self.scheduler_cfg(optim)
            # scheduler = hydra.utils.instantiate(self.scheduler_cfg, optim)
            # return optimizers and schedulers
            return [optim], [scheduler]
        except:
            return [optim]

    def training_step(self, batch, batch_idx):
        """
        Arguments
            batch - tensor
        """
        image, target = batch
        outputs = self.forward(image.float())
        loss = self.losses(outputs, target.long())
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        image, target = batch
        outputs = self.forward(image.float())
        loss = self.losses(outputs, target.long())
        self.log("val_loss", loss, prog_bar=True)
        return {"val_loss": loss}

    def predict_step(self, batch, batch_idx, **kwargs):
        outputs = self.forward(batch.float())
        outputs = torch.argmax(outputs, 1)
        return outputs

    # def validation_epoch_end(self, outputs):
    #     val_loss_mean = torch.stack([x['val_loss'] for x in outputs]).mean()
    #     logging.info(f'VAL_LOSS:{val_loss_mean}')
    #     return {'val_loss': val_loss_mean}
