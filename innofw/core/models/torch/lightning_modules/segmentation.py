# third party libraries
from typing import Any

# from pytorch_lightning import LightningModule
from innofw.core.models.torch.lightning_modules.base import BaseLightningModule
import torch


class SemanticSegmentationLightningModule(
    BaseLightningModule
):
    """
    PyTorchLightning module for Semantic Segmentation task
    ...

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
    @property
    def metric_to_track(self):
        return "val_loss"

    def __init__(
            self,
            model,
            losses,
            optimizer_cfg,
            scheduler_cfg,
            threshold: float = 0.5,
            *args,
            **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.model = model
        self.losses = losses

        self.optimizer_cfg = optimizer_cfg
        self.scheduler_cfg = scheduler_cfg

        self.threshold = threshold

        assert self.losses is not None
        assert self.optimizer_cfg is not None


    def model_load_checkpoint(self, path):
        self.model.load_state_dict(torch.load(path)["state_dict"])

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        """Make a prediction"""
        logits = self.model(batch)
        outs = (logits > self.threshold).to(torch.uint8)
        return outs

    def predict_proba(self, batch: torch.Tensor) -> torch.Tensor:
        """Predict and output probabilities"""
        out = self.model(batch)
        return out

    def training_step(self, batch, batch_idx):
        """Process a batch in a training loop"""
        images, masks = batch["scenes"], batch["labels"]
        logits = self.predict_proba(images)
        # compute and log losses
        total_loss = self.log_losses("train", logits.squeeze(), masks.squeeze())
        self.log_metrics("train", torch.sigmoid(logits).view(-1), masks.to(torch.uint8).squeeze().unsqueeze(1).view(-1))
        return {"loss": total_loss, "logits": logits}

    def validation_step(self, batch, batch_id):
        """Process a batch in a validation loop"""
        images, masks = batch["scenes"], batch["labels"]
        logits = self.predict_proba(images)
        # compute and log losses
        total_loss = self.log_losses("val", logits.squeeze(), masks.squeeze())
        self.log("val_loss", total_loss, prog_bar=True)
        return {"loss": total_loss, "logits": logits}

    def test_step(self, batch, batch_index):
        """Process a batch in a testing loop"""
        images = batch["scenes"]

        preds = self.forward(images)
        return {"preds": preds}

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Any:
        if isinstance(batch, dict):
            batch = batch['scenes']
        preds = self.forward(batch)
        return preds

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
