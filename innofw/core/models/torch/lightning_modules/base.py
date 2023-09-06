# standard libraries
import logging
from abc import abstractproperty

import hydra
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig

# third party libraries


class BaseLightningModule(pl.LightningModule):
    """
    Class that defines an interface for lightning modules
    """

    @abstractproperty
    def metric_to_track(self) -> str:
        """Literal specifying the metric to track when the reduceonplateau is used"""
        ...

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.metrics = None

    def on_train_epoch_start(self) -> None:
        self._setup_metrics()

    def _setup_metrics(self):
        try:
            self.metrics = {
                i["_target_"].split(".")[-1]: hydra.utils.instantiate(i).to(self.device)
                for i in self.metrics_cfg
            }
        except AttributeError:
            logging.warning("no metrics provided")

    def setup_metrics(self, metrics):
        self.metrics_cfg = metrics

    def log_metrics(self, stage, predictions, labels):
        if self.metrics is None:
            return
        for name, func in self.metrics.items():
            value = func(predictions, labels)
            self.log(f"metrics/{stage}/{name}", value)

    def configure_optimizers(self):
        """Function to set up optimizers and schedulers"""
        # get all trainable model parameters
        params = [x for x in self.parameters() if x.requires_grad]
        # instantiate models from configurations
        if isinstance(self.optimizer_cfg, DictConfig):
            optim = hydra.utils.instantiate(self.optimizer_cfg, params=params)
        else:
            optim = self.optimizer_cfg(params=params)

        if self.scheduler_cfg is None or self.scheduler_cfg is {}:
            return [optim]
        else:
            # instantiate scheduler from configurations
            if isinstance(self.scheduler_cfg, DictConfig):
                scheduler = hydra.utils.instantiate(self.scheduler_cfg, optim)
            else:
                scheduler = self.scheduler_cfg(optimizer=optim)
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                return {
                    "optimizer": optim,
                    "lr_scheduler": scheduler,
                    "monitor": self.metric_to_track,
                }
            return [optim], [scheduler]
