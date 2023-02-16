# standard libraries
from abc import abstractproperty
from typing import Dict, Any
import logging

# third party libraries
import hydra
import pytorch_lightning as pl
from omegaconf import DictConfig
import torch


class BaseLightningModule(pl.LightningModule):
    """
        Class that defines an interface for lightning modules
    """
    @abstractproperty
    def metric_to_track(self) -> str:
        """Literal specifying the metric to track when the reduceonplateau is used
        """
        ...

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.metrics = None

    def on_train_epoch_start(self) -> None:
        self._setup_metrics()

    def _setup_metrics(self):
        self.metrics = {
            i['_target_'].split('.')[-1]: hydra.utils.instantiate(i).to(self.device) for i in
            self.metrics_cfg}

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
        
        if self.scheduler_cfg is None:
            return [optim]
        else:
            # instantiate scheduler from configurations
            try:
                if isinstance(self.optimizer_cfg, DictConfig):
                    scheduler = hydra.utils.instantiate(self.scheduler_cfg, optim)
                else:
                    scheduler = self.scheduler_cfg(optim)
                if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    return {"optimizer": optim, "lr_scheduler": scheduler, "monitor": self.metric_to_track}
                return [optim], [scheduler]
            except Exception as e:
                logging.warning(f"Unable to instantiate lr scheduler, running without scheduler. Error is: {e}")
                # raise NotImplementedError()
