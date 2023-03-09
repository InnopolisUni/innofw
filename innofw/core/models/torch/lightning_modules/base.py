# standard libraries
import logging

import hydra
import pytorch_lightning as pl
from omegaconf import DictConfig

# third party libraries


class BaseLightningModule(pl.LightningModule):
    """
    Class that defines an interface for lightning modules
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.metrics = None

    def on_train_epoch_start(self) -> None:
        self._setup_metrics()

    def _setup_metrics(self):
        try:
            self.metrics = {
                i["_target_"]
                .split(".")[-1]: hydra.utils.instantiate(i)
                .to(self.device)
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
        # instantiate scheduler from configurations
        try:
            scheduler = self.scheduler_cfg(optim)
            # scheduler = hydra.utils.instantiate(self.scheduler_cfg, optim)
            # return optimizers and schedulers
            return [optim], [scheduler]
        except:
            return [optim]
