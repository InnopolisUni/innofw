# standard libraries
from typing import Dict, Any

# third party libraries
import pytorch_lightning as pl


class BaseLightningModule(pl.LightningModule):
    """
        Class that defines an interface for lightning modules
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.metrics = None

    def setup_up_metrics(self, metrics):
        import hydra
        self.metrics = {
            i['_target_'].split('.')[-1]: hydra.utils.instantiate(i) for i in
            metrics}

    def log_metrics(self, stage, predictions, labels):
        for name, func in self.metrics.items():
            print(predictions, labels)
            value = func(predictions, labels)
            self.log(f"metrics/{stage}/{name}", value)
