# local modules
from ..base_adapter import BaseAdapter


class UltralyticsOptimizerBaseAdapter(BaseAdapter):
    """Class defines adapter interface to conform to Ultralytics optimizer specifications

    Methods
    -------
    adapt(optimizer: DictConfig) -> dict
        converts the optimizer configuration into Ultralytics suitable format
    """

    def __init__(self):
        self.possible_values = {"optimizer": ["Adam", "AdamW", "SGD"]}
        self.opt = {"optimizer": "SGD"}  # Setting SGD as the default optimizer
        self.hyp = {"momentum": 0.937, "weight_decay": 5e-4}

    def adapt(self, optimizer) -> dict:
        if optimizer is None:
            return {"opt": self.opt, "hyp": self.hyp}

        if optimizer._target_.lower().endswith("adam"):
            self.opt = {"optimizer": "Adam"}
        elif optimizer._target_.lower().endswith("adamw"):
            self.opt = {"optimizer": "AdamW"}
        elif optimizer._target_.lower().endswith("sgd"):
            self.opt = {"optimizer": "SGD"}

        if "lr0" in optimizer:
            self.hyp["lr0"] = optimizer.lr0
        if "lrf" in optimizer:
            self.hyp["lrf"] = optimizer.lrf

        return {"opt": self.opt, "hyp": self.hyp}

    def from_cfg(self, cfg):
        return {}, {}

    def from_obj(self, obj):
        return {}, {}
