# local modules
from ..base_adapter import BaseAdapter


class YOLOV5OptimizerBaseAdapter(BaseAdapter):
    """Class defines adapter interface to conform to YOLOv5 optimizer specifications

    Methods
    -------
    adapt(optimizer: DictConfig) -> dict
        converts the optimizer configuration into YOLOv5 suitable format
    """

    def __init__(self):
        self.possible_values = {"optimizer": ["Adam", "AdamW", "SGD"]}
        self.opt = {"optimizer": "SGD"} # Setting SGD as the default optimizer
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

        return {"opt": self.opt, "hyp": self.hyp}
            

    def from_cfg(self, cfg):
        return {}, {}

    def from_obj(self, obj):
        return {}, {}
