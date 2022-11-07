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
        self.opt = {"optimizer": "Adam"}
        self.hyp = {"momentum": 0.937, "weight decay": 5e-4}

    def adapt(self, optimizer) -> dict:
        return {"opt": self.opt, "hyp": self.hyp}

    def from_cfg(self, cfg):
        return {}, {}

    def from_obj(self, obj):
        return {}, {}
